import cv2
import numpy as np
import ArducamDepthCamera as ac
import pyttsx3
import wave
import tempfile
import RPi.GPIO as gpio
from ultralytics import YOLO
import os
from collections import Counter
import threading
import osmnx as ox
import networkx as nx
import math
import time

gpio.setwarnings(False)
gpio.setmode(gpio.BCM)
gpio.setup(17, gpio.IN, pull_up_down=gpio.PUD_UP)

# Global variables 
threshold = 50  # 50 cm
frame_counter = 0
model = YOLO('yolov8n.pt')
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 100)

# Walking speed (meters per second)
AVERAGE_WALKING_SPEED = 0.89 

########################## Functions ########################## 

def save_text_to_wav(text, filename="output.wav"):
    """ Convert text to speech and save it as a WAV file """
    print(f"Saving '{text}' as {filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        temp_filename = temp_wav.name
        engine.save_to_file(text, temp_filename)
        engine.runAndWait()

    with wave.open(temp_filename, 'rb') as original_wav, wave.open(filename, 'wb') as new_wav:
        new_wav.setnchannels(original_wav.getnchannels())
        new_wav.setsampwidth(original_wav.getsampwidth())
        new_wav.setframerate(original_wav.getframerate())
        new_wav.writeframes(original_wav.readframes(original_wav.getnframes()))

    print(f"Audio saved as {filename}")

def find_closest_dist(depth_map):
    """ Find the closest point in the depth map """
    valid_depths = depth_map[depth_map > 0]  # Ignore zero values
    return np.min(valid_depths) if valid_depths.size > 0 else None

def capture_img_USB(output_path):
    """ Capture image from USB camera """
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()

def process_yolo_results(results):
    """ Process YOLO predictions to generate a spoken output """
    detected_classes = [model.names[int(box.cls)] for result in results for box in result.boxes]
    object_counts = Counter(detected_classes)
    
    if not object_counts:
        return "No objects detected."

    object_text = ", ".join(f"{count} {name}{'s' if count > 1 else ''}" for name, count in object_counts.items())
    return f"I see {object_text}."

def run_CV_script(camera):
    """ Run object detection in response to a button press """
    global frame_counter

    camera.stop()
    camera.close()
    print("Arducam closed. Opening USB camera...")

    img_path = f"frame_{frame_counter}.jpg"
    capture_img_USB(img_path)
    
    results = model.predict(source=img_path, save=True, save_dir=output_dir)
    print(f"Processed frame {frame_counter}. Output saved to {output_dir}")

    detected_text = process_yolo_results(results)
    print(detected_text)
    save_text_to_wav(detected_text, f"detected_objects_{frame_counter}.wav")

    frame_counter += 1

    print("Reopening Arducam...")
    ret = camera.open(ac.Connection.CSI, 0)
    if ret != 0:
        print("Failed to reopen Arducam. Error code:", ret)
        return
    camera.start(ac.FrameType.DEPTH)

########################## Navigation ##########################

def calculate_bearing(lat1, lon1, lat2, lon2):
    """ Calculate the direction (bearing) from one point to another """
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360

def get_direction(bearing):
    """ Convert bearing angle to a compass direction """
    directions = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    index = round(bearing / 45) % 8
    return directions[index]

def generate_directions(graph, route):
    """ Generate spoken navigation directions """
    directions = []
    total_distance = 0

    for i in range(1, len(route)):
        start_node = route[i - 1]
        end_node = route[i]
        edge = graph[start_node][end_node][0]

        start_lat, start_lon = graph.nodes[start_node]['y'], graph.nodes[start_node]['x']
        end_lat, end_lon = graph.nodes[end_node]['y'], graph.nodes[end_node]['x']

        bearing = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
        direction = get_direction(bearing)

        street_name = edge.get('name', 'Unnamed street')
        distance = edge['length']
        total_distance += distance

        directions.append((f"Head {direction} on {street_name} for {distance:.0f} meters", distance))

    directions.append((f"You have reached your destination. Total distance: {total_distance:.0f} meters", 0))
    return directions

def navigation_thread():
    """ Runs navigation instructions in parallel without blocking other operations """
    start_lat, start_lon = 47.65334, -122.30568  # Example: Paul Allen School
    end_lat, end_lon = 47.65820, -122.31346  # Example: The Hub

    graph = ox.load_graphml('gps/seattle_graph.graphml')
    start_node = ox.distance.nearest_nodes(graph, start_lon, start_lat)
    end_node = ox.distance.nearest_nodes(graph, end_lon, end_lat)
    route = nx.shortest_path(graph, start_node, end_node, weight='length')

    directions = generate_directions(graph, route)

    for i, (step, step_dist) in enumerate(directions):
        save_text_to_wav(step, f"nav_step_{i}.wav")

        # Run non-blocking delay for walking speed
        end_time = time.time() + (step_dist / AVERAGE_WALKING_SPEED) if step_dist > 0 else 0
        # while time.time() < end_time:
        #     time.sleep(0.1)

########################## Main ########################## 

def main():
    camera = ac.ArducamCamera()
    if camera.open(ac.Connection.CSI, 0) != 0:
        print("Failed to open camera.")
        return
    if camera.start(ac.FrameType.DEPTH) != 0:
        print("Failed to start camera.")
        camera.close()
        return

    threading.Thread(target=navigation_thread, daemon=True).start()  # Start navigation in the background

    try:
        while True:
            if gpio.input(17) == gpio.LOW:
                run_CV_script(camera)
    except KeyboardInterrupt:
        print("Stopping camera...")
    finally:
        camera.stop()
        camera.close()

if __name__ == "__main__":
    main()
