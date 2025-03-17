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

gpio.setwarnings(False)
gpio.setmode(gpio.BCM)
gpio.setup(17, gpio.IN, pull_up_down=gpio.PUD_DOWN)

# Global variables 
threshold = 50  # 50 cm
frame_counter = 0
model = YOLO('yolov8n.pt')
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 100)

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
    engine.say(text)
    engine.runAndWait()
    

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

    # Extract detected class names
    detected_classes = [model.names[int(box.cls)] for result in results for box in result.boxes]
    
    # Count occurrences of each class
    object_counts = Counter(detected_classes)
    
    if not object_counts:
       return "No objects detected."

    # Create a spoken sentence
    object_text = ", ".join(f"{count} {name}{'s' if count > 1 else ''}" for name, count in object_counts.items())
    return object_text

def run_CV_script(camera):
    """ Run the object detection script, ensuring only one camera is open at a time """
    global frame_counter

    # Close the Arducam before opening USB camera
    camera.stop()
    camera.close()
    print("Arducam closed. Opening USB camera...")

    img_path = f"frame_{frame_counter}.jpg"
    capture_img_USB(img_path)
    
    results = model.predict(source=img_path, save=True, save_dir=output_dir)
    print(f"Processed frame {frame_counter}. Output saved to {output_dir}")

    # Process results and generate audio
    detected_text = process_yolo_results(results)
    print(detected_text)
    save_text_to_wav(detected_text, f"detected_objects_{frame_counter}.wav")

    frame_counter += 1

    # Reopen the Arducam
    print("Reopening Arducam...")
    ret = camera.open(ac.Connection.CSI, 0)
    if ret != 0:
        print("Failed to reopen Arducam. Error code:", ret)
        return
    camera.start(ac.FrameType.DEPTH)

########################## Main ########################## 

def main():
    camera = ac.ArducamCamera()
    cfg_path = None

    ret = camera.open(ac.Connection.CSI, 0)
    if ret != 0:
        print("Failed to open camera. Error code:", ret)
        return
    else:
        print("open")

    ret = camera.start(ac.FrameType.DEPTH)
    if ret != 0:
        print("Failed to start camera. Error code:", ret)
        camera.close()
        return

    try:
        while True:
            # Capture depth frame
            depth_frame = camera.requestFrame(200)  # Timeout of 200ms
            
            if depth_frame is not None:
                depth_map = np.array(depth_frame.depth_data, dtype=np.float32)
                _, width = depth_map.shape
                left_depth = depth_map[:, :width // 2]
                right_depth = depth_map[:, width // 2:]

                too_close_left = np.any((left_depth > 0) & (left_depth < threshold))
                too_close_right = np.any((right_depth > 0) & (right_depth < threshold))
                dist = find_closest_dist(depth_map)

                if too_close_right:
                    text = f"Careful, something is on your right, approximately {dist} centimeters away"
                    save_text_to_wav(text, "object_right.wav")
                elif too_close_left:
                    text = f"Careful, something is on your left, approximately {dist} centimeters away"
                    save_text_to_wav(text, "object_left.wav")
                else:
                    print("No close objects")

                camera.releaseFrame(depth_frame)

            if gpio.input(17) == gpio.HIGH:
                run_CV_script(camera)

    except KeyboardInterrupt:
        print("Stopping camera...")
    finally:
        camera.stop()
        camera.close()

if __name__ == "__main__":
    main()
