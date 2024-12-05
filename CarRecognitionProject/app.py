from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import base64

# Code below was generated using ChatGPT
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})#update to match frontend url
socketio = SocketIO(app, cors_allowed_origins="*")

model_path = 'models/yolov8m.pt' #make sure to put model in model folder
model = YOLO(model_path)  # Load YOLOv8 model
# Model from https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

current_url = None
capture_thread = None  # Store the thread object
stop_event = threading.Event()  # Event to stop the thread


@app.route('/')
def index():
    return "Flask Server for YOLOv8 Stream"


@app.route('/count-cars', methods=['POST'])
def start_counting():
    global current_url, capture_thread, stop_event

    video_url = request.json.get('url')
    if not video_url:
        return jsonify({'error': 'URL parameter is required'}), 400

    # If there's a new URL, reset the stop_event and update the current_url
    if video_url != current_url:
        current_url = video_url

        if capture_thread and capture_thread.is_alive():
            stop_event.set()  # Signal the thread to stop
            capture_thread.join()  # Wait for the previous thread to finish

        stop_event.clear()  # Clear the stop event before starting a new thread

        def generate_frames():
            # driver = webdriver.Chrome(options=chrome_options)
            driver = webdriver.Remote("http://selenium:4444/wd/hub", options=chrome_options)
            driver.get(video_url)
            time.sleep(5)  # Wait for the page to load

            # Attempt to auto-play the video by clicking the play button
            driver.execute_script("""
                var video = document.querySelector('video');
                if (video) {
                    video.muted = true;
                    video.play();
                } else {
                    var playButton = document.querySelector('button');
                    if (playButton) {
                        playButton.click();
                    }
                }
                // Keep the video playing
                setInterval(() => {
                    if (video.paused) {
                        video.play();
                    }
                }, 10000);  // Check every 10 seconds
            """)

            frame_count = 0

            while not stop_event.is_set():  # Check if the stop event is set
                screenshot = driver.get_screenshot_as_png()
                frame = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)

                # Crop a larger region around the video
                height, width, _ = frame.shape
                larger_region = frame[0:height // 1, 0:width // 2]  # Adjust as needed to capture the desired region

                results = model(larger_region)
                car_count = 0
                boxes = []

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])  # Get the class of the detected object
                        confidence = box.conf[0]  # Get the confidence of the detected object
                        if confidence < 0.25:  # Only count if confidence >= 0.25
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extract coordinates as a list
                        if cls == 2:  # Car
                            label = f"Car: {confidence:.2f}"
                        elif cls == 3:  # Motorcycle
                            label = f"Motorcycle: {confidence:.2f}"
                        elif cls == 5:  # Bus
                            label = f"Bus: {confidence:.2f}"
                        elif cls == 7:  # Truck
                            label = f"Truck: {confidence:.2f}"
                        else:
                            continue

                        car_count += 1
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
                        cv2.rectangle(larger_region, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Put label on the bounding box
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(larger_region, (int(x1), int(y1) - label_size[1] - 10),
                                      (int(x1) + label_size[0], int(y1)), (0, 255, 0), cv2.FILLED)
                        cv2.putText(larger_region, label, (int(x1), int(y1) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1)

                _, buffer = cv2.imencode('.jpg', larger_region)
                frame = base64.b64encode(buffer).decode('utf-8')

                frame_count += 1
                #print(f"Emitting frame {frame_count}")
                socketio.emit('frame', {'frame': frame, 'boxes': boxes})
                socketio.emit('car_count', {'car_count': car_count})

                time.sleep(5)  # Capture screenshot every 5 seconds

            driver.quit()

        capture_thread = threading.Thread(target=generate_frames)
        capture_thread.daemon = True
        capture_thread.start()

    return jsonify({'status': 'Counting started'}), 200


if __name__ == '__main__':
    socketio.run(app, debug=True)
