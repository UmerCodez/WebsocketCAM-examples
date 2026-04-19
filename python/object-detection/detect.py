import websocket
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from threading import Thread, Lock
import time
import os
import urllib.request
from collections import deque

class ObjectDetector:
    def __init__(self, websocket_url, max_queue_size=2):
        self.websocket_url = websocket_url
        self.ws = None
        self.running = False
        self.max_queue_size = max_queue_size
        
        # Thread-safe frame queue - keep only latest frames
        self.frame_queue = deque(maxlen=max_queue_size)
        self.queue_lock = Lock()
        
        # Download object detector model if not exists
        model_path = 'efficientdet_lite0.tflite'
        if not os.path.exists(model_path):
            print("Downloading object detection model...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite'
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")
        
        # Initialize MediaPipe Object Detector
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            max_results=10,  # Maximum number of objects to detect
            score_threshold=0.5,  # Minimum confidence threshold
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        
        # Color palette for different object classes
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
            (128, 0, 255), (0, 128, 255)
        ]
        
        # For FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
    
    def draw_detection(self, frame, detection):
        """Draw bounding box and label for detected object"""
        h, w, _ = frame.shape
        
        # Get bounding box
        bbox = detection.bounding_box
        x_min = int(bbox.origin_x)
        y_min = int(bbox.origin_y)
        x_max = int(bbox.origin_x + bbox.width)
        y_max = int(bbox.origin_y + bbox.height)
        
        # Get category and confidence
        category = detection.categories[0]
        class_name = category.category_name
        confidence = category.score
        
        # Choose color based on class name hash
        color = self.colors[hash(class_name) % len(self.colors)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get label size for background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x_min, y_min - label_height - 10),
            (x_min + label_width + 10, y_min),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x_min + 5, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    def on_message(self, ws, message):
        """Callback when receiving JPEG frame from WebSocket"""
        try:
            self.received_frames += 1
            
            # Add frame to queue (automatically drops oldest if full)
            with self.queue_lock:
                if len(self.frame_queue) >= self.max_queue_size:
                    self.dropped_frames += 1
                self.frame_queue.append(message)
                
        except Exception as e:
            print(f"Error receiving frame: {e}")
    
    def process_frames(self):
        """Process frames from queue in a loop"""
        while self.running:
            try:
                # Get frame from queue
                with self.queue_lock:
                    if not self.frame_queue:
                        time.sleep(0.001)
                        continue
                    message = self.frame_queue.popleft()
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(message, np.uint8)
                
                # Decode JPEG image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to MediaPipe Image format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect objects
                detection_result = self.detector.detect(mp_image)
                
                # Draw detections on the frame
                if detection_result.detections:
                    for detection in detection_result.detections:
                        self.draw_detection(frame, detection)
                
                self.processed_frames += 1
                
                # Calculate FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Display stats on frame
                cv2.putText(frame, f"Processing FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                num_objects = len(detection_result.detections) if detection_result.detections else 0
                cv2.putText(frame, f"Objects: {num_objects}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show queue status
                with self.queue_lock:
                    queue_size = len(self.frame_queue)
                cv2.putText(frame, f"Queue: {queue_size}/{self.max_queue_size}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Show dropped frames
                if self.dropped_frames > 0:
                    cv2.putText(frame, f"Dropped: {self.dropped_frames}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow('Object Detection', frame)
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
    
    def on_error(self, ws, error):
        """Callback when WebSocket error occurs"""
        print(f"WebSocket Error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Callback when WebSocket connection closes"""
        print(f"WebSocket closed. Status: {close_status_code}, Message: {close_msg}")
        self.running = False
    
    def on_open(self, ws):
        """Callback when WebSocket connection opens"""
        print("WebSocket connection established")
        self.running = True
    
    def start(self):
        """Start the WebSocket client"""
        print(f"Connecting to {self.websocket_url}...")
        self.running = True
        
        # Start frame processing thread
        process_thread = Thread(target=self.process_frames, daemon=True)
        process_thread.start()
        
        # Create WebSocket connection with timeout
        self.ws = websocket.WebSocketApp(
            self.websocket_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Run WebSocket with timeout
        try:
            self.ws.run_forever(
                ping_interval=30,
                ping_timeout=10,
                reconnect=5  # Attempt to reconnect every 5 seconds
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.stop()
        except Exception as e:
            print(f"Connection error: {e}")
            self.stop()
    
    def stop(self):
        """Stop the WebSocket client and cleanup"""
        print("Stopping...")
        self.running = False
        if self.ws:
            self.ws.close()
        self.detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration
    WEBSOCKET_URL = "ws://192.168.18.50:8080"
    
    # Queue size: Lower = more responsive but more dropped frames
    # Higher = smoother but more lag
    # Recommended: 1-3 for 30fps stream
    MAX_QUEUE_SIZE = 2
    
    print("=" * 60)
    print("Object Detection - Real-time Mode")
    print("=" * 60)
    print(f"Stream: {WEBSOCKET_URL}")
    print(f"Queue size: {MAX_QUEUE_SIZE} frames")
    print(f"Model: EfficientDet-Lite0 (COCO dataset - 80 classes)")
    print("Press 'q' in the video window to quit")
    print("=" * 60)
    
    detector = ObjectDetector(WEBSOCKET_URL, max_queue_size=MAX_QUEUE_SIZE)
    detector.start()
