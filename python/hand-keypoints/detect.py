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

class HandKeypointDetector:
    def __init__(self, websocket_url, max_queue_size=2):
        self.websocket_url = websocket_url
        self.ws = None
        self.running = False
        self.max_queue_size = max_queue_size
        
        # Thread-safe frame queue - keep only latest frames
        self.frame_queue = deque(maxlen=max_queue_size)
        self.queue_lock = Lock()
        
        # Download hand landmarker model if not exists
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")
        
        # Initialize MediaPipe Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Define hand connections (21 landmarks, connections between them)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 17)  # Palm
        ]
        
        # For FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.received_frames = 0
        self.processed_frames = 0
        self.dropped_frames = 0
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks and connections on the frame"""
        h, w, _ = frame.shape
        
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = hand_landmarks[start_idx]
            end_landmark = hand_landmarks[end_idx]
            
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            
            # Draw line between landmarks
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for idx, landmark in enumerate(hand_landmarks):
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            # Different colors for different finger parts
            if idx in [0]:  # Wrist
                color = (255, 0, 0)
            elif idx in [1, 2, 3, 4]:  # Thumb
                color = (255, 255, 0)
            elif idx in [5, 6, 7, 8]:  # Index
                color = (0, 255, 255)
            elif idx in [9, 10, 11, 12]:  # Middle
                color = (255, 0, 255)
            elif idx in [13, 14, 15, 16]:  # Ring
                color = (128, 0, 255)
            else:  # Pinky
                color = (0, 128, 255)
            
            # Draw filled circle for each landmark
            cv2.circle(frame, (x, y), 5, color, -1)
            # Draw circle outline
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)
        
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
                
                # Detect hands
                detection_result = self.detector.detect(mp_image)
                
                # Draw hand landmarks on the frame
                if detection_result.hand_landmarks:
                    for hand_landmarks in detection_result.hand_landmarks:
                        self.draw_hand_landmarks(frame, hand_landmarks)
                
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
                
                num_hands = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
                cv2.putText(frame, f"Hands: {num_hands}", (10, 60),
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
                cv2.imshow('Hand Keypoint Detection', frame)
                
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
    print("Hand Keypoint Detection - Real-time Mode")
    print("=" * 60)
    print(f"Stream: {WEBSOCKET_URL}")
    print(f"Queue size: {MAX_QUEUE_SIZE} frames")
    print("Press 'q' in the video window to quit")
    print("=" * 60)
    
    detector = HandKeypointDetector(WEBSOCKET_URL, max_queue_size=MAX_QUEUE_SIZE)
    detector.start()
