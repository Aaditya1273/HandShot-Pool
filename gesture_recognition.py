import cv2
import mediapipe as mp
import numpy as np
import time


class GestureRecognizer:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Initialize the hand gesture recognition module using MediaPipe.
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # We only track one hand for simplicity
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Gesture states
        self.prev_landmarks = None
        self.gesture_history = []
        self.velocity = np.zeros(2)
        self.last_time = time.time()
        
        # Calibration data
        self.calibration_mode = False
        self.calibration_data = {
            'open_hand': [],
            'closed_fist': [],
            'range_of_motion': {
                'top': 0, 'bottom': 0, 'left': 0, 'right': 0
            }
        }
    
    def process_frame(self, frame):
        """
        Process a video frame and detect hand landmarks.
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with hand landmarks drawn
            landmarks: Detected hand landmarks or None if no hand is detected
            gesture: Detected gesture ('open_hand', 'closed_fist', or None)
            velocity: Hand movement velocity as a 2D vector
        """
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        landmarks = None
        gesture = None
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            
            # Draw hand landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Calculate hand velocity
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            if self.prev_landmarks is not None and dt > 0:
                # Calculate velocity based on the wrist position
                wrist_velocity = (landmarks[0, :2] - self.prev_landmarks[0, :2]) / dt
                # Apply simple low-pass filter for smoothing
                alpha = 0.7
                self.velocity = alpha * wrist_velocity + (1 - alpha) * self.velocity
            
            self.prev_landmarks = landmarks.copy()
            
            # Detect gesture
            gesture = self._detect_gesture(landmarks)
            
            # Update gesture history (for gesture transitions)
            self.gesture_history.append(gesture)
            if len(self.gesture_history) > 10:
                self.gesture_history.pop(0)
            
            # Draw hand information on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If in calibration mode, collect calibration data
            if self.calibration_mode:
                self._collect_calibration_data(landmarks, gesture)
        
        return frame, landmarks, gesture, self.velocity
    
    def _detect_gesture(self, landmarks):
        """
        Detect hand gesture based on landmark positions.
        
        Args:
            landmarks: Hand landmarks detected by MediaPipe
            
        Returns:
            gesture: Detected gesture ('open_hand', 'closed_fist', or None)
        """
        # Calculate fingertip distances from palm center
        palm_center = np.mean(landmarks[[0, 1, 5, 9, 13, 17]], axis=0)
        
        # Get fingertips
        fingertips = landmarks[[4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky
        
        # Calculate distances from palm center to fingertips
        distances = np.linalg.norm(fingertips - palm_center, axis=1)
        
        # Average distance to determine if fingers are extended
        avg_distance = np.mean(distances)
        
        # If we have calibration data, use it for better gesture recognition
        if self.calibration_data['open_hand'] and self.calibration_data['closed_fist']:
            open_hand_dist = np.mean(self.calibration_data['open_hand'])
            closed_fist_dist = np.mean(self.calibration_data['closed_fist'])
            
            # Determine threshold based on calibration
            threshold = (open_hand_dist + closed_fist_dist) / 2
            
            if avg_distance > threshold:
                return 'open_hand'
            else:
                return 'closed_fist'
        else:
            # Default recognition without calibration
            # Simple threshold based on empirical testing
            if avg_distance > 0.1:
                return 'open_hand'
            else:
                return 'closed_fist'
    
    def start_calibration(self):
        """Start the calibration process."""
        self.calibration_mode = True
        self.calibration_data = {
            'open_hand': [],
            'closed_fist': [],
            'range_of_motion': {
                'top': 1.0, 'bottom': 0.0, 'left': 1.0, 'right': 0.0
            }
        }
        print("Calibration started. Please follow the on-screen instructions.")
    
    def end_calibration(self):
        """End the calibration process."""
        self.calibration_mode = False
        print("Calibration completed.")
    
    def _collect_calibration_data(self, landmarks, gesture):
        """
        Collect calibration data for better gesture recognition.
        
        Args:
            landmarks: Hand landmarks
            gesture: Current detected gesture
        """
        # Update range of motion
        wrist_pos = landmarks[0, :2]  # x, y coordinates of wrist
        
        self.calibration_data['range_of_motion']['left'] = min(
            self.calibration_data['range_of_motion']['left'], wrist_pos[0])
        self.calibration_data['range_of_motion']['right'] = max(
            self.calibration_data['range_of_motion']['right'], wrist_pos[0])
        self.calibration_data['range_of_motion']['top'] = min(
            self.calibration_data['range_of_motion']['top'], wrist_pos[1])
        self.calibration_data['range_of_motion']['bottom'] = max(
            self.calibration_data['range_of_motion']['bottom'], wrist_pos[1])
        
        # Calculate fingertip distances from palm center for gesture calibration
        palm_center = np.mean(landmarks[[0, 1, 5, 9, 13, 17]], axis=0)
        fingertips = landmarks[[4, 8, 12, 16, 20]]
        distances = np.linalg.norm(fingertips - palm_center, axis=1)
        avg_distance = np.mean(distances)
        
        # Store the distance data for the appropriate gesture
        if gesture == 'open_hand':
            self.calibration_data['open_hand'].append(avg_distance)
        elif gesture == 'closed_fist':
            self.calibration_data['closed_fist'].append(avg_distance)
    
    def detect_strike(self):
        """
        Detect a striking motion (transition from closed fist to open hand).
        
        Returns:
            strike_power: Power of the strike (0.0 to 1.0) or None if no strike detected
        """
        if len(self.gesture_history) < 3:
            return None
        
        # Check for pattern: closed_fist -> open_hand
        if (self.gesture_history[-3:] == ['closed_fist', 'closed_fist', 'open_hand'] or
            self.gesture_history[-2:] == ['closed_fist', 'open_hand']):
            
            # Calculate strike power based on hand velocity
            speed = np.linalg.norm(self.velocity)
            # Normalize speed to 0-1 range (assuming max speed around 2.0)
            strike_power = min(1.0, speed / 2.0)
            
            # Clear history to prevent multiple detections
            self.gesture_history = []
            
            return strike_power
        
        return None
    
    def get_cue_position(self, frame_width, frame_height):
        """
        Get the normalized position of the hand for cue positioning.
        
        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            position: Normalized position (x, y) in range [0, 1] or None if no hand is detected
        """
        if self.prev_landmarks is None:
            return None
        
        # Use the wrist position for cue positioning
        wrist_pos = self.prev_landmarks[0, :2]  # x, y coordinates
        
        # Normalize based on calibration data if available
        if (self.calibration_data['range_of_motion']['right'] > 
            self.calibration_data['range_of_motion']['left']):
            
            # Use calibration data
            x_normalized = (wrist_pos[0] - self.calibration_data['range_of_motion']['left']) / (
                self.calibration_data['range_of_motion']['right'] - 
                self.calibration_data['range_of_motion']['left'])
            
            y_normalized = (wrist_pos[1] - self.calibration_data['range_of_motion']['top']) / (
                self.calibration_data['range_of_motion']['bottom'] - 
                self.calibration_data['range_of_motion']['top'])
        else:
            # Default normalization without calibration
            x_normalized = wrist_pos[0]
            y_normalized = wrist_pos[1]
        
        # Clamp values to [0, 1] range
        x_normalized = max(0.0, min(1.0, x_normalized))
        y_normalized = max(0.0, min(1.0, y_normalized))
        
        return (x_normalized, y_normalized)
    
    def release(self):
        """Release resources."""
        self.hands.close()
