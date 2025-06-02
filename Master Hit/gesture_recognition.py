import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


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
        self.last_gesture = None
        self.gesture_history = []
        self.gesture_start_time = time.time()
        
        # Landmark smoothing buffers
        self.landmark_history = deque(maxlen=8)
        self.velocity_history = deque(maxlen=4)
        self.position_history = deque(maxlen=5)
        
        # Track all 21 landmarks individually for maximum smoothness
        self.landmark_buffers = [deque(maxlen=10) for _ in range(21)]
        
        # Calibration data
        self.calibration_mode = False
        self.calibration_data = {
            'range_of_motion': {
                'top': 1.0, 'bottom': 0.0, 'left': 1.0, 'right': 0.0
            }
        }
        self.calibration_start_time = 0
    
    def process_frame(self, frame):
        """
        Process a video frame and detect hand landmarks.
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with hand landmarks drawn
            gesture: Detected gesture ('move', 'shoot', 'dodge', or None)
            normalized_position: Normalized hand position (x, y)
        """
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        gesture = None
        normalized_position = None
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            
            # Draw hand landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Apply landmark smoothing
            if self.prev_landmarks is not None:
                # Calculate velocity
                velocity = np.mean(landmarks[:, :2] - self.prev_landmarks[:, :2], axis=0)
                self.velocity_history.append(velocity)
                
                # Apply individual landmark smoothing
                smoothed_landmarks = landmarks.copy()
                
                # Update each landmark buffer
                for i, landmark in enumerate(landmarks):
                    self.landmark_buffers[i].append(landmark)
                    
                    # Apply weighted smoothing if we have enough history
                    if len(self.landmark_buffers[i]) >= 3:
                        # More weight to recent positions
                        weights = np.linspace(0.5, 1.0, len(self.landmark_buffers[i]))
                        weights = weights / weights.sum()  # Normalize
                        
                        # Calculate weighted average
                        buffer_array = np.array(self.landmark_buffers[i])
                        smoothed_landmarks[i] = np.sum(buffer_array * weights[:, np.newaxis], axis=0)
                
                landmarks = smoothed_landmarks
            
            # Store landmarks in history
            self.landmark_history.append(landmarks)
            
            # Update calibration data if in calibration mode
            if self.calibration_mode:
                self._collect_calibration_data(landmarks)
                
                # Check if calibration time is over (5 seconds)
                if time.time() - self.calibration_start_time > 5:
                    self.calibration_mode = False
                    print("Calibration complete!")
            
            # Detect gesture
            gesture = self._detect_gesture(landmarks)
            
            # Get normalized position using index finger tip for pointing
            control_points = np.array([
                landmarks[8, :2],   # Index finger tip (weight: 0.5)
                landmarks[0, :2],   # Wrist (weight: 0.3)
                landmarks[5, :2],   # Index finger MCP (weight: 0.2)
            ])
            
            weights = np.array([0.5, 0.3, 0.2])
            weighted_position = np.sum(control_points * weights[:, np.newaxis], axis=0)
            normalized_position = self._get_normalized_position(weighted_position)
            
            # Store position in history for smoothing
            self.position_history.append(normalized_position)
            
            # Update gesture history
            if self.last_gesture != gesture:
                self.gesture_start_time = time.time()
            self.last_gesture = gesture
            
            self.gesture_history.append(gesture)
            if len(self.gesture_history) > 10:
                self.gesture_history.pop(0)
            
            # Draw gesture and position info
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if normalized_position:
                cv2.putText(frame, f"Position: ({normalized_position[0]:.2f}, {normalized_position[1]:.2f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.prev_landmarks = landmarks
        
        # Draw calibration status
        if self.calibration_mode:
            cv2.putText(frame, "CALIBRATING: Move hand around screen", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Draw progress bar
            progress = min(1.0, (time.time() - self.calibration_start_time) / 5)
            bar_width = int(frame.shape[1] * progress)
            cv2.rectangle(frame, (0, frame.shape[0] - 10), (bar_width, frame.shape[0]), (0, 255, 0), -1)
        
        return frame, gesture, normalized_position
    
    def _detect_gesture(self, landmarks):
        """
        Detect hand gesture based on landmark positions.
        
        Args:
            landmarks: Hand landmarks detected by MediaPipe
            
        Returns:
            gesture: Detected gesture ('move', 'shoot', 'dodge')
        """
        # Get finger status (extended or curled)
        finger_status = self._get_finger_status(landmarks)
        
        # Count extended fingers
        extended_count = sum(finger_status)
        
        # Prevent gesture flickering by requiring consistent gestures
        if len(self.gesture_history) >= 3:
            # If the last 3 gestures were the same, maintain that gesture
            if len(set(self.gesture_history[-3:])) == 1:
                # But still allow transitioning from shoot to move (for quick recovery)
                if self.gesture_history[-1] == 'shoot' and extended_count >= 3:
                    return 'move'
                # Otherwise maintain the stable gesture
                return self.gesture_history[-1]
        
        # Master Hit specific gestures:
        # 1. Move (open hand or at least 3 fingers extended) - for moving the eye
        if extended_count >= 3:
            return 'move'
        
        # 2. Shoot (only index finger extended, others curled) - for shooting
        if finger_status[1] and not finger_status[2] and not finger_status[3] and not finger_status[4]:
            return 'shoot'
        
        # 3. Dodge (closed fist) - for dodging attacks or collecting power-ups
        if extended_count <= 1:
            return 'dodge'
        
        # Default to move gesture
        return 'move'
    
    def _collect_calibration_data(self, landmarks):
        """
        Collect calibration data for better gesture recognition.
        
        Args:
            landmarks: Hand landmarks
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
    
    def _get_normalized_position(self, position):
        """
        Get normalized position based on calibration data with enhanced smoothing.
        
        Args:
            position: Raw position (x, y)
            
        Returns:
            Normalized position (x, y) in range [0, 1]
        """
        # If calibration data exists and is valid
        if (self.calibration_data['range_of_motion']['right'] > 
            self.calibration_data['range_of_motion']['left'] and
            self.calibration_data['range_of_motion']['bottom'] > 
            self.calibration_data['range_of_motion']['top']):
            
            # Calculate range width and height
            range_width = self.calibration_data['range_of_motion']['right'] - self.calibration_data['range_of_motion']['left']
            range_height = self.calibration_data['range_of_motion']['bottom'] - self.calibration_data['range_of_motion']['top']
            
            # Ensure we don't divide by zero
            if range_width > 0.001 and range_height > 0.001:
                # Use calibration data with padding to make movement more comfortable
                padding = 0.1  # 10% padding on each side
                
                # Normalize with padding
                x_normalized = (position[0] - (self.calibration_data['range_of_motion']['left'] - range_width * padding)) / (
                    range_width * (1 + 2 * padding))
                
                y_normalized = (position[1] - (self.calibration_data['range_of_motion']['top'] - range_height * padding)) / (
                    range_height * (1 + 2 * padding))
            else:
                # Fallback if range is too small
                x_normalized = position[0]
                y_normalized = position[1]
        else:
            # Default normalization without calibration
            x_normalized = position[0]
            y_normalized = position[1]
        
        # Clamp values to [0, 1] range
        x_normalized = max(0.0, min(1.0, x_normalized))
        y_normalized = max(0.0, min(1.0, y_normalized))
        
        # Apply motion prediction based on velocity
        if len(self.velocity_history) >= 2 and not self.calibration_mode:
            # Get average velocity from recent history
            recent_velocity = np.mean([v for v in self.velocity_history], axis=0)
            
            # Predict position based on velocity
            prediction_factor = 0.12
            predicted_x = x_normalized + recent_velocity[0] * prediction_factor
            predicted_y = y_normalized + recent_velocity[1] * prediction_factor
            
            # Blend current position with prediction
            x_normalized = x_normalized * 0.9 + predicted_x * 0.1
            y_normalized = y_normalized * 0.9 + predicted_y * 0.1
            
            # Clamp again after prediction
            x_normalized = max(0.0, min(1.0, x_normalized))
            y_normalized = max(0.0, min(1.0, y_normalized))
        
        return (x_normalized, y_normalized)
    
    def _get_finger_status(self, landmarks):
        """
        Determine which fingers are extended or curled.
        
        Args:
            landmarks: Hand landmarks detected by MediaPipe
            
        Returns:
            List of booleans indicating if each finger is extended [thumb, index, middle, ring, pinky]
        """
        # Get finger joint positions
        finger_status = [False] * 5  # Default all fingers to curled
        
        # Thumb is special case - use angle-based detection
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Calculate angle at the thumb IP joint
        v1 = thumb_mcp - thumb_ip
        v2 = thumb_tip - thumb_ip
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        finger_status[0] = angle > 1.7  # Threshold to detect thumb extension
        
        # For other fingers, use both distance and angle for more accurate detection
        # Define finger landmarks
        fingertips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        pips = [6, 10, 14, 18]       # Second joints
        mcps = [5, 9, 13, 17]        # Knuckles
        
        # Check each finger
        for i, (tip, pip, mcp) in enumerate(zip(fingertips, pips, mcps)):
            # Method 1: Height-based detection (traditional)
            height_extended = landmarks[tip, 1] < landmarks[pip, 1]
            
            # Method 2: Angle-based detection (more accurate)
            # Calculate angle at PIP joint
            v1 = landmarks[mcp] - landmarks[pip]
            v2 = landmarks[tip] - landmarks[pip]
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            angle_extended = angle > 2.0  # Finger is straight if angle is large
            
            # Method 3: Distance-based detection
            # Compare distance from fingertip to wrist vs from knuckle to wrist
            tip_to_wrist = np.linalg.norm(landmarks[tip] - landmarks[0])
            mcp_to_wrist = np.linalg.norm(landmarks[mcp] - landmarks[0])
            distance_ratio = tip_to_wrist / mcp_to_wrist
            distance_extended = distance_ratio > 1.5
            
            # Combine methods - finger is extended if at least 2 methods agree
            methods = [height_extended, angle_extended, distance_extended]
            finger_status[i+1] = sum(methods) >= 2
        
        return finger_status
    
    def start_calibration(self):
        """Start the calibration process."""
        print("Calibration started. Move your hand around the screen for 5 seconds...")
        self.calibration_mode = True
        self.calibration_data = {
            'range_of_motion': {
                'top': 1.0, 'bottom': 0.0, 'left': 1.0, 'right': 0.0
            }
        }
        self.calibration_start_time = time.time()
    
    def release(self):
        """Release resources."""
        self.hands.close()
