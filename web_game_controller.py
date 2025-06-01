import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import math
from collections import deque

# Disable pyautogui fail-safe (optional, but prevents issues if hand moves too fast)
pyautogui.FAILSAFE = False

class GestureGameController:
    def __init__(self):
        """Initialize the hand gesture controller for web-based games."""
        # Initialize MediaPipe Hands with enhanced settings for maximum tracking points
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for simplicity
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.5,   # Lower threshold for better tracking
            model_complexity=1             # Higher complexity for better accuracy
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Custom drawing specs for more visible landmarks
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=4)
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=2, circle_radius=2)
        
        # Screen dimensions for mapping
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture states with enhanced tracking
        self.prev_landmarks = None
        self.last_gesture = None
        self.gesture_history = []
        self.gesture_start_time = time.time()
        
        # Landmark smoothing buffers (optimized for responsiveness)
        self.landmark_history = deque(maxlen=8)  # Reduced buffer size for faster response
        self.velocity_history = deque(maxlen=4)  # Reduced velocity history
        self.acceleration = np.zeros(2)  # For motion prediction
        
        # Track all 21 landmarks individually for maximum smoothness
        self.landmark_buffers = [deque(maxlen=10) for _ in range(21)]  # Reduced buffer size
        
        # Enhanced smoothing parameters
        self.smoothing_factor = 0.25  # Reduced for better responsiveness
        self.prev_mouse_pos = pyautogui.position()
        self.position_history = deque(maxlen=5)  # Reduced for faster response
        self.cursor_weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # More weight to recent positions
        
        # Calibration data
        self.calibration_mode = True
        self.calibration_data = {
            'range_of_motion': {
                'top': 1.0, 'bottom': 0.0, 'left': 1.0, 'right': 0.0
            }
        }
        
        # Game control state
        self.is_aiming = False
        self.aim_start_pos = None
        self.power_adjustment = False
        self.strike_cooldown = 0
        
        # Debug info
        self.show_debug = True
    
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
    
    def process_frame(self, frame):
        """
        Process a video frame and detect hand landmarks with enhanced tracking.
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with hand landmarks drawn
            gesture: Detected gesture ('open_hand', 'closed_fist', 'pointing', or None)
            normalized_position: Normalized (x, y) position of hand
        """
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        gesture = None
        normalized_position = None
        velocity = np.zeros(2)
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            
            # Draw hand landmarks on the frame with enhanced styling and more visible points
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.landmark_drawing_spec,
                self.connection_drawing_spec
            )
            
            # Draw additional tracking points for better visualization
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                # Draw larger circles at key points (fingertips, knuckles)
                if idx in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                elif idx in [0, 5, 9, 13, 17]:  # Knuckles
                    cv2.circle(frame, (x, y), 6, (255, 0, 255), -1)
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Apply enhanced landmark smoothing using individual landmark history
            if self.prev_landmarks is not None:
                # Calculate velocity with more precision
                velocity = np.mean(landmarks[:, :2] - self.prev_landmarks[:, :2], axis=0)
                self.velocity_history.append(velocity)
                
                # Calculate acceleration for motion prediction
                if len(self.velocity_history) >= 2:
                    current_velocity = self.velocity_history[-1]
                    prev_velocity = self.velocity_history[-2]
                    self.acceleration = current_velocity - prev_velocity
                
                # Apply individual landmark smoothing for maximum precision
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
                    print(f"Range of motion: Top={self.calibration_data['range_of_motion']['top']:.2f}, "
                          f"Bottom={self.calibration_data['range_of_motion']['bottom']:.2f}, "
                          f"Left={self.calibration_data['range_of_motion']['left']:.2f}, "
                          f"Right={self.calibration_data['range_of_motion']['right']:.2f}")
            
            # Detect gesture with enhanced detection
            gesture = self._detect_gesture(landmarks)
            
            # Get normalized position using more control points for ultra-smooth tracking
            # Use a weighted average of multiple hand points
            control_points = np.array([
                landmarks[8, :2],   # Index finger tip (weight: 0.3)
                landmarks[12, :2],  # Middle finger tip (weight: 0.15)
                landmarks[9, :2],   # Middle finger MCP (weight: 0.15)
                landmarks[5, :2],   # Index finger MCP (weight: 0.15)
                landmarks[0, :2],   # Wrist (weight: 0.15)
                landmarks[4, :2]    # Thumb tip (weight: 0.1)
            ])
            
            weights = np.array([0.3, 0.15, 0.15, 0.15, 0.15, 0.1])
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
            
            # Draw enhanced debug information
            if self.show_debug:
                # Draw gesture and position info
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if normalized_position:
                    cv2.putText(frame, f"Position: ({normalized_position[0]:.2f}, {normalized_position[1]:.2f})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw finger status
                finger_status = self._get_finger_status(landmarks)
                status_text = "".join(["E" if status else "C" for status in finger_status])
                cv2.putText(frame, f"Fingers: {status_text}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw control points used for cursor
                for i, point in enumerate(control_points):
                    x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"{i+1}", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
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
        Detect hand gesture based on landmark positions with enhanced accuracy.
        Simplified for pool game with focus on aiming, clicking, and pulling back.
        
        Args:
            landmarks: Hand landmarks detected by MediaPipe
            
        Returns:
            gesture: Detected gesture ('aim', 'click', 'pull', 'release')
        """
        # Get finger status (extended or curled)
        finger_status = self._get_finger_status(landmarks)
        
        # Calculate palm direction for pull detection
        wrist = landmarks[0, :2]
        middle_mcp = landmarks[9, :2]
        palm_direction = middle_mcp - wrist
        palm_angle = np.arctan2(palm_direction[1], palm_direction[0])
        
        # Measure hand tilt (for pull back detection)
        index_mcp = landmarks[5, :2]
        pinky_mcp = landmarks[17, :2]
        hand_tilt = index_mcp[1] - pinky_mcp[1]  # Positive when index is lower than pinky
        
        # Use gesture history for stability
        if len(self.gesture_history) > 3:
            from collections import Counter
            most_common = Counter(self.gesture_history[-3:]).most_common(1)[0][0]
            # If we have a consistent non-aim gesture, prioritize it
            if most_common != 'aim' and most_common == self.gesture_history[-1]:
                return most_common
        
        # Detect pull gesture (hand tilted back)
        if hand_tilt > 0.05 and sum(finger_status) <= 2:
            return 'pull'
        
        # Click gesture: closed fist or only thumb extended
        if sum(finger_status) <= 1:
            return 'click'
        
        # Release gesture: open hand after click/pull
        if sum(finger_status) >= 4:
            return 'release'
        
        # Default to aim gesture (for moving the cue)
        return 'aim'
    
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
        
        # Apply motion prediction based on velocity (not acceleration)
        if len(self.velocity_history) >= 2 and not self.calibration_mode:
            # Get average velocity from recent history
            recent_velocity = np.mean([v for v in self.velocity_history], axis=0)
            
            # Predict position based on velocity (more stable than acceleration)
            prediction_factor = 0.12  # Reduced prediction factor for stability
            predicted_x = x_normalized + recent_velocity[0] * prediction_factor
            predicted_y = y_normalized + recent_velocity[1] * prediction_factor
            
            # Blend current position with prediction (more weight to current position)
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
        # For each finger, we need the tip, pip (middle joint), and mcp (base joint)
        finger_status = [False] * 5  # Default all fingers to curled
        
        # Thumb is special case - compare distance from tip to palm vs distance from base to palm
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        thumb_cmc = landmarks[1]
        wrist = landmarks[0]
        
        # Calculate angle at the thumb IP joint
        v1 = thumb_mcp - thumb_ip
        v2 = thumb_tip - thumb_ip
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        finger_status[0] = angle > 2.0  # Thumb is extended if angle is large enough
        
        # For other fingers, check if tip is extended beyond PIP joint
        # Index finger
        if landmarks[8, 1] < landmarks[6, 1]:  # Y-coordinate comparison (lower value is higher up)
            finger_status[1] = True
            
        # Middle finger
        if landmarks[12, 1] < landmarks[10, 1]:
            finger_status[2] = True
            
        # Ring finger
        if landmarks[16, 1] < landmarks[14, 1]:
            finger_status[3] = True
            
        # Pinky finger
        if landmarks[20, 1] < landmarks[18, 1]:
            finger_status[4] = True
            
        return finger_status
    
    def control_game(self, gesture, normalized_position):
        """
        Control the 8-ball pool game with simplified gestures optimized for the game mechanics.
        
        Args:
            gesture: Detected gesture ('aim', 'click', 'pull', 'release')
            normalized_position: Normalized hand position (x, y)
        """
        if normalized_position is None:
            return
        
        # Decrease strike cooldown if it's active
        if self.strike_cooldown > 0:
            self.strike_cooldown -= 1
            return
        
        # Apply lighter weighted average smoothing to position using history
        if len(self.position_history) >= 5:  # Reduced from 10 to 5 for faster response
            # Calculate weighted average position with more weight to recent positions
            positions = np.array(list(self.position_history))
            weights = np.linspace(0.6, 1.5, len(positions))  # More weight to recent positions
            weights = weights / weights.sum()  # Normalize
            weighted_position = np.sum(positions * weights[:, np.newaxis], axis=0)
            normalized_position = weighted_position
        
        # Map normalized position to screen coordinates
        target_x = int(normalized_position[0] * self.screen_width)
        target_y = int(normalized_position[1] * self.screen_height)
        
        # Apply smoother cursor movement
        current_x, current_y = pyautogui.position()
        
        # Calculate distance to target
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        # Adaptive smoothing - less smoothing for small movements, more for large ones
        # Reduced overall smoothing to make cursor more responsive
        adaptive_factor = min(0.5, max(0.05, distance / 500))
        smooth_factor = 0.25 * (1 - adaptive_factor)  # Reduced base smoothing factor
        
        # Final smoothing - more direct control
        smooth_x = int(current_x + (target_x - current_x) * (1 - smooth_factor))
        smooth_y = int(current_y + (target_y - current_y) * (1 - smooth_factor))
        
        # Use majority voting for more stable gesture detection
        stable_gesture = gesture
        if len(self.gesture_history) >= 5:
            from collections import Counter
            gesture_counts = Counter(self.gesture_history[-5:])
            most_common = gesture_counts.most_common(1)[0]
            if most_common[1] >= 3:  # If the most common gesture appears at least 3 times in last 5
                stable_gesture = most_common[0]
        
        # Display current gesture on frame
        if self.show_debug and hasattr(self, 'last_frame'):
            cv2.putText(self.last_frame, f"Gesture: {stable_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Always move the cursor for better responsiveness, regardless of gesture
        pyautogui.moveTo(smooth_x, smooth_y)
        
        # Handle pool game specific gestures
        if stable_gesture == 'aim':
            # If we were in pull mode and now we're aiming, release the mouse
            if self.is_aiming:
                pyautogui.mouseUp()
                self.is_aiming = False
                self.power_adjustment = False
        
        elif stable_gesture == 'click':
            # Click and hold to start aiming the cue
            if not self.is_aiming and not self.power_adjustment and self.last_gesture != 'click':
                pyautogui.mouseDown()
                self.is_aiming = True
                self.aim_start_pos = (smooth_x, smooth_y)
                self.strike_cooldown = 5
        
        elif stable_gesture == 'pull':
            # Pull back to adjust power (drag while holding)
            if self.is_aiming:
                self.power_adjustment = True
        
        elif stable_gesture == 'release':
            # Release to strike the ball
            if (self.is_aiming or self.power_adjustment) and self.last_gesture != 'release':
                pyautogui.mouseUp()
                self.is_aiming = False
                self.power_adjustment = False
                self.strike_cooldown = 15
        
        # Store the last frame for debug overlay
        if hasattr(self, 'last_frame'):
            # Draw aim/pull status
            status_text = "AIMING" if self.is_aiming else "READY"
            status_text = "PULLING" if self.power_adjustment else status_text
            cv2.putText(self.last_frame, status_text, (10, self.last_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    def toggle_debug(self):
        """Toggle debug information display."""
        self.show_debug = not self.show_debug
    
    def release(self):
        """Release resources."""
        self.hands.close()


def main():
    # Initialize the controller
    controller = GestureGameController()
    
    # Initialize the webcam with higher resolution for better tracking
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Higher resolution
    cap.set(cv2.CAP_PROP_FPS, 30)           # Higher frame rate if possible
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("\n=== 8-Ball Billiards Web Game Hand Gesture Controller - SIMPLIFIED ===")
    print("1. Open your web browser and navigate to: https://www.crazygames.com/game/8-ball-billiards-classic")
    print("2. Position the game window so it's visible alongside this controller window")
    print("3. Use the following SIMPLIFIED gestures to control the pool game:")
    print("   - AIM: Move your hand with fingers partially extended to aim the cue")
    print("   - CLICK: Make a fist to click and hold the mouse button")
    print("   - PULL: Tilt your hand back while keeping a fist to pull back the cue")
    print("   - RELEASE: Open your hand fully to release and strike the ball")
    print("4. Press 'C' to recalibrate, 'D' to toggle debug info, '+/-' to adjust sensitivity, 'Q' to quit\n")
    
    # Start calibration
    controller.start_calibration()
    
    try:
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip the frame horizontally for a more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame, gesture, normalized_position = controller.process_frame(frame)
            
            # Store the frame for debug overlay
            controller.last_frame = processed_frame
            
            # Control the game based on detected gestures
            if not controller.calibration_mode:
                controller.control_game(gesture, normalized_position)
                
            # Add additional visual feedback
            if not controller.calibration_mode:
                # Show current mode
                mode_text = "CLICK & PULL MODE" if controller.is_aiming or controller.power_adjustment else "AIM MODE"
                cv2.putText(processed_frame, mode_text, (processed_frame.shape[1] - 300, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('8-Ball Billiards Gesture Controller', processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                controller.start_calibration()
            elif key == ord('d'):
                controller.toggle_debug()
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        controller.release()


if __name__ == "__main__":
    main()
