import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import math

# Disable pyautogui fail-safe (optional, but prevents issues if hand moves too fast)
pyautogui.FAILSAFE = False

class GestureGameController:
    def __init__(self):
        """Initialize the hand gesture controller for web-based games."""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for simplicity
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Screen dimensions for mapping
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture states
        self.prev_landmarks = None
        self.last_gesture = None
        self.gesture_history = []
        self.gesture_start_time = time.time()
        
        # Smoothing parameters
        self.smoothing_factor = 0.5  # Higher = more smoothing
        self.prev_mouse_pos = pyautogui.position()
        
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
        Process a video frame and detect hand landmarks.
        
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
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            
            # Draw hand landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
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
            
            # Detect gesture
            gesture = self._detect_gesture(landmarks)
            
            # Get normalized position (for mouse control)
            normalized_position = self._get_normalized_position(landmarks[0, :2])  # Use wrist position
            
            # Update gesture history
            if self.last_gesture != gesture:
                self.gesture_start_time = time.time()
            self.last_gesture = gesture
            
            self.gesture_history.append(gesture)
            if len(self.gesture_history) > 10:
                self.gesture_history.pop(0)
            
            # Draw gesture and position info on frame if debug is enabled
            if self.show_debug:
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
            gesture: Detected gesture ('open_hand', 'closed_fist', 'pointing', or None)
        """
        # Calculate fingertip distances from palm center
        palm_center = np.mean(landmarks[[0, 1, 5, 9, 13, 17]], axis=0)
        
        # Get fingertips
        fingertips = landmarks[[4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky
        
        # Calculate distances from palm center to fingertips
        distances = np.linalg.norm(fingertips - palm_center, axis=1)
        
        # Check if index finger is extended but others are closed (pointing gesture)
        index_extended = distances[1] > 0.1
        others_closed = np.mean(distances[[0, 2, 3, 4]]) < 0.1
        
        if index_extended and others_closed:
            return 'pointing'
        
        # Check for open hand vs closed fist
        avg_distance = np.mean(distances)
        if avg_distance > 0.1:
            return 'open_hand'
        else:
            return 'closed_fist'
    
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
        Get normalized position based on calibration data.
        
        Args:
            position: Raw position (x, y)
            
        Returns:
            Normalized position (x, y) in range [0, 1]
        """
        if (self.calibration_data['range_of_motion']['right'] > 
            self.calibration_data['range_of_motion']['left']):
            
            # Use calibration data
            x_normalized = (position[0] - self.calibration_data['range_of_motion']['left']) / (
                self.calibration_data['range_of_motion']['right'] - 
                self.calibration_data['range_of_motion']['left'])
            
            y_normalized = (position[1] - self.calibration_data['range_of_motion']['top']) / (
                self.calibration_data['range_of_motion']['bottom'] - 
                self.calibration_data['range_of_motion']['top'])
        else:
            # Default normalization without calibration
            x_normalized = position[0]
            y_normalized = position[1]
        
        # Clamp values to [0, 1] range
        x_normalized = max(0.0, min(1.0, x_normalized))
        y_normalized = max(0.0, min(1.0, y_normalized))
        
        return (x_normalized, y_normalized)
    
    def control_game(self, gesture, normalized_position):
        """
        Control the game based on detected gestures and hand position.
        
        Args:
            gesture: Detected gesture
            normalized_position: Normalized hand position (x, y)
        """
        if normalized_position is None:
            return
        
        # Decrease strike cooldown if it's active
        if self.strike_cooldown > 0:
            self.strike_cooldown -= 1
            return
        
        # Map normalized position to screen coordinates
        target_x = int(normalized_position[0] * self.screen_width)
        target_y = int(normalized_position[1] * self.screen_height)
        
        # Apply smoothing to mouse movement
        current_x, current_y = pyautogui.position()
        smooth_x = current_x + (target_x - current_x) * (1 - self.smoothing_factor)
        smooth_y = current_y + (target_y - current_y) * (1 - self.smoothing_factor)
        
        # Handle different gestures
        if gesture == 'open_hand':
            # Move mouse cursor (for aiming)
            pyautogui.moveTo(smooth_x, smooth_y)
            
            # If we were in power adjustment mode, end it and strike
            if self.power_adjustment:
                self.power_adjustment = False
                # Strike the ball (mouse click)
                pyautogui.click()
                self.strike_cooldown = 10  # Prevent multiple clicks
        
        elif gesture == 'closed_fist':
            # Start power adjustment if we weren't already
            if not self.power_adjustment and not self.is_aiming:
                # First click to start aiming
                pyautogui.moveTo(smooth_x, smooth_y)
                pyautogui.mouseDown()
                self.is_aiming = True
                self.aim_start_pos = (smooth_x, smooth_y)
            elif self.is_aiming:
                # We're already aiming, now adjust power
                self.power_adjustment = True
                self.is_aiming = False
                # Release mouse to end aiming
                pyautogui.mouseUp()
        
        elif gesture == 'pointing':
            # Click at the current position (for menu navigation)
            if self.last_gesture != 'pointing':
                pyautogui.click()
                self.strike_cooldown = 10  # Prevent multiple clicks
    
    def toggle_debug(self):
        """Toggle debug information display."""
        self.show_debug = not self.show_debug
    
    def release(self):
        """Release resources."""
        self.hands.close()


def main():
    # Initialize the controller
    controller = GestureGameController()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("\n=== 8-Ball Billiards Web Game Hand Gesture Controller ===")
    print("1. Open your web browser and navigate to: https://www.crazygames.com/game/8-ball-billiards-classic")
    print("2. Position the game window so it's visible alongside this controller window")
    print("3. Use the following gestures to control the game:")
    print("   - Open hand: Move the mouse cursor for aiming")
    print("   - Closed fist: Start aiming (first time) or adjust power (second time)")
    print("   - Open hand after closed fist: Strike the ball")
    print("   - Pointing (index finger): Click for menu navigation")
    print("4. Press 'C' to recalibrate, 'D' to toggle debug info, 'Q' to quit\n")
    
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
            
            # Control the game based on detected gestures
            if not controller.calibration_mode:
                controller.control_game(gesture, normalized_position)
            
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
