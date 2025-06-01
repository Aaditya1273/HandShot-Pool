import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import math
from collections import deque

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

class SimpleGestureController:
    def __init__(self):
        """Initialize a simple hand gesture controller for web-based billiards games."""
        # Initialize MediaPipe Hands with improved settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for simplicity
            min_detection_confidence=0.6,  # Lower threshold for better detection
            min_tracking_confidence=0.6,   # Lower threshold for better tracking
            model_complexity=1             # Use more complex model for better accuracy
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Custom drawing specs for better visualization
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture states with history for smoothing
        self.prev_landmarks = None
        self.last_gesture = None
        self.gesture_start_time = time.time()
        self.gesture_history = deque(maxlen=10)  # Store last 10 gestures for smoothing
        self.landmark_history = deque(maxlen=5)  # Store last 5 landmark sets for smoothing
        self.angle_history = deque(maxlen=8)     # Store last 8 angles for smoothing
        
        # Game control state
        self.is_aiming = False
        self.ready_to_hit = False
        self.strike_cooldown = 0
        
        # Hand rotation tracking with sensitivity
        self.prev_angle = None
        self.angle_sensitivity = 1.5  # Adjust rotation sensitivity
        self.position_smoothing = 0.7  # Higher = more smoothing
        
        # Gesture detection parameters
        self.finger_extension_threshold = 0.09  # Threshold for detecting extended fingers
        self.fist_threshold = 0.07            # Threshold for detecting closed fist
        
        # Debug info
        self.show_debug = True
        self.show_landmarks = True
        self.show_finger_status = True
    
    def process_frame(self, frame):
        """Process a video frame and detect hand landmarks."""
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        gesture = None
        hand_angle = None
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
            
            # Draw hand landmarks on the frame with enhanced visualization
            if self.show_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Store landmarks in history for smoothing
            self.landmark_history.append(landmarks)
            
            # Use smoothed landmarks for better stability
            smoothed_landmarks = self._smooth_landmarks()
            
            # Detect gesture (open hand or closed fist) with improved detection
            gesture = self._detect_gesture(smoothed_landmarks)
            self.gesture_history.append(gesture)
            
            # Get smoothed gesture (majority vote from history)
            smoothed_gesture = self._get_smoothed_gesture()
            
            # Calculate hand angle for rotation with smoothing
            hand_angle = self._calculate_hand_angle(smoothed_landmarks)
            if hand_angle is not None:
                self.angle_history.append(hand_angle)
            
            # Get smoothed angle
            smoothed_angle = self._get_smoothed_angle()
            
            # Draw detailed debug info on frame if enabled
            if self.show_debug:
                # Draw gesture info
                cv2.putText(frame, f"Detected: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Smoothed: {smoothed_gesture}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw angle info
                if hand_angle is not None:
                    cv2.putText(frame, f"Angle: {hand_angle:.1f} degrees", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw game state
                if self.ready_to_hit:
                    cv2.putText(frame, "READY TO HIT", (frame.shape[1] - 200, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw finger status if enabled
                if self.show_finger_status and smoothed_landmarks is not None:
                    self._draw_finger_status(frame, smoothed_landmarks)
            
            self.prev_landmarks = landmarks
        
        return frame, smoothed_gesture, smoothed_angle
    
    def _smooth_landmarks(self):
        """Apply smoothing to landmarks for more stable detection."""
        if len(self.landmark_history) == 0:
            return None
            
        # Use weighted average of recent landmarks, giving more weight to recent frames
        weights = np.linspace(0.5, 1.0, len(self.landmark_history))
        weights = weights / np.sum(weights)  # Normalize weights
        
        smoothed = np.zeros_like(self.landmark_history[0])
        for i, landmarks in enumerate(self.landmark_history):
            smoothed += landmarks * weights[i]
            
        return smoothed
    
    def _get_smoothed_gesture(self):
        """Get smoothed gesture using majority vote from history."""
        if not self.gesture_history:
            return None
            
        # Count occurrences of each gesture
        gestures = list(self.gesture_history)
        counts = {}
        for g in gestures:
            if g in counts:
                counts[g] += 1
            else:
                counts[g] = 1
                
        # Return the most common gesture
        return max(counts, key=counts.get) if counts else None
    
    def _get_smoothed_angle(self):
        """Get smoothed angle from angle history."""
        if not self.angle_history:
            return None
            
        # Use median for angle to avoid outliers
        return np.median(list(self.angle_history))
    
    def _detect_gesture(self, landmarks):
        """Detect hand gestures with improved accuracy."""
        if landmarks is None:
            return None
            
        # Calculate palm center using more points for stability
        palm_center = np.mean(landmarks[[0, 1, 5, 9, 13, 17]], axis=0)
        
        # Get fingertips and middle joints
        fingertips = landmarks[[4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky
        finger_bases = landmarks[[2, 5, 9, 13, 17]]  # Base of each finger
        middle_joints = landmarks[[3, 6, 10, 14, 18]]  # Middle joints of each finger
        
        # Calculate distances from palm center to fingertips
        fingertip_distances = np.linalg.norm(fingertips - palm_center, axis=1)
        
        # Calculate finger curl (distance from tip to base relative to finger length)
        finger_lengths = np.linalg.norm(fingertips - finger_bases, axis=1)
        finger_curls = []
        
        for i in range(5):  # For each finger
            # Calculate the ratio of the distance from fingertip to base
            # compared to the length of the finger when extended
            tip_to_base_dist = np.linalg.norm(fingertips[i] - finger_bases[i])
            middle_to_base_dist = np.linalg.norm(middle_joints[i] - finger_bases[i])
            
            # A lower ratio means the finger is more curled
            if middle_to_base_dist > 0:
                curl_ratio = tip_to_base_dist / (2 * middle_to_base_dist)  # Normalized curl
                finger_curls.append(curl_ratio)
            else:
                finger_curls.append(1.0)  # Default to extended if division issue
        
        # Detect gestures based on finger curls and distances
        # For a closed fist, most fingers should be curled
        num_curled = sum(1 for curl in finger_curls if curl < self.fist_threshold)
        num_extended = sum(1 for curl in finger_curls if curl > self.finger_extension_threshold)
        
        if num_curled >= 4:  # At least 4 fingers curled
            return 'closed_fist'
        elif num_extended >= 4:  # At least 4 fingers extended
            return 'open_hand'
        elif finger_curls[1] > self.finger_extension_threshold and num_curled >= 3:  # Index finger extended, others curled
            return 'pointing'
        else:
            # Default to the most likely gesture based on average curl
            avg_curl = np.mean(finger_curls)
            return 'closed_fist' if avg_curl < 0.15 else 'open_hand'
    
    def _calculate_hand_angle(self, landmarks):
        """Calculate the angle of the hand for rotation control with improved accuracy."""
        if landmarks is None:
            return None
            
        # Use multiple points for more stable angle calculation
        # Wrist as the base point
        wrist = landmarks[0, :2]
        
        # Use the middle finger MCP joint and the pinky MCP joint to form a line across the hand
        middle_mcp = landmarks[9, :2]  # Middle finger MCP joint
        pinky_mcp = landmarks[17, :2]  # Pinky MCP joint
        
        # Calculate the midpoint between these two points
        hand_center = (middle_mcp + pinky_mcp) / 2
        
        # Calculate the angle between the wrist and this midpoint
        dx = hand_center[0] - wrist[0]
        dy = hand_center[1] - wrist[1]
        
        if dx == 0 and dy == 0:  # Avoid division by zero
            return self.prev_angle if self.prev_angle is not None else 0
            
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
        
    def _draw_finger_status(self, frame, landmarks):
        """Draw finger status (extended/curled) on the frame."""
        if landmarks is None:
            return
            
        # Calculate palm center
        palm_center = np.mean(landmarks[[0, 1, 5, 9, 13, 17]], axis=0)
        
        # Get fingertips and bases
        fingertips = landmarks[[4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky
        finger_bases = landmarks[[2, 5, 9, 13, 17]]  # Base of each finger
        middle_joints = landmarks[[3, 6, 10, 14, 18]]  # Middle joints
        
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        finger_statuses = []
        
        for i in range(5):
            # Calculate curl ratio
            tip_to_base_dist = np.linalg.norm(fingertips[i] - finger_bases[i])
            middle_to_base_dist = np.linalg.norm(middle_joints[i] - finger_bases[i])
            
            if middle_to_base_dist > 0:
                curl_ratio = tip_to_base_dist / (2 * middle_to_base_dist)
            else:
                curl_ratio = 1.0
                
            # Determine if extended or curled
            status = 'Ext' if curl_ratio > self.finger_extension_threshold else 'Curl'
            finger_statuses.append(f"{finger_names[i][0]}: {status}")
        
        # Draw finger status
        y_pos = 120
        for status in finger_statuses:
            cv2.putText(frame, status, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25
    
    def control_game(self, gesture, hand_angle):
        """Control the game based on detected gestures and hand angle with improved sensitivity."""
        # Decrease strike cooldown if it's active
        if self.strike_cooldown > 0:
            self.strike_cooldown -= 1
            return
        
        # Handle hand rotation for stick rotation with improved sensitivity
        if hand_angle is not None and self.prev_angle is not None:
            # Calculate angle difference
            angle_diff = hand_angle - self.prev_angle
            
            # Apply sensitivity factor
            adjusted_diff = angle_diff * self.angle_sensitivity
            
            # Only respond to significant changes but with better sensitivity
            if abs(adjusted_diff) > 1.5:  # Lower threshold for more responsive rotation
                # Determine number of key presses based on rotation speed
                num_presses = min(3, max(1, int(abs(adjusted_diff) / 5)))
                
                # Map hand rotation to arrow key presses for stick rotation
                for _ in range(num_presses):
                    if adjusted_diff > 0:
                        # Rotate clockwise - right arrow
                        pyautogui.press('right')
                    else:
                        # Rotate counter-clockwise - left arrow
                        pyautogui.press('left')
        
        # Update previous angle with smoothing
        if hand_angle is not None:
            if self.prev_angle is None:
                self.prev_angle = hand_angle
            else:
                # Apply smoothing to prevent jitter
                self.prev_angle = self.prev_angle * self.position_smoothing + hand_angle * (1 - self.position_smoothing)
        
        # Handle gesture changes with improved transition detection
        if gesture != self.last_gesture:
            # Add a small delay to ensure gesture is stable
            if time.time() - self.gesture_start_time > 0.2:  # 200ms stability check
                if gesture == 'closed_fist' and not self.ready_to_hit:
                    # Closed fist = ready to hit
                    self.ready_to_hit = True
                    # Click and hold to start aiming
                    pyautogui.mouseDown()
                    print("Ready to hit - Mouse down")
                
                elif gesture == 'open_hand' and self.last_gesture == 'closed_fist' and self.ready_to_hit:
                    # Open hand after closed fist = hit the ball
                    # Release mouse to strike
                    pyautogui.mouseUp()
                    print("Strike! - Mouse up")
                    self.ready_to_hit = False
                    self.strike_cooldown = 15  # Prevent multiple hits with longer cooldown
                
                # Update gesture start time for stability tracking
                self.gesture_start_time = time.time()
                self.last_gesture = gesture
    
    def toggle_debug(self):
        """Toggle debug information display."""
        self.show_debug = not self.show_debug
    
    def toggle_landmarks(self):
        """Toggle landmark visualization."""
        self.show_landmarks = not self.show_landmarks
        
    def toggle_finger_status(self):
        """Toggle finger status display."""
        self.show_finger_status = not self.show_finger_status
        
    def adjust_sensitivity(self, increase=True):
        """Adjust rotation sensitivity."""
        if increase:
            self.angle_sensitivity = min(3.0, self.angle_sensitivity + 0.2)
        else:
            self.angle_sensitivity = max(0.5, self.angle_sensitivity - 0.2)
        print(f"Rotation sensitivity: {self.angle_sensitivity:.1f}")
    
    def release(self):
        """Release resources."""
        self.hands.close()


def main():
    # Initialize the controller
    controller = SimpleGestureController()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("\n=== Simple 8-Ball Billiards Web Game Hand Gesture Controller ===")
    print("1. Open your web browser and navigate to: https://www.crazygames.com/game/8-ball-billiards-classic")
    print("2. Position the game window so it's visible alongside this controller window")
    print("3. Use the following gestures to control the game:")
    print("   - Hand rotation: Rotate the cue stick")
    print("   - Closed fist: Ready to hit (click and hold)")
    print("   - Open hand after closed fist: Hit the ball")
    print("4. Keyboard controls:")
    print("   - 'D': Toggle debug information")
    print("   - 'L': Toggle hand landmark visualization")
    print("   - 'F': Toggle finger status display")
    print("   - '+'/'-': Increase/decrease rotation sensitivity")
    print("   - 'Q': Quit the controller\n")
    
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
            processed_frame, gesture, hand_angle = controller.process_frame(frame)
            
            # Control the game based on detected gestures
            controller.control_game(gesture, hand_angle)
            
            # Display the frame
            cv2.imshow('Simple Billiards Gesture Controller', processed_frame)
            
            # Check for key presses with more options
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                controller.toggle_debug()
            elif key == ord('l'):
                controller.toggle_landmarks()
            elif key == ord('f'):
                controller.toggle_finger_status()
            elif key == ord('+') or key == ord('='):
                controller.adjust_sensitivity(increase=True)
            elif key == ord('-'):
                controller.adjust_sensitivity(increase=False)
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        controller.release()


if __name__ == "__main__":
    main()
