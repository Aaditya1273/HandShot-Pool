import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import math

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

class SimpleGestureController:
    def __init__(self):
        """Initialize a simple hand gesture controller for web-based billiards games."""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for simplicity
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Gesture states
        self.prev_landmarks = None
        self.last_gesture = None
        self.gesture_start_time = time.time()
        
        # Game control state
        self.is_aiming = False
        self.ready_to_hit = False
        self.strike_cooldown = 0
        
        # Hand rotation tracking
        self.prev_angle = None
        
        # Debug info
        self.show_debug = True
    
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
            
            # Draw hand landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Detect gesture (open hand or closed fist)
            gesture = self._detect_gesture(landmarks)
            
            # Calculate hand angle for rotation
            hand_angle = self._calculate_hand_angle(landmarks)
            
            # Draw gesture info on frame if debug is enabled
            if self.show_debug:
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if hand_angle is not None:
                    cv2.putText(frame, f"Angle: {hand_angle:.1f} degrees", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw game state
                if self.ready_to_hit:
                    cv2.putText(frame, "READY TO HIT", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            self.prev_landmarks = landmarks
        
        return frame, gesture, hand_angle
    
    def _detect_gesture(self, landmarks):
        """Detect if the hand is open or closed."""
        # Calculate fingertip distances from palm center
        palm_center = np.mean(landmarks[[0, 1, 5, 9, 13, 17]], axis=0)
        
        # Get fingertips
        fingertips = landmarks[[4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky
        
        # Calculate distances from palm center to fingertips
        distances = np.linalg.norm(fingertips - palm_center, axis=1)
        
        # Check for open hand vs closed fist
        avg_distance = np.mean(distances)
        if avg_distance > 0.1:
            return 'open_hand'
        else:
            return 'closed_fist'
    
    def _calculate_hand_angle(self, landmarks):
        """Calculate the angle of the hand for rotation control."""
        # Use wrist and middle finger base to determine hand orientation
        wrist = landmarks[0, :2]
        middle_base = landmarks[9, :2]
        
        # Calculate angle
        dx = middle_base[0] - wrist[0]
        dy = middle_base[1] - wrist[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def control_game(self, gesture, hand_angle):
        """Control the game based on detected gestures and hand angle."""
        # Decrease strike cooldown if it's active
        if self.strike_cooldown > 0:
            self.strike_cooldown -= 1
            return
        
        # Handle hand rotation for stick rotation
        if hand_angle is not None and self.prev_angle is not None:
            # Calculate angle difference
            angle_diff = hand_angle - self.prev_angle
            
            # Only respond to significant changes
            if abs(angle_diff) > 2:
                # Map hand rotation to arrow key presses for stick rotation
                if angle_diff > 0:
                    # Rotate clockwise - right arrow
                    pyautogui.press('right')
                else:
                    # Rotate counter-clockwise - left arrow
                    pyautogui.press('left')
        
        # Update previous angle
        self.prev_angle = hand_angle
        
        # Handle gesture changes
        if gesture != self.last_gesture:
            if gesture == 'closed_fist':
                # Closed fist = ready to hit
                self.ready_to_hit = True
                # Click and hold to start aiming
                pyautogui.mouseDown()
            
            elif gesture == 'open_hand' and self.last_gesture == 'closed_fist':
                # Open hand after closed fist = hit the ball
                if self.ready_to_hit:
                    # Release mouse to strike
                    pyautogui.mouseUp()
                    self.ready_to_hit = False
                    self.strike_cooldown = 10  # Prevent multiple hits
        
        self.last_gesture = gesture
    
    def toggle_debug(self):
        """Toggle debug information display."""
        self.show_debug = not self.show_debug
    
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
    print("4. Press 'D' to toggle debug info, 'Q' to quit\n")
    
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
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                controller.toggle_debug()
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        controller.release()


if __name__ == "__main__":
    main()
