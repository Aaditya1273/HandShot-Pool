import cv2
import numpy as np
import time
import pyautogui
import mediapipe as mp

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

class DirectKnifeController:
    def __init__(self):
        """A simplified and direct controller for knife-throwing games."""
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Tracking variables
        self.prev_x, self.prev_y = self.screen_width // 2, self.screen_height // 2
        self.smoothing = 0.5  # Lower = more responsive, Higher = smoother
        
        # Click variables
        self.is_clicking = False
        self.click_cooldown = 0
        self.max_cooldown = 15  # Frames to wait between clicks
        
        # Debug
        self.show_debug = True
        
        # Finger state tracking
        self.finger_states = []
        self.finger_state_history_size = 5
        self.finger_state_threshold = 3  # Number of consistent states to trigger action
        
        # Calibration
        self.min_finger_distance = 0.1  # Minimum distance from palm to consider finger extended
        self.calibration_frames = 0
        self.calibration_max_frames = 30
        self.is_calibrating = True
        self.calibration_distances = []
    
    def calibrate(self, landmarks):
        """Calibrate finger distances based on user's hand."""
        if self.calibration_frames < self.calibration_max_frames:
            # Get wrist position
            wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Get fingertip positions
            fingertips = [
                landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
                landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            ]
            
            # Calculate distances from wrist to fingertips
            distances = []
            for tip in fingertips:
                distance = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5
                distances.append(distance)
            
            self.calibration_distances.append(distances)
            self.calibration_frames += 1
            
            return True
        else:
            # Calculate average distances
            avg_distances = []
            for i in range(4):  # 4 fingers
                finger_distances = [frame[i] for frame in self.calibration_distances]
                avg_distances.append(sum(finger_distances) / len(finger_distances))
            
            # Set minimum finger distance as 40% of average extended distance
            self.min_finger_distance = min(avg_distances) * 0.4
            
            print(f"Calibration complete! Minimum finger distance: {self.min_finger_distance:.4f}")
            self.is_calibrating = False
            return False
    
    def process_frame(self, frame):
        """Process a video frame and control the game."""
        if frame is None or frame.size == 0:
            return frame
        
        # Flip the frame horizontally for a more intuitive experience
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Create a copy for drawing
        output_frame = frame.copy()
        
        # Draw calibration status
        if self.is_calibrating:
            progress = int((self.calibration_frames / self.calibration_max_frames) * 100)
            cv2.putText(output_frame, f"Calibrating: {progress}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, "Open and close your hand a few times", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Check if hands are detected
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                output_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            # If still calibrating, continue calibration
            if self.is_calibrating:
                self.calibrate(hand_landmarks)
            else:
                # Get finger state and control cursor
                finger_state = self.get_finger_state(hand_landmarks)
                self.control_cursor(hand_landmarks, finger_state, output_frame)
                
                # Draw finger state
                state_text = "OPEN" if finger_state else "CLOSED"
                cv2.putText(output_frame, f"Hand: {state_text}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Track finger state history
                self.finger_states.append(finger_state)
                if len(self.finger_states) > self.finger_state_history_size:
                    self.finger_states.pop(0)
                
                # Check for throw gesture (transition from closed to open)
                if self.detect_throw_gesture():
                    if self.click_cooldown == 0:
                        # Click to throw knife
                        pyautogui.click()
                        
                        # Show throw indicator
                        cv2.putText(output_frame, "THROW!", (output_frame.shape[1] - 150, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # Set cooldown
                        self.click_cooldown = self.max_cooldown
                        self.is_clicking = True
        
        # Decrease cooldown
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
            if self.click_cooldown == 0:
                self.is_clicking = False
        
        # Draw cooldown indicator
        if self.is_clicking:
            cv2.putText(output_frame, f"Cooldown: {self.click_cooldown}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw instructions
        cv2.putText(output_frame, "Open hand: Move cursor", (10, output_frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, "Close -> Open: Throw knife", (10, output_frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame
    
    def get_finger_state(self, landmarks):
        """Determine if hand is open (True) or closed (False)."""
        # Get wrist position
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Get fingertip positions
        fingertips = [
            landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        # Get finger base positions (for better detection)
        finger_bases = [
            landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP],
            landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        ]
        
        # Count extended fingers
        extended_fingers = 0
        for tip, base in zip(fingertips, finger_bases):
            # Calculate distance from wrist to fingertip
            tip_distance = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5
            
            # Calculate distance from wrist to base
            base_distance = ((base.x - wrist.x) ** 2 + (base.y - wrist.y) ** 2) ** 0.5
            
            # If fingertip is farther from wrist than base by threshold, finger is extended
            if tip_distance > base_distance + self.min_finger_distance:
                extended_fingers += 1
        
        # Hand is considered open if at least 2 fingers are extended
        return extended_fingers >= 2
    
    def detect_throw_gesture(self):
        """Detect a throw gesture (transition from closed to open hand)."""
        if len(self.finger_states) < 3:
            return False
        
        # Check for a sequence of closed followed by open
        # We need at least 2 closed states followed by an open state
        if (not self.finger_states[-3] and not self.finger_states[-2] and self.finger_states[-1]):
            return True
        
        return False
    
    def control_cursor(self, landmarks, is_open, frame):
        """Control the cursor based on hand position and state."""
        # Use index finger for cursor position
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Map to screen coordinates
        raw_x = int(index_tip.x * self.screen_width)
        raw_y = int(index_tip.y * self.screen_height)
        
        # Apply smoothing
        smooth_x = int(self.prev_x * self.smoothing + raw_x * (1 - self.smoothing))
        smooth_y = int(self.prev_y * self.smoothing + raw_y * (1 - self.smoothing))
        
        # Update previous positions
        self.prev_x, self.prev_y = smooth_x, smooth_y
        
        # Move cursor only if hand is open
        if is_open and not self.is_clicking:
            try:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)
                
                # Draw cursor position
                if self.show_debug:
                    cv2.putText(frame, f"Cursor: ({smooth_x}, {smooth_y})", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error moving cursor: {e}")
    
    def toggle_debug(self):
        """Toggle debug information."""
        self.show_debug = not self.show_debug
    
    def release(self):
        """Release resources."""
        self.hands.close()


def main():
    """Main function to run the direct knife controller."""
    # Initialize the controller
    controller = DirectKnifeController()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("=" * 50)
    print("Master Hit - Direct Knife Controller")
    print("=" * 50)
    print("Controls:")
    print("- Open hand: Move cursor")
    print("- Close -> Open: Throw knife")
    print("Press 'q' to quit, 'd' to toggle debug info")
    print("=" * 50)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process the frame
        processed_frame = controller.process_frame(frame)
        
        # Display the frame
        cv2.imshow("Master Hit - Direct Knife Controller", processed_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            controller.toggle_debug()
    
    # Release resources
    controller.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
