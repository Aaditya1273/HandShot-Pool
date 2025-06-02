import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import math
from collections import deque
from gesture_recognition import GestureRecognizer

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

class MasterHitController:
    def __init__(self):
        """Initialize the hand gesture controller for Master Hit: Boss Hunter."""
        # Initialize gesture recognizer
        self.gesture_recognizer = GestureRecognizer(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Screen dimensions for mapping
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Game control state
        self.is_shooting = False
        self.shoot_cooldown = 0
        self.dodge_active = False
        self.dodge_cooldown = 0
        self.last_position = None
        
        # Debug info
        self.show_debug = True
        self.last_frame = None
    
    def process_frame(self, frame):
        """
        Process a video frame and control the game based on detected gestures.
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with hand landmarks and debug info
        """
        # Process frame with gesture recognizer
        processed_frame, gesture, normalized_position = self.gesture_recognizer.process_frame(frame)
        
        # Store frame for debug overlay
        self.last_frame = processed_frame
        
        # Control the game if gesture and position are detected
        if gesture and normalized_position:
            self.control_game(gesture, normalized_position)
        
        # Draw game control state
        if self.show_debug:
            # Draw shooting status
            if self.is_shooting:
                cv2.putText(processed_frame, "SHOOTING", (processed_frame.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw dodge status
            if self.dodge_active:
                cv2.putText(processed_frame, "DODGING", (processed_frame.shape[1] - 200, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw control instructions
            cv2.putText(processed_frame, "Open hand: Move", (10, processed_frame.shape[0] - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Index finger only: Shoot", (10, processed_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Closed fist: Dodge/Collect", (10, processed_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return processed_frame
    
    def control_game(self, gesture, normalized_position):
        """
        Control the Master Hit game with gestures.
        
        Args:
            gesture: Detected gesture ('move', 'shoot', 'dodge')
            normalized_position: Normalized hand position (x, y)
        """
        if normalized_position is None:
            return
        
        # Decrease cooldowns
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        if self.dodge_cooldown > 0:
            self.dodge_cooldown -= 1
            if self.dodge_cooldown == 0:
                self.dodge_active = False
        
        # Map normalized position to screen coordinates
        target_x = int(normalized_position[0] * self.screen_width)
        target_y = int(normalized_position[1] * self.screen_height)
        
        # Apply smoother cursor movement
        current_x, current_y = pyautogui.position()
        
        # Calculate distance to target
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        # Adaptive smoothing - less smoothing for small movements, more for large ones
        adaptive_factor = min(0.5, max(0.05, distance / 500))
        smooth_factor = 0.3 * (1 - adaptive_factor)
        
        # Final smoothing
        smooth_x = int(current_x + (target_x - current_x) * (1 - smooth_factor))
        smooth_y = int(current_y + (target_y - current_y) * (1 - smooth_factor))
        
        # Move cursor (for all gestures - the eye follows the cursor)
        pyautogui.moveTo(smooth_x, smooth_y)
        
        # Handle game-specific gestures
        if gesture == 'move':
            # Just move the cursor (already done above)
            # Release any pressed keys/buttons
            if self.is_shooting:
                pyautogui.mouseUp()
                self.is_shooting = False
        
        elif gesture == 'shoot' and self.shoot_cooldown == 0:
            # Shoot by clicking and holding the mouse button
            if not self.is_shooting:
                pyautogui.mouseDown()
                self.is_shooting = True
                
                # Debug output for shoot
                if self.show_debug and self.last_frame is not None:
                    cv2.putText(self.last_frame, "SHOOT!", 
                              (self.last_frame.shape[1]//2-50, self.last_frame.shape[0]//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        elif gesture == 'dodge' and self.dodge_cooldown == 0:
            # Dodge by pressing space bar or collect power-ups
            pyautogui.press('space')
            self.dodge_active = True
            self.dodge_cooldown = 15  # Prevent rapid dodging
            
            # Debug output for dodge
            if self.show_debug and self.last_frame is not None:
                cv2.putText(self.last_frame, "DODGE!", 
                          (self.last_frame.shape[1]//2-50, self.last_frame.shape[0]//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        
        # Store last position
        self.last_position = (smooth_x, smooth_y)
    
    def toggle_debug(self):
        """Toggle debug information display."""
        self.show_debug = not self.show_debug
    
    def start_calibration(self):
        """Start the calibration process."""
        self.gesture_recognizer.start_calibration()
    
    def release(self):
        """Release resources."""
        self.gesture_recognizer.release()
        # Ensure mouse button is released
        pyautogui.mouseUp()


def main():
    # Initialize the controller
    controller = MasterHitController()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Start calibration
    controller.start_calibration()
    
    print("=" * 50)
    print("Master Hit: Boss Hunter - Hand Gesture Controller")
    print("=" * 50)
    print("Controls:")
    print("- Open hand: Move the eye")
    print("- Index finger only: Shoot")
    print("- Closed fist: Dodge or collect power-ups")
    print("Press 'q' to quit, 'd' to toggle debug info")
    print("=" * 50)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a more intuitive experience
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        processed_frame = controller.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Master Hit: Boss Hunter - Hand Gesture Controller', processed_frame)
        
        # Check for key presses
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
    main()
