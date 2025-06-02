import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

def main():
    """Basic cursor control with hand tracking."""
    print("=" * 50)
    print("Basic Hand Cursor Control")
    print("=" * 50)
    
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    print(f"Screen resolution: {screen_width}x{screen_height}")
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables for smoothing
    prev_x, prev_y = screen_width // 2, screen_height // 2
    smoothing = 0.7  # Higher value = more smoothing
    
    # Variables for clicking
    is_clicking = False
    click_cooldown = 0
    
    print("Controls:")
    print("- Move your hand to control the cursor")
    print("- Make a pinching gesture (thumb and index finger) to click")
    print("- Press 'q' to quit")
    print("=" * 50)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and control cursor
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get index finger tip position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Get thumb tip position for click detection
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                
                # Calculate raw screen coordinates
                raw_x = int(index_tip.x * screen_width)
                raw_y = int(index_tip.y * screen_height)
                
                # Apply smoothing
                smooth_x = int(prev_x * smoothing + raw_x * (1 - smoothing))
                smooth_y = int(prev_y * smoothing + raw_y * (1 - smoothing))
                
                # Update previous positions
                prev_x, prev_y = smooth_x, smooth_y
                
                # Ensure coordinates are within screen bounds
                cursor_x = max(0, min(screen_width - 1, smooth_x))
                cursor_y = max(0, min(screen_height - 1, smooth_y))
                
                # Move cursor
                try:
                    pyautogui.moveTo(cursor_x, cursor_y)
                    
                    # Display cursor position
                    cv2.putText(frame, f"Cursor: ({cursor_x}, {cursor_y})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error moving cursor: {e}")
                
                # Detect click (pinch gesture)
                thumb_to_index_distance = np.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 + 
                    (thumb_tip.y - index_tip.y) ** 2
                )
                
                # Display distance
                cv2.putText(frame, f"Pinch: {thumb_to_index_distance:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Click when pinching and cooldown is zero
                if thumb_to_index_distance < 0.05 and click_cooldown == 0:
                    if not is_clicking:
                        try:
                            pyautogui.click()
                            is_clicking = True
                            click_cooldown = 10  # Set cooldown to prevent multiple clicks
                            
                            # Display click status
                            cv2.putText(frame, "CLICK!", (frame.shape[1] - 150, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error clicking: {e}")
                else:
                    is_clicking = False
                
                # Decrease cooldown
                if click_cooldown > 0:
                    click_cooldown -= 1
        
        # Display the frame
        cv2.imshow("Basic Hand Cursor Control", frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
