import cv2
import pygame
import numpy as np
import time
import sys
import math

from gesture_recognition import GestureRecognizer
from pool_physics import PoolTable
from game_ui import GameUI

# Constants
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
TABLE_WIDTH = 900
TABLE_HEIGHT = 450
FPS = 60

# Game states
GAME_STATE_CALIBRATION = 0
GAME_STATE_AIMING = 1
GAME_STATE_SHOOTING = 2
GAME_STATE_BALLS_MOVING = 3
GAME_STATE_GAME_OVER = 4


class PoolGame:
    def __init__(self):
        """Initialize the 8-Ball Pool Game with hand gesture controls."""
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("HandShot Pool - Gesture Controlled 8-Ball")
        self.clock = pygame.time.Clock()
        
        # Initialize OpenCV camera
        self.camera = cv2.VideoCapture(0)
        # Check if camera opened successfully
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
            
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize the gesture recognizer
        self.gesture_recognizer = GestureRecognizer()
        
        # Initialize the pool table
        table_x = (SCREEN_WIDTH - TABLE_WIDTH) // 2
        table_y = (SCREEN_HEIGHT - TABLE_HEIGHT) // 2
        self.pool_table = PoolTable(TABLE_WIDTH, TABLE_HEIGHT)
        
        # Initialize game UI
        self.game_ui = GameUI(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Game state variables
        self.game_state = GAME_STATE_CALIBRATION
        self.calibration_step = 0
        self.calibration_countdown = 3  # Seconds
        self.calibration_last_time = time.time()
        
        # Cue control variables
        self.target_angle = 0
        self.power = 0.0
        self.hand_position = None
        self.last_gesture = None
        self.gesture_start_time = 0
        
        # Create a surface for the camera feed
        self.camera_surface = pygame.Surface((320, 240))
        
        # Font for text rendering
        self.font = pygame.font.Font(None, 36)
        
        # Game timing
        self.prev_time = time.time()
        self.running = True
    
    def process_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:
                    # Start calibration
                    self.start_calibration()
                elif event.key == pygame.K_r:
                    # Reset the game
                    self.reset_game()
                elif event.key == pygame.K_h:
                    # Toggle help text
                    self.game_ui.toggle_help()
    
    def process_camera(self):
        """Process camera input and detect hand gestures."""
        # Capture frame from camera
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Failed to capture frame.")
            return None, None, None
        
        # Flip the frame horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Process the frame with the gesture recognizer
        processed_frame, landmarks, gesture, velocity = self.gesture_recognizer.process_frame(frame)
        
        # Convert OpenCV frame to Pygame surface
        processed_frame = cv2.resize(processed_frame, (320, 240))
        camera_surface_array = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Ensure dimensions match before blitting
        # The array needs to be transposed to match Pygame's expected format
        camera_surface_array = np.transpose(camera_surface_array, (1, 0, 2))
        
        try:
            pygame.surfarray.blit_array(self.camera_surface, camera_surface_array)
        except ValueError as e:
            print(f"Surface dimensions: {self.camera_surface.get_size()}")
            print(f"Array dimensions: {camera_surface_array.shape}")
            print(f"Error: {e}")
            # Create a new surface with the correct dimensions as a fallback
            self.camera_surface = pygame.Surface((camera_surface_array.shape[0], camera_surface_array.shape[1]))
            pygame.surfarray.blit_array(self.camera_surface, camera_surface_array)
        
        return processed_frame, landmarks, gesture
    
    def update(self):
        """Update the game state."""
        # Calculate delta time
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # Process camera and gestures
        frame, landmarks, gesture = self.process_camera()
        
        # Update hand position if landmarks are detected
        if landmarks is not None:
            # Get normalized hand position (0 to 1 range)
            self.hand_position = self.gesture_recognizer.get_cue_position(
                frame.shape[1], frame.shape[0])
        
        # Update game based on the current state
        if self.game_state == GAME_STATE_CALIBRATION:
            self.update_calibration(gesture)
        elif self.game_state == GAME_STATE_AIMING:
            self.update_aiming(gesture, dt)
        elif self.game_state == GAME_STATE_SHOOTING:
            self.update_shooting(gesture, dt)
        elif self.game_state == GAME_STATE_BALLS_MOVING:
            self.update_balls_moving(dt)
        
        # Always update the pool table physics
        pocketed_balls = self.pool_table.update(dt)
        
        # Handle pocketed balls
        if pocketed_balls:
            self.handle_pocketed_balls(pocketed_balls)
    
    def update_calibration(self, gesture):
        """Update the calibration state."""
        if self.calibration_step == 0:
            # Starting calibration
            self.gesture_recognizer.start_calibration()
            self.calibration_step = 1
            self.calibration_last_time = time.time()
        elif self.calibration_step == 1:
            # Show open hand instructions
            calibration_text = "Show OPEN HAND and move around"
            self.game_ui.update(game_state="Calibration", 
                               calibration_mode=True,
                               calibration_text=calibration_text)
            
            # Check if enough time has passed
            if time.time() - self.calibration_last_time > 5:
                self.calibration_step = 2
                self.calibration_last_time = time.time()
        elif self.calibration_step == 2:
            # Show closed fist instructions
            calibration_text = "Show CLOSED FIST and move around"
            self.game_ui.update(game_state="Calibration", 
                               calibration_mode=True,
                               calibration_text=calibration_text)
            
            # Check if enough time has passed
            if time.time() - self.calibration_last_time > 5:
                # End calibration
                self.gesture_recognizer.end_calibration()
                self.calibration_step = 0
                self.game_state = GAME_STATE_AIMING
                self.game_ui.update(game_state="Player 1's Turn", 
                                   calibration_mode=False)
    
    def update_aiming(self, gesture, dt):
        """Update the aiming state."""
        cue_ball = self.pool_table.get_cue_ball()
        if cue_ball is None or cue_ball.in_pocket:
            # Handle cue ball in pocket
            self.handle_cue_ball_in_pocket()
            return
        
        # Only update if we have valid hand position
        if self.hand_position is not None:
            # Map hand position to table coordinates for visualization
            hand_x = self.hand_position[0] * SCREEN_WIDTH
            hand_y = self.hand_position[1] * SCREEN_HEIGHT
            
            # Calculate angle based on hand position relative to cue ball
            table_x = (SCREEN_WIDTH - TABLE_WIDTH) // 2
            table_y = (SCREEN_HEIGHT - TABLE_HEIGHT) // 2
            cue_ball_screen_x = table_x + cue_ball.position[0]
            cue_ball_screen_y = table_y + cue_ball.position[1]
            
            # Calculate angle between hand and cue ball
            dx = hand_x - cue_ball_screen_x
            dy = hand_y - cue_ball_screen_y
            self.target_angle = math.atan2(dy, dx)
            
            # Update cue stick with the new angle
            self.game_ui.update(cue_ball_pos=cue_ball.position, 
                               target_angle=self.target_angle,
                               power=self.power)
            
            # Check for striking gesture
            strike_power = self.gesture_recognizer.detect_strike()
            if strike_power is not None:
                # Start shooting
                self.power = strike_power
                self.game_state = GAME_STATE_SHOOTING
                self.gesture_start_time = time.time()
                
                # Update UI to show power meter
                self.game_ui.update(cue_ball_pos=cue_ball.position,
                                   target_angle=self.target_angle,
                                   power=self.power,
                                   game_state="Taking Shot")
        
        # Check for closed fist gesture to start power adjustment
        if gesture == 'closed_fist' and self.last_gesture != 'closed_fist':
            self.gesture_start_time = time.time()
        
        self.last_gesture = gesture
    
    def update_shooting(self, gesture, dt):
        """Update the shooting state."""
        cue_ball = self.pool_table.get_cue_ball()
        if cue_ball is None:
            return
        
        # Calculate shot power based on how long the closed fist is held
        hold_time = min(1.0, (time.time() - self.gesture_start_time) / 2.0)  # Max 2 seconds for full power
        self.power = hold_time
        
        # Update UI with current power
        self.game_ui.update(cue_ball_pos=cue_ball.position,
                           target_angle=self.target_angle,
                           power=self.power)
        
        # Check for open hand gesture to strike
        if gesture == 'open_hand' and self.last_gesture == 'closed_fist':
            # Strike the cue ball
            direction = self.game_ui.get_cue_direction()
            self.pool_table.strike_cue_ball(direction, self.power)
            
            # Change game state
            self.game_state = GAME_STATE_BALLS_MOVING
            self.power = 0.0
            
            # Update UI
            self.game_ui.update(cue_ball_pos=cue_ball.position,
                               power=0.0,
                               game_state="Balls Moving")
        
        self.last_gesture = gesture
    
    def update_balls_moving(self, dt):
        """Update the balls moving state."""
        # Check if balls have stopped moving
        if not self.pool_table.are_balls_moving():
            # Switch to aiming state
            self.game_state = GAME_STATE_AIMING
            
            # Check for game over condition
            if self.check_game_over():
                self.game_state = GAME_STATE_GAME_OVER
            else:
                # Switch player
                self.game_ui.switch_player()
    
    def handle_pocketed_balls(self, pocketed_balls):
        """Handle balls that went into pockets."""
        current_player = self.game_ui.current_player
        
        for ball_number in pocketed_balls:
            if ball_number == 0:  # Cue ball
                # Scratch - switch player
                self.game_ui.switch_player()
            elif ball_number == 8:  # 8-ball
                cue_ball = self.pool_table.get_cue_ball()
                if cue_ball and cue_ball.in_pocket:
                    # Pocketed cue ball and 8-ball - opponent wins
                    self.game_ui.update(game_state=f"Player {3 - current_player} Wins!")
                    self.game_state = GAME_STATE_GAME_OVER
                else:
                    # Pocketed just the 8-ball
                    # Check if this is a valid win or a loss
                    if (current_player == 1 and self.game_ui.player1_score >= 7) or \
                       (current_player == 2 and self.game_ui.player2_score >= 7):
                        # Valid win - all other balls of player's type are pocketed
                        self.game_ui.update(game_state=f"Player {current_player} Wins!")
                    else:
                        # Invalid win - player loses
                        self.game_ui.update(game_state=f"Player {3 - current_player} Wins!")
                    self.game_state = GAME_STATE_GAME_OVER
            else:
                # Regular ball
                self.game_ui.update_score(current_player, 1)
    
    def handle_cue_ball_in_pocket(self):
        """Handle the case when the cue ball goes into a pocket."""
        # Place the cue ball back on the table
        cue_ball = self.pool_table.get_cue_ball()
        if cue_ball and cue_ball.in_pocket:
            # Reset cue ball position to starting position
            cue_ball.position = np.array([TABLE_WIDTH * 0.25, TABLE_HEIGHT / 2], dtype=float)
            cue_ball.velocity = np.zeros(2)
            cue_ball.in_pocket = False
            
            # Switch player (scratch penalty)
            self.game_ui.switch_player()
    
    def check_game_over(self):
        """Check if the game is over."""
        # Check if 8-ball is pocketed
        for ball in self.pool_table.balls:
            if ball.number == 8 and ball.in_pocket:
                return True
        
        return False
    
    def start_calibration(self):
        """Start the calibration process."""
        self.game_state = GAME_STATE_CALIBRATION
        self.calibration_step = 0
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.pool_table.reset_balls()
        self.game_ui.reset_scores()
        self.power = 0.0
        self.target_angle = 0
        self.game_state = GAME_STATE_AIMING
    
    def draw(self):
        """Draw the game on the screen."""
        # Fill screen with background color
        self.screen.fill((50, 50, 50))
        
        # Calculate table position
        table_x = (SCREEN_WIDTH - TABLE_WIDTH) // 2
        table_y = (SCREEN_HEIGHT - TABLE_HEIGHT) // 2
        
        # Create a surface for the table
        table_surface = pygame.Surface((TABLE_WIDTH, TABLE_HEIGHT))
        
        # Draw pool table and balls on the table surface
        self.pool_table.draw(table_surface)
        
        # Draw UI elements (cue stick, power meter) on the table surface
        cue_ball = self.pool_table.get_cue_ball()
        if cue_ball and not cue_ball.in_pocket and self.game_state != GAME_STATE_BALLS_MOVING:
            self.game_ui.draw(table_surface, cue_ball.position)
        else:
            self.game_ui.draw(table_surface, None)
        
        # Draw the table surface on the screen
        self.screen.blit(table_surface, (table_x, table_y))
        
        # Draw camera feed in the corner
        self.screen.blit(self.camera_surface, (SCREEN_WIDTH - 330, 10))
        
        # Draw a border around the camera feed
        pygame.draw.rect(self.screen, (200, 200, 200), 
                         (SCREEN_WIDTH - 330, 10, 320, 240), 2)
        
        # Draw calibration text if in calibration mode
        if self.game_state == GAME_STATE_CALIBRATION:
            if self.calibration_step == 1:
                text = "Show OPEN HAND and move around"
            elif self.calibration_step == 2:
                text = "Show CLOSED FIST and move around"
            else:
                text = "Calibrating..."
                
            text_surf = self.font.render(text, True, (255, 255, 0))
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
            self.screen.blit(text_surf, text_rect)
        
        # Draw game over text if game is over
        if self.game_state == GAME_STATE_GAME_OVER:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            # Draw game over text
            game_over_text = self.font.render(self.game_ui.game_state, True, (255, 255, 0))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(game_over_text, text_rect)
            
            # Draw restart instructions
            restart_text = self.font.render("Press 'R' to play again", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
            self.screen.blit(restart_text, restart_rect)
        
        # Update the display
        pygame.display.flip()
    
    def run(self):
        """Run the main game loop."""
        try:
            while self.running:
                # Process events
                self.process_events()
                
                # Update game state
                self.update()
                
                # Draw the game
                self.draw()
                
                # Cap the frame rate
                self.clock.tick(FPS)
        finally:
            # Clean up
            self.camera.release()
            pygame.quit()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create and run the game
    game = PoolGame()
    game.run()
