import pygame
import numpy as np
import math


class CueStick:
    def __init__(self, length=200, width=6, color=(150, 75, 0)):
        """
        Initialize the cue stick.
        
        Args:
            length: Length of the cue stick in pixels
            width: Width of the cue stick in pixels
            color: Color of the cue stick (RGB tuple)
        """
        self.length = length
        self.width = width
        self.color = color
        self.angle = 0  # Angle in radians
        self.distance = 40  # Distance from cue ball
        self.max_pull_distance = 100  # Maximum pull back distance for power
        self.pull_distance = 0  # Current pull back distance
        self.visible = True
        
    def update(self, cue_ball_pos, target_angle=None, pull_back=0.0):
        """
        Update the cue stick position and orientation.
        
        Args:
            cue_ball_pos: Position of the cue ball (x, y)
            target_angle: Target angle in radians (if None, keep current angle)
            pull_back: Pull back amount for power (0.0 to 1.0)
        """
        if target_angle is not None:
            # Smoothly rotate towards the target angle
            angle_diff = target_angle - self.angle
            
            # Normalize angle difference to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Apply smooth rotation (adjust 0.1 for faster/slower rotation)
            self.angle += angle_diff * 0.1
            
        # Update pull back distance
        self.pull_distance = pull_back * self.max_pull_distance
    
    def draw(self, surface, cue_ball_pos):
        """
        Draw the cue stick on the given surface.
        
        Args:
            surface: Pygame surface to draw on
            cue_ball_pos: Position of the cue ball (x, y)
        """
        if not self.visible:
            return
        
        # Calculate cue stick end points
        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        
        # Start point (near cue ball)
        start_point = np.array(cue_ball_pos) + direction * (self.distance + self.pull_distance)
        
        # End point
        end_point = start_point + direction * self.length
        
        # Draw cue stick
        pygame.draw.line(surface, self.color, 
                         start_point.astype(int), 
                         end_point.astype(int), 
                         self.width)
        
        # Draw cue tip
        pygame.draw.circle(surface, (200, 200, 200), 
                           start_point.astype(int), 
                           self.width // 2)
    
    def get_direction_vector(self):
        """
        Get the direction vector of the cue stick.
        
        Returns:
            Normalized direction vector (away from the cue ball)
        """
        # Direction is opposite to the visual cue stick direction
        # because we want to hit the ball in the opposite direction
        return np.array([-math.cos(self.angle), -math.sin(self.angle)])


class PowerMeter:
    def __init__(self, max_width=150, height=20, position=(50, 50)):
        """
        Initialize the power meter.
        
        Args:
            max_width: Maximum width of the power meter
            height: Height of the power meter
            position: Position of the power meter (x, y) - top left corner
        """
        self.max_width = max_width
        self.height = height
        self.position = position
        self.power = 0.0  # Current power (0.0 to 1.0)
        self.visible = False
        
        # Define colors for the power gradient (green to red)
        self.colors = [
            (0, 255, 0),    # Green (low power)
            (255, 255, 0),  # Yellow (medium power)
            (255, 0, 0)     # Red (high power)
        ]
    
    def update(self, power):
        """
        Update the power meter.
        
        Args:
            power: New power value (0.0 to 1.0)
        """
        self.power = max(0.0, min(1.0, power))  # Clamp to [0, 1]
    
    def draw(self, surface):
        """
        Draw the power meter on the given surface.
        
        Args:
            surface: Pygame surface to draw on
        """
        if not self.visible:
            return
        
        # Draw power meter background
        pygame.draw.rect(surface, (50, 50, 50), 
                         (self.position[0], self.position[1], 
                          self.max_width, self.height))
        
        # Draw power meter fill
        fill_width = int(self.power * self.max_width)
        if fill_width > 0:
            # Calculate color based on power
            if self.power < 0.5:
                # Interpolate between green and yellow
                t = self.power * 2
                color = tuple(int(self.colors[0][i] * (1 - t) + self.colors[1][i] * t) for i in range(3))
            else:
                # Interpolate between yellow and red
                t = (self.power - 0.5) * 2
                color = tuple(int(self.colors[1][i] * (1 - t) + self.colors[2][i] * t) for i in range(3))
                
            pygame.draw.rect(surface, color, 
                             (self.position[0], self.position[1], 
                              fill_width, self.height))
        
        # Draw power meter border
        pygame.draw.rect(surface, (200, 200, 200), 
                         (self.position[0], self.position[1], 
                          self.max_width, self.height), 
                         2)
        
        # Draw power text
        font = pygame.font.Font(None, 24)
        power_text = f"Power: {int(self.power * 100)}%"
        text = font.render(power_text, True, (255, 255, 255))
        text_rect = text.get_rect(midleft=(self.position[0] + self.max_width + 10, 
                                          self.position[1] + self.height // 2))
        surface.blit(text, text_rect)


class GameUI:
    def __init__(self, screen_width, screen_height):
        """
        Initialize the game UI.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create UI components
        self.cue_stick = CueStick()
        self.power_meter = PowerMeter(position=(50, screen_height - 40))
        
        # Game state text
        self.game_state = "Player 1's Turn"
        self.calibration_text = ""
        
        # Font for text rendering
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Score tracking
        self.player1_score = 0
        self.player2_score = 0
        self.current_player = 1
        
        # Help text
        self.show_help = True
        self.help_text = [
            "Controls:",
            "- Hand gestures control cue position and power",
            "- Open hand: Position the cue stick",
            "- Move hand left/right: Adjust angle",
            "- Closed fist to open hand: Strike the ball",
            "- Press 'C' to calibrate hand gestures",
            "- Press 'R' to reset the game",
            "- Press 'H' to toggle this help",
            "- Press 'ESC' to exit"
        ]
    
    def update(self, cue_ball_pos=None, target_angle=None, power=None, 
               game_state=None, calibration_mode=False, calibration_text=""):
        """
        Update the game UI.
        
        Args:
            cue_ball_pos: Position of the cue ball
            target_angle: Target angle for the cue stick
            power: Current power level (0.0 to 1.0)
            game_state: Current game state text
            calibration_mode: Whether calibration mode is active
            calibration_text: Text to display during calibration
        """
        # Update cue stick if cue ball position is provided
        if cue_ball_pos is not None:
            self.cue_stick.update(cue_ball_pos, target_angle, power if power else 0.0)
            
            # Show/hide cue stick based on game state
            self.cue_stick.visible = not calibration_mode
        
        # Update power meter
        if power is not None:
            self.power_meter.update(power)
            self.power_meter.visible = not calibration_mode and power > 0.0
        
        # Update game state text
        if game_state is not None:
            self.game_state = game_state
        
        # Update calibration text
        self.calibration_text = calibration_text
    
    def draw(self, surface, cue_ball_pos=None):
        """
        Draw the game UI on the given surface.
        
        Args:
            surface: Pygame surface to draw on
            cue_ball_pos: Position of the cue ball
        """
        # Draw cue stick if cue ball position is provided
        if cue_ball_pos is not None and self.cue_stick.visible:
            self.cue_stick.draw(surface, cue_ball_pos)
        
        # Draw power meter
        self.power_meter.draw(surface)
        
        # Draw game state text
        state_text = self.font.render(self.game_state, True, (255, 255, 255))
        state_rect = state_text.get_rect(midtop=(self.screen_width // 2, 10))
        surface.blit(state_text, state_rect)
        
        # Draw score
        score_text = self.font.render(
            f"Player 1: {self.player1_score}  |  Player 2: {self.player2_score}", 
            True, (255, 255, 255))
        score_rect = score_text.get_rect(midtop=(self.screen_width // 2, 50))
        surface.blit(score_text, score_rect)
        
        # Draw calibration text if provided
        if self.calibration_text:
            calib_text = self.font.render(self.calibration_text, True, (255, 255, 0))
            calib_rect = calib_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            surface.blit(calib_text, calib_rect)
        
        # Draw help text if enabled
        if self.show_help:
            # Draw semi-transparent background
            help_surface = pygame.Surface((400, 250), pygame.SRCALPHA)
            help_surface.fill((0, 0, 0, 150))
            surface.blit(help_surface, (10, 10))
            
            # Draw help text
            for i, line in enumerate(self.help_text):
                text = self.small_font.render(line, True, (255, 255, 255))
                surface.blit(text, (20, 20 + i * 25))
    
    def get_cue_direction(self):
        """
        Get the current direction vector of the cue stick.
        
        Returns:
            Normalized direction vector
        """
        return self.cue_stick.get_direction_vector()
    
    def get_power(self):
        """
        Get the current power level.
        
        Returns:
            Power level (0.0 to 1.0)
        """
        return self.power_meter.power
    
    def toggle_help(self):
        """Toggle help text visibility."""
        self.show_help = not self.show_help
        
    def update_score(self, player, points):
        """
        Update a player's score.
        
        Args:
            player: Player number (1 or 2)
            points: Points to add
        """
        if player == 1:
            self.player1_score += points
        else:
            self.player2_score += points
    
    def switch_player(self):
        """Switch the current player."""
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        self.game_state = f"Player {self.current_player}'s Turn"
    
    def reset_scores(self):
        """Reset both players' scores."""
        self.player1_score = 0
        self.player2_score = 0
        self.current_player = 1
        self.game_state = "Player 1's Turn"
