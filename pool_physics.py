import numpy as np
import pygame
import math


class Ball:
    def __init__(self, number, position, radius=15, color=(255, 255, 255), is_cue_ball=False):
        """
        Initialize a billiard ball.
        
        Args:
            number: Ball number (0 for cue ball, 1-7 for solids, 8 for 8-ball, 9-15 for stripes)
            position: Initial position (x, y)
            radius: Ball radius in pixels
            color: Ball color (RGB tuple)
            is_cue_ball: Whether this is the cue ball
        """
        self.number = number
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.radius = radius
        self.mass = 1.0  # All balls have the same mass
        self.color = color
        self.is_cue_ball = is_cue_ball
        self.in_pocket = False
        self.friction = 0.98  # Friction coefficient (velocity multiplier per frame)
        
        # Standard pool ball colors and patterns
        self.colors = {
            0: (255, 255, 255),  # Cue ball (white)
            1: (255, 255, 0),    # Yellow solid
            2: (0, 0, 255),      # Blue solid
            3: (255, 0, 0),      # Red solid
            4: (128, 0, 128),    # Purple solid
            5: (255, 165, 0),    # Orange solid
            6: (0, 128, 0),      # Green solid
            7: (128, 0, 0),      # Maroon solid
            8: (0, 0, 0),        # 8-ball (black)
            9: (255, 255, 0),    # Yellow stripe
            10: (0, 0, 255),     # Blue stripe
            11: (255, 0, 0),     # Red stripe
            12: (128, 0, 128),   # Purple stripe
            13: (255, 165, 0),   # Orange stripe
            14: (0, 128, 0),     # Green stripe
            15: (128, 0, 0),     # Maroon stripe
        }
        
        # Set the proper color based on ball number
        if number in self.colors:
            self.color = self.colors[number]
    
    def update(self, dt, table_width, table_height, pockets):
        """
        Update ball position and velocity.
        
        Args:
            dt: Time step
            table_width: Width of the pool table
            table_height: Height of the pool table
            pockets: List of pocket positions and radii [(x, y, radius), ...]
        
        Returns:
            True if the ball is in a pocket, False otherwise
        """
        if self.in_pocket:
            return True
        
        # Apply velocity to position
        self.position += self.velocity * dt
        
        # Apply friction
        self.velocity *= self.friction
        
        # Stop very slow balls to prevent endless tiny movements
        if np.linalg.norm(self.velocity) < 0.1:
            self.velocity = np.zeros(2)
        
        # Check for pocket collisions
        for pocket_x, pocket_y, pocket_radius in pockets:
            pocket_pos = np.array([pocket_x, pocket_y])
            distance = np.linalg.norm(self.position - pocket_pos)
            
            if distance < pocket_radius:
                self.in_pocket = True
                self.velocity = np.zeros(2)
                return True
        
        # Check for wall collisions with cushion bounce
        # Left wall
        if self.position[0] - self.radius < 0:
            self.position[0] = self.radius
            self.velocity[0] *= -0.8  # Bounce with some energy loss
        
        # Right wall
        if self.position[0] + self.radius > table_width:
            self.position[0] = table_width - self.radius
            self.velocity[0] *= -0.8
        
        # Top wall
        if self.position[1] - self.radius < 0:
            self.position[1] = self.radius
            self.velocity[1] *= -0.8
        
        # Bottom wall
        if self.position[1] + self.radius > table_height:
            self.position[1] = table_height - self.radius
            self.velocity[1] *= -0.8
        
        return False
    
    def check_collision(self, other_ball):
        """
        Check and resolve collision with another ball.
        
        Args:
            other_ball: The other ball to check collision with
            
        Returns:
            True if collision occurred, False otherwise
        """
        if self.in_pocket or other_ball.in_pocket:
            return False
        
        # Vector from this ball to the other ball
        delta_pos = other_ball.position - self.position
        
        # Distance between ball centers
        distance = np.linalg.norm(delta_pos)
        
        # Check if balls are overlapping
        if distance < self.radius + other_ball.radius:
            # Normalize the collision vector
            if distance == 0:  # Avoid division by zero
                collision_vec = np.array([1, 0])
            else:
                collision_vec = delta_pos / distance
            
            # Calculate relative velocity
            relative_vel = self.velocity - other_ball.velocity
            
            # Calculate velocity along the collision vector
            velocity_along_collision = np.dot(relative_vel, collision_vec)
            
            # Only resolve collision if balls are moving toward each other
            if velocity_along_collision < 0:
                # Calculate impulse
                impulse = 2.0 * velocity_along_collision / (self.mass + other_ball.mass)
                
                # Apply impulse to the velocities
                self.velocity -= impulse * other_ball.mass * collision_vec
                other_ball.velocity += impulse * self.mass * collision_vec
                
                # Separate balls to prevent sticking
                overlap = self.radius + other_ball.radius - distance
                self.position -= overlap * 0.5 * collision_vec
                other_ball.position += overlap * 0.5 * collision_vec
                
                # Add a small energy loss in collisions
                self.velocity *= 0.98
                other_ball.velocity *= 0.98
                
                return True
        
        return False
    
    def draw(self, surface):
        """
        Draw the ball on the given surface.
        
        Args:
            surface: Pygame surface to draw on
        """
        if self.in_pocket:
            return
        
        # Draw the ball
        pygame.draw.circle(surface, self.color, self.position.astype(int), self.radius)
        
        # Add stripe for striped balls (9-15)
        if 9 <= self.number <= 15:
            # Draw a white stripe in the middle
            pygame.draw.circle(surface, (255, 255, 255), self.position.astype(int), self.radius * 0.7)
            pygame.draw.circle(surface, self.color, self.position.astype(int), self.radius * 0.5)
        
        # Draw the number on the ball (except for the cue ball)
        if not self.is_cue_ball:
            # Choose text color for good contrast
            text_color = (255, 255, 255) if self.number == 8 else (0, 0, 0)
            font = pygame.font.Font(None, self.radius)
            text = font.render(str(self.number), True, text_color)
            text_rect = text.get_rect(center=self.position.astype(int))
            surface.blit(text, text_rect)


class PoolTable:
    def __init__(self, width=800, height=400, cushion_color=(0, 100, 0), felt_color=(0, 150, 0)):
        """
        Initialize the pool table.
        
        Args:
            width: Table width in pixels
            height: Table height in pixels
            cushion_color: Color of the table cushions
            felt_color: Color of the table felt
        """
        self.width = width
        self.height = height
        self.cushion_color = cushion_color
        self.felt_color = felt_color
        self.cushion_width = 20
        
        # Define pockets positions and size
        # Format: (x, y, radius)
        self.pockets = [
            (0, 0, 25),                      # Top-left pocket
            (width // 2, 0, 25),             # Top-middle pocket
            (width, 0, 25),                  # Top-right pocket
            (0, height, 25),                 # Bottom-left pocket
            (width // 2, height, 25),        # Bottom-middle pocket
            (width, height, 25)              # Bottom-right pocket
        ]
        
        # Initialize balls
        self.balls = []
        self.reset_balls()
    
    def reset_balls(self):
        """Reset all balls to their initial positions."""
        self.balls = []
        
        # Create cue ball
        cue_ball = Ball(0, (self.width * 0.25, self.height / 2), is_cue_ball=True)
        self.balls.append(cue_ball)
        
        # Create the triangle rack of 15 balls
        rack_center_x = self.width * 0.75
        rack_center_y = self.height / 2
        
        # Distance between ball centers
        ball_distance = 30
        
        # Create the rack formation (5 rows)
        rack_positions = []
        
        # First row (1 ball)
        rack_positions.append((rack_center_x, rack_center_y))
        
        # Second row (2 balls)
        row_y = rack_center_y - ball_distance * math.sin(math.pi / 3)
        rack_positions.append((rack_center_x + ball_distance, row_y))
        rack_positions.append((rack_center_x + ball_distance, rack_center_y + (rack_center_y - row_y)))
        
        # Third row (3 balls)
        row_y = rack_center_y - 2 * ball_distance * math.sin(math.pi / 3)
        for i in range(3):
            y = row_y + i * ball_distance * math.sin(math.pi / 3) * 2
            rack_positions.append((rack_center_x + 2 * ball_distance, y))
        
        # Fourth row (4 balls)
        row_y = rack_center_y - 3 * ball_distance * math.sin(math.pi / 3)
        for i in range(4):
            y = row_y + i * ball_distance * math.sin(math.pi / 3) * 2
            rack_positions.append((rack_center_x + 3 * ball_distance, y))
        
        # Fifth row (5 balls)
        row_y = rack_center_y - 4 * ball_distance * math.sin(math.pi / 3)
        for i in range(5):
            y = row_y + i * ball_distance * math.sin(math.pi / 3) * 2
            rack_positions.append((rack_center_x + 4 * ball_distance, y))
        
        # Create the remaining balls with a mix of solids and stripes
        # 8-ball should be in the center of the third row
        
        # Arrange balls: solids (1-7), 8-ball, and stripes (9-15)
        ball_numbers = list(range(1, 8)) + [8] + list(range(9, 16))
        
        # Place 8-ball in the center
        center_position_index = 0  # First position (apex)
        
        # Shuffle the remaining numbers (except 8)
        remaining_numbers = ball_numbers.copy()
        remaining_numbers.remove(8)
        np.random.shuffle(remaining_numbers)
        
        # Place 8-ball in the center
        self.balls.append(Ball(8, rack_positions[center_position_index]))
        
        # Place the rest of the balls
        for i, pos in enumerate(rack_positions):
            if i != center_position_index:
                if remaining_numbers:
                    self.balls.append(Ball(remaining_numbers.pop(0), pos))
    
    def update(self, dt):
        """
        Update all balls on the table.
        
        Args:
            dt: Time step
            
        Returns:
            List of ball numbers that went into pockets this update
        """
        pocketed_balls = []
        
        # Update ball positions
        for ball in self.balls:
            was_in_pocket = ball.in_pocket
            is_in_pocket = ball.update(dt, self.width, self.height, self.pockets)
            
            # Check if ball just went into a pocket
            if is_in_pocket and not was_in_pocket:
                pocketed_balls.append(ball.number)
        
        # Check for collisions between all ball pairs
        for i in range(len(self.balls)):
            for j in range(i+1, len(self.balls)):
                self.balls[i].check_collision(self.balls[j])
        
        return pocketed_balls
    
    def draw(self, surface):
        """
        Draw the pool table and all balls.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw table cushions
        pygame.draw.rect(surface, self.cushion_color, (0, 0, self.width, self.height), self.cushion_width)
        
        # Draw table felt
        pygame.draw.rect(surface, self.felt_color, 
                         (self.cushion_width, self.cushion_width, 
                          self.width - 2*self.cushion_width, 
                          self.height - 2*self.cushion_width))
        
        # Draw pockets
        for x, y, radius in self.pockets:
            pygame.draw.circle(surface, (0, 0, 0), (int(x), int(y)), radius)
        
        # Draw all balls
        for ball in self.balls:
            ball.draw(surface)
    
    def get_cue_ball(self):
        """
        Get the cue ball.
        
        Returns:
            The cue ball object or None if it's not available
        """
        for ball in self.balls:
            if ball.is_cue_ball:
                return ball
        return None
    
    def are_balls_moving(self):
        """
        Check if any balls are currently moving.
        
        Returns:
            True if any ball is moving, False otherwise
        """
        for ball in self.balls:
            if not ball.in_pocket and np.linalg.norm(ball.velocity) > 0.1:
                return True
        return False
    
    def strike_cue_ball(self, direction, power):
        """
        Strike the cue ball in the given direction with the given power.
        
        Args:
            direction: Direction vector (normalized)
            power: Strike power (0.0 to 1.0)
            
        Returns:
            True if the cue ball was struck, False otherwise
        """
        cue_ball = self.get_cue_ball()
        if cue_ball and not cue_ball.in_pocket and not self.are_balls_moving():
            # Scale power (adjust as needed for good gameplay)
            max_speed = 800.0  # Maximum speed
            speed = power * max_speed
            
            # Set cue ball velocity
            cue_ball.velocity = np.array(direction) * speed
            return True
        
        return False
