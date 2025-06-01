import pygame
import numpy as np
import math
import time
from typing import Tuple, Optional, List


class ParticleSystem:
    """Advanced particle system for visual effects"""
    
    def __init__(self):
        self.particles = []
        
    def emit(self, pos: Tuple[float, float], count: int = 10, 
             color: Tuple[int, int, int] = (255, 255, 255),
             velocity_range: float = 50, life_span: float = 1.0):
        """Emit particles from a position"""
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(10, velocity_range)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            
            particle = {
                'pos': np.array(pos, dtype=float),
                'velocity': velocity,
                'color': color,
                'life': life_span,
                'max_life': life_span,
                'size': np.random.uniform(2, 6)
            }
            self.particles.append(particle)
    
    def update(self, dt: float):
        """Update all particles"""
        for particle in self.particles[:]:
            particle['life'] -= dt
            if particle['life'] <= 0:
                self.particles.remove(particle)
                continue
                
            # Update position
            particle['pos'] += particle['velocity'] * dt
            
            # Apply gravity and friction
            particle['velocity'][1] += 200 * dt  # Gravity
            particle['velocity'] *= 0.98  # Friction
    
    def draw(self, surface):
        """Draw all particles with fade effect"""
        for particle in self.particles:
            alpha = particle['life'] / particle['max_life']
            size = particle['size'] * alpha
            if size > 0.5:
                color = (*particle['color'], int(255 * alpha))
                # Create a surface with per-pixel alpha
                particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                surface.blit(particle_surf, particle['pos'] - size)


class AnimatedElement:
    """Base class for animated UI elements"""
    
    def __init__(self, duration: float = 1.0):
        self.start_time = time.time()
        self.duration = duration
        self.progress = 0.0
        
    def update(self):
        """Update animation progress"""
        elapsed = time.time() - self.start_time
        self.progress = min(1.0, elapsed / self.duration)
        
    def ease_out_cubic(self, t: float) -> float:
        """Smooth easing function"""
        return 1 - pow(1 - t, 3)
    
    def ease_in_out_cubic(self, t: float) -> float:
        """Smooth bidirectional easing"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2


class Enhanced3DCueStick:
    """Professional 3D-style cue stick with advanced visuals"""
    
    def __init__(self, length=240, width=8, color=(139, 69, 19)):
        self.length = length
        self.width = width
        self.base_color = color
        self.angle = 0
        self.distance = 45
        self.max_pull_distance = 120
        self.pull_distance = 0
        self.visible = True
        
        # Animation properties
        self.target_angle = 0
        self.angle_velocity = 0
        self.smooth_factor = 0.15
        
        # Visual enhancement properties
        self.tip_glow = 0.0
        self.shaft_segments = 8
        self.wood_grain_offset = 0
        
        # Strike animation
        self.strike_animation = AnimatedElement(0.3)
        self.is_striking = False
        
    def update(self, cue_ball_pos, target_angle=None, pull_back=0.0):
        """Update with smooth physics-based movement"""
        if target_angle is not None:
            self.target_angle = target_angle
            
        # Smooth angle interpolation with physics
        angle_diff = self.target_angle - self.angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Apply spring physics for smooth movement
        self.angle_velocity += angle_diff * 0.2
        self.angle_velocity *= 0.85  # Damping
        self.angle += self.angle_velocity * 0.1
        
        # Update pull distance with easing
        target_pull = pull_back * self.max_pull_distance
        self.pull_distance += (target_pull - self.pull_distance) * 0.2
        
        # Update tip glow based on power
        self.tip_glow = pull_back * 255
        
        # Update wood grain animation
        self.wood_grain_offset += 0.5
        
        # Update strike animation
        if self.is_striking:
            self.strike_animation.update()
            if self.strike_animation.progress >= 1.0:
                self.is_striking = False
                
    def start_strike_animation(self):
        """Start the strike animation"""
        self.is_striking = True
        self.strike_animation = AnimatedElement(0.3)
        
    def draw_3d_cylinder(self, surface, start_pos, end_pos, radius, base_color):
        """Draw a 3D-looking cylinder"""
        # Calculate direction and length
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        
        if length == 0:
            return
            
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])
        
        # Create multiple segments for 3D effect
        segments = self.shaft_segments
        colors = []
        
        for i in range(segments):
            # Create gradient effect
            angle = i * 2 * math.pi / segments
            light_factor = 0.7 + 0.3 * math.cos(angle + math.pi/4)
            
            color = tuple(int(c * light_factor) for c in base_color)
            colors.append(color)
            
            # Calculate segment vertices
            offset1 = perpendicular * radius * math.cos(angle)
            offset2 = perpendicular * radius * math.cos(angle + 2 * math.pi / segments)
            
            # Draw segment as a quad
            points = [
                start_pos + offset1,
                start_pos + offset2,
                end_pos + offset2,
                end_pos + offset1
            ]
            
            pygame.draw.polygon(surface, color, points.astype(int))
            
        # Add wood grain effect
        for i in range(5):
            grain_offset = (self.wood_grain_offset + i * 20) % length
            grain_pos = start_pos + direction * grain_offset
            grain_color = tuple(max(0, c - 20) for c in base_color)
            
            # Draw wood grain lines
            grain_perp = perpendicular * (radius * 0.8)
            pygame.draw.line(surface, grain_color,
                           (grain_pos + grain_perp).astype(int),
                           (grain_pos - grain_perp).astype(int), 2)
    
    def draw(self, surface, cue_ball_pos):
        """Draw enhanced 3D cue stick"""
        if not self.visible:
            return
            
        # Calculate positions
        direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        
        # Strike animation offset
        strike_offset = 0
        if self.is_striking:
            progress = self.strike_animation.ease_out_cubic(self.strike_animation.progress)
            strike_offset = -30 * progress
        
        start_point = np.array(cue_ball_pos) + direction * (self.distance + self.pull_distance + strike_offset)
        end_point = start_point + direction * self.length
        
        # Draw cue stick shadow
        shadow_offset = np.array([3, 3])
        shadow_start = start_point + shadow_offset
        shadow_end = end_point + shadow_offset
        self.draw_3d_cylinder(surface, shadow_start, shadow_end, 
                             self.width // 2, (50, 50, 50))
        
        # Draw main cue stick shaft with segments
        segment_length = self.length // 3
        
        # Handle section (darker wood)
        handle_end = start_point + direction * segment_length
        self.draw_3d_cylinder(surface, start_point, handle_end,
                             self.width // 2, (101, 67, 33))
        
        # Middle section (medium wood)
        middle_start = handle_end
        middle_end = handle_end + direction * segment_length
        self.draw_3d_cylinder(surface, middle_start, middle_end,
                             self.width // 2 - 1, (139, 90, 43))
        
        # Tip section (lighter wood)
        tip_start = middle_end
        self.draw_3d_cylinder(surface, tip_start, end_point,
                             self.width // 2 - 2, (160, 120, 80))
        
        # Draw leather tip with glow effect
        tip_radius = self.width // 2 + 2
        tip_color = (150, 75, 50)
        
        # Glow effect
        if self.tip_glow > 0:
            glow_radius = tip_radius + int(self.tip_glow / 50)
            glow_color = (255, 200, 100, int(self.tip_glow))
            glow_surf = pygame.Surface((glow_radius * 4, glow_radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, 
                             (glow_radius * 2, glow_radius * 2), glow_radius)
            surface.blit(glow_surf, start_point.astype(int) - glow_radius * 2)
        
        # Main tip
        pygame.draw.circle(surface, tip_color, start_point.astype(int), tip_radius)
        
        # Tip highlight
        highlight_pos = start_point + np.array([-2, -2])
        pygame.draw.circle(surface, (200, 150, 100), highlight_pos.astype(int), tip_radius - 2)
        
        # Draw decorative rings
        for i, ring_pos in enumerate([0.3, 0.7]):
            ring_center = start_point + direction * (self.length * ring_pos)
            ring_colors = [(255, 215, 0), (192, 192, 192), (255, 215, 0)]  # Gold-Silver-Gold
            for j, ring_color in enumerate(ring_colors):
                pygame.draw.circle(surface, ring_color, ring_center.astype(int), 
                                 self.width // 2 + 2 - j, 2)
    
    def get_direction_vector(self):
        return np.array([-math.cos(self.angle), -math.sin(self.angle)])


class Premium3DPowerMeter:
    """Professional 3D power meter with advanced effects"""
    
    def __init__(self, max_width=200, height=25, position=(60, 60)):
        self.max_width = max_width
        self.height = height
        self.position = position
        self.power = 0.0
        self.visible = False
        self.target_power = 0.0
        
        # Animation properties
        self.power_animation = AnimatedElement(0.2)
        self.pulse_time = 0
        
        # Visual effects
        self.gradient_colors = [
            (46, 204, 113),   # Emerald green
            (241, 196, 15),   # Golden yellow
            (231, 76, 60),    # Crimson red
            (155, 89, 182)    # Purple (overpowered)
        ]
        
        self.glow_intensity = 0.0
        
    def update(self, power):
        """Update with smooth animations"""
        self.target_power = max(0.0, min(1.0, power))
        
        # Smooth power interpolation
        self.power += (self.target_power - self.power) * 0.15
        
        # Update pulse animation
        self.pulse_time += 0.1
        
        # Calculate glow intensity
        self.glow_intensity = self.power * math.sin(self.pulse_time * 5) * 0.3 + self.power * 0.7
        
    def draw_3d_frame(self, surface, rect, depth=4):
        """Draw 3D frame effect"""
        x, y, w, h = rect
        
        # Outer shadow
        shadow_rect = (x + depth, y + depth, w, h)
        pygame.draw.rect(surface, (20, 20, 20), shadow_rect)
        
        # Main frame
        pygame.draw.rect(surface, (60, 60, 60), rect)
        
        # Inner bevel (top-left highlight)
        pygame.draw.lines(surface, (120, 120, 120), False, 
                         [(x, y + h), (x, y), (x + w, y)], 2)
        
        # Inner bevel (bottom-right shadow)
        pygame.draw.lines(surface, (30, 30, 30), False,
                         [(x + w, y), (x + w, y + h), (x, y + h)], 2)
    
    def get_power_color(self, power_level):
        """Get interpolated color based on power level"""
        if power_level <= 0.33:
            # Green to Yellow
            t = power_level / 0.33
            return tuple(int(self.gradient_colors[0][i] * (1-t) + self.gradient_colors[1][i] * t) 
                        for i in range(3))
        elif power_level <= 0.66:
            # Yellow to Red
            t = (power_level - 0.33) / 0.33
            return tuple(int(self.gradient_colors[1][i] * (1-t) + self.gradient_colors[2][i] * t) 
                        for i in range(3))
        else:
            # Red to Purple (danger zone)
            t = (power_level - 0.66) / 0.34
            return tuple(int(self.gradient_colors[2][i] * (1-t) + self.gradient_colors[3][i] * t) 
                        for i in range(3))
    
    def draw(self, surface):
        """Draw premium 3D power meter"""
        if not self.visible:
            return
            
        x, y = self.position
        
        # Draw 3D frame
        frame_rect = (x - 3, y - 3, self.max_width + 6, self.height + 6)
        self.draw_3d_frame(surface, frame_rect)
        
        # Draw background with subtle gradient
        bg_rect = (x, y, self.max_width, self.height)
        pygame.draw.rect(surface, (25, 25, 25), bg_rect)
        
        # Draw power fill with advanced effects
        fill_width = int(self.power * self.max_width)
        if fill_width > 0:
            # Get base color
            base_color = self.get_power_color(self.power)
            
            # Create gradient fill
            for i in range(fill_width):
                progress = i / fill_width if fill_width > 0 else 0
                
                # Vertical gradient
                for j in range(self.height):
                    vertical_progress = j / self.height
                    
                    # Calculate color with gradient
                    brightness = 0.6 + 0.4 * (1 - vertical_progress)
                    color = tuple(int(c * brightness) for c in base_color)
                    
                    # Add glow effect
                    if self.glow_intensity > 0:
                        glow_factor = self.glow_intensity * (1 - abs(vertical_progress - 0.5) * 2)
                        color = tuple(min(255, int(c + glow_factor * 50)) for c in color)
                    
                    pygame.draw.rect(surface, color, (x + i, y + j, 1, 1))
        
        # Draw power level indicators
        for i in range(1, 4):
            indicator_x = x + (self.max_width * i // 4)
            color = (100, 100, 100) if self.power < i/4 else (200, 200, 200)
            pygame.draw.line(surface, color, 
                           (indicator_x, y), (indicator_x, y + self.height), 1)
        
        # Draw glass reflection effect
        reflection_rect = (x, y, fill_width, self.height // 3)
        reflection_surf = pygame.Surface((fill_width, self.height // 3), pygame.SRCALPHA)
        reflection_surf.fill((255, 255, 255, 40))
        surface.blit(reflection_surf, (x, y))
        
        # Draw percentage text with shadow
        font = pygame.font.Font(None, 28)
        power_text = f"{int(self.power * 100)}%"
        
        # Text shadow
        shadow_text = font.render(power_text, True, (0, 0, 0))
        surface.blit(shadow_text, (x + self.max_width + 12, y + 2))
        
        # Main text
        text_color = (255, 255, 255) if self.power < 0.8 else (255, 200, 100)
        main_text = font.render(power_text, True, text_color)
        surface.blit(main_text, (x + self.max_width + 10, y))
        
        # Power label
        label_font = pygame.font.Font(None, 20)
        label_text = label_font.render("POWER", True, (150, 150, 150))
        surface.blit(label_text, (x, y - 25))


class PremiumGameUI:
    """Premium professional game UI system"""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Enhanced UI components
        self.cue_stick = Enhanced3DCueStick()
        self.power_meter = Premium3DPowerMeter(position=(60, screen_height - 60))
        self.particle_system = ParticleSystem()
        
        # Game state
        self.game_state = "Player 1's Turn"
        self.calibration_text = ""
        
        # Premium fonts
        pygame.font.init()
        self.title_font = pygame.font.Font(None, 48)
        self.main_font = pygame.font.Font(None, 36)
        self.ui_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Score system
        self.player1_score = 0
        self.player2_score = 0
        self.current_player = 1
        
        # UI animations
        self.ui_animations = {}
        self.background_time = 0
        
        # Enhanced help system
        self.show_help = True
        self.help_panel_alpha = 0
        self.target_help_alpha = 200
        
        self.help_sections = {
            "Controls": [
                "Hand gestures control cue position and power",
                "Open hand: Position the cue stick",
                "Move hand left/right: Adjust angle",
                "Closed fist to open hand: Strike the ball"
            ],
            "Keyboard": [
                "Press 'C' to calibrate hand gestures",
                "Press 'R' to reset the game",
                "Press 'H' to toggle this help",
                "Press 'ESC' to exit"
            ]
        }
        
        # Professional color scheme
        self.colors = {
            'primary': (52, 152, 219),      # Modern blue
            'secondary': (46, 204, 113),    # Emerald green
            'accent': (241, 196, 15),       # Golden yellow
            'danger': (231, 76, 60),        # Crimson red
            'dark': (44, 62, 80),           # Dark blue-gray
            'light': (236, 240, 241),       # Off-white
            'glass': (255, 255, 255, 60)    # Transparent white
        }
        
    def create_glass_surface(self, size, alpha=60):
        """Create glassmorphism effect surface"""
        surface = pygame.Surface(size, pygame.SRCALPHA)
        surface.fill((255, 255, 255, alpha))
        return surface
        
    def draw_gradient_rect(self, surface, rect, color1, color2, vertical=True):
        """Draw gradient rectangle"""
        x, y, w, h = rect
        if vertical:
            for i in range(h):
                ratio = i / h
                color = tuple(int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3))
                pygame.draw.line(surface, color, (x, y + i), (x + w, y + i))
        else:
            for i in range(w):
                ratio = i / w
                color = tuple(int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3))
                pygame.draw.line(surface, color, (x + i, y), (x + i, y + h))
    
    def draw_premium_panel(self, surface, rect, title="", content_lines=None):
        """Draw premium glass panel with title"""
        x, y, w, h = rect
        
        # Glass background with blur effect
        glass_surf = self.create_glass_surface((w, h), 40)
        surface.blit(glass_surf, (x, y))
        
        # Border with gradient
        border_color1 = (255, 255, 255, 100)
        border_color2 = (255, 255, 255, 20)
        
        # Draw border segments
        pygame.draw.rect(surface, border_color1[:3], (x, y, w, h), 2)
        
        # Title bar
        if title:
            title_height = 35
            title_rect = (x, y, w, title_height)
            self.draw_gradient_rect(surface, title_rect, 
                                  self.colors['primary'], 
                                  (self.colors['primary'][0] - 30, 
                                   self.colors['primary'][1] - 30, 
                                   self.colors['primary'][2] - 30))
            
            # Title text
            title_text = self.ui_font.render(title, True, (255, 255, 255))
            title_text_rect = title_text.get_rect(center=(x + w//2, y + title_height//2))
            surface.blit(title_text, title_text_rect)
            
            # Content area
            content_y = y + title_height + 10
        else:
            content_y = y + 10
            
        # Draw content
        if content_lines:
            for i, line in enumerate(content_lines):
                text = self.small_font.render(line, True, (255, 255, 255))
                surface.blit(text, (x + 15, content_y + i * 22))
    
    def update(self, cue_ball_pos=None, target_angle=None, power=None, 
               game_state=None, calibration_mode=False, calibration_text=""):
        """Update all UI components"""
        # Update background animation
        self.background_time += 0.02
        
        # Update cue stick
        if cue_ball_pos is not None:
            self.cue_stick.update(cue_ball_pos, target_angle, power if power else 0.0)
            self.cue_stick.visible = not calibration_mode
        
        # Update power meter
        if power is not None:
            self.power_meter.update(power)
            self.power_meter.visible = not calibration_mode and power > 0.0
            
        # Emit particles on high power
        if power and power > 0.8 and cue_ball_pos:
            self.particle_system.emit(cue_ball_pos, count=2, 
                                    color=(255, 200, 100), 
                                    velocity_range=30, life_span=0.5)
        
        # Update particle system
        self.particle_system.update(0.016)  # ~60 FPS
        
        # Update game state
        if game_state is not None:
            self.game_state = game_state
        
        self.calibration_text = calibration_text
        
        # Update help panel animation
        if self.show_help:
            self.help_panel_alpha = min(self.target_help_alpha, self.help_panel_alpha + 10)
        else:
            self.help_panel_alpha = max(0, self.help_panel_alpha - 15)
    
    def draw_animated_background(self, surface):
        """Draw animated background with subtle effects"""
        # Create subtle moving pattern
        for i in range(0, self.screen_width, 50):
            for j in range(0, self.screen_height, 50):
                offset = math.sin(self.background_time + i * 0.01 + j * 0.01) * 10
                alpha = int(20 + offset)
                if alpha > 0:
                    color = (255, 255, 255, alpha)
                    dot_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
                    dot_surf.fill(color)
                    surface.blit(dot_surf, (i + offset, j + offset))
    
    def draw_hud(self, surface):
        """Draw heads-up display"""
        # Top panel with game state
        top_panel_rect = (self.screen_width//2 - 200, 10, 400, 80)
        self.draw_premium_panel(surface, top_panel_rect)
        
        # Game state text with glow effect
        state_text = self.main_font.render(self.game_state, True, (255, 255, 255))
        
        # Text glow
        glow_text = self.main_font.render(self.game_state, True, self.colors['primary'])
        for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
            surface.blit(glow_text, (self.screen_width//2 - state_text.get_width()//2 + offset[0], 
                                   25 + offset[1]))
        
        # Main text
        surface.blit(state_text, (self.screen_width//2 - state_text.get_width()//2, 25))
        
        # Score display
        score_text = f"Player 1: {self.player1_score}  |  Player 2: {self.player2_score}"
        score_surface = self.ui_font.render(score_text, True, (200, 200, 200))
        surface.blit(score_surface, (self.screen_width//2 - score_surface.get_width()//2, 55))
    
    def draw_help_panel(self, surface):
        """Draw premium help panel"""
        if self.help_panel_alpha <= 0:
            return
            
        # Help panel dimensions
        panel_width = 450
        panel_height = 300
        panel_x = self.screen_width - panel_width - 20
        panel_y = 20
        
        # Create help surface with alpha
        help_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        help_surface.fill((0, 0, 0, self.help_panel_alpha))
        
        # Draw glass effect
        glass_surf = self.create_glass_surface((panel_width, panel_height), 
                                             int(self.help_panel_alpha * 0.3))
        help_surface.blit(glass_surf, (0, 0))
        
        # Title
        title_text = self.ui_font.render("GAME CONTROLS", True, (255, 255, 255))
        help_surface.blit(title_text, (20, 20))
        
        # Draw help sections
        y_offset = 60
        for section_title, lines in self.help_sections.items():
            # Section title
            section_text = self.small_font.render(section_title + ":", True, self.colors['accent'])
            help_surface.blit(section_text, (20, y_offset))
            y_offset += 25
            
            # Section content
            for line in lines:
                line_text = self.small_font.render("• " + line, True, (200, 200, 200))
                help_surface.blit(line_text, (30, y_offset))
                y_offset += 20
            
            y_offset += 10
        
        # Blit to main surface
        surface.blit(help_surface, (panel_x, panel_y))
        
        # Draw border
        pygame.draw.rect(surface, (255, 255, 255, 100), 
                        (panel_x, panel_y, panel_width, panel_height), 2)
    
    def draw(self, surface, cue_ball_pos=None):
        """Draw complete premium UI"""
        # Draw animated background
        self.draw_animated_background(surface)
        
        # Draw particle effects
        self.particle_system.draw(surface)
        
        # Draw cue stick
        if cue_ball_pos is not None and self.cue_stick.visible:
            self.cue_stick.draw(surface, cue_ball_pos)
        
        # Draw power meter
        self.power_meter.draw(surface)
        
        # Draw HUD
        self.draw_hud(surface)
        
        # Draw calibration text
        if self.calibration_text:
            calib_rect = (self.screen_width//2 - 200, self.screen_height//2 - 40, 400, 80)
            self.draw_premium_panel(surface, calib_rect)
            
            calib_text = self.main_font.render(self.calibration_text, True, self.colors['accent'])
            calib_rect_center = (self.screen_width//2, self.screen_height//2)
            text_rect = calib_text.get_rect(center=calib_rect_center)
            surface.blit(calib_text, text_rect)
        
        # Draw help panel
        self.draw_help_panel(surface)
        
        # Draw premium overlays and effects
        self.draw_screen_effects(surface)
    
    def draw_screen_effects(self, surface):
        """Draw premium screen effects and overlays"""
        # Subtle vignette effect
        vignette_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        
        # Create radial gradient for vignette
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        for y in range(0, self.screen_height, 4):  # Step by 4 for performance
            for x in range(0, self.screen_width, 4):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                alpha = int((distance / max_distance) * 40)  # Subtle effect
                if alpha > 0:
                    pygame.draw.rect(vignette_surf, (0, 0, 0, alpha), (x, y, 4, 4))
        
        surface.blit(vignette_surf, (0, 0))
        
        # Animated corner decorations
        self.draw_corner_decorations(surface)
    
    def draw_corner_decorations(self, surface):
        """Draw animated corner decorations"""
        # Top-left corner
        corner_size = 60
        pulse = math.sin(self.background_time * 2) * 0.3 + 0.7
        
        # Top-left decoration
        tl_color = tuple(int(c * pulse) for c in self.colors['primary'])
        pygame.draw.arc(surface, tl_color, (10, 10, corner_size, corner_size), 
                       0, math.pi/2, 3)
        pygame.draw.arc(surface, tl_color, (15, 15, corner_size-10, corner_size-10), 
                       0, math.pi/2, 2)
        
        # Top-right decoration
        tr_x = self.screen_width - corner_size - 10
        tr_color = tuple(int(c * pulse) for c in self.colors['secondary'])
        pygame.draw.arc(surface, tr_color, (tr_x, 10, corner_size, corner_size), 
                       math.pi/2, math.pi, 3)
        pygame.draw.arc(surface, tr_color, (tr_x+5, 15, corner_size-10, corner_size-10), 
                       math.pi/2, math.pi, 2)
        
        # Bottom-left decoration
        bl_y = self.screen_height - corner_size - 10
        bl_color = tuple(int(c * pulse) for c in self.colors['accent'])
        pygame.draw.arc(surface, bl_color, (10, bl_y, corner_size, corner_size), 
                       3*math.pi/2, 2*math.pi, 3)
        pygame.draw.arc(surface, bl_color, (15, bl_y+5, corner_size-10, corner_size-10), 
                       3*math.pi/2, 2*math.pi, 2)
        
        # Bottom-right decoration
        br_x = self.screen_width - corner_size - 10
        br_y = self.screen_height - corner_size - 10
        br_color = tuple(int(c * pulse) for c in self.colors['danger'])
        pygame.draw.arc(surface, br_color, (br_x, br_y, corner_size, corner_size), 
                       math.pi, 3*math.pi/2, 3)
        pygame.draw.arc(surface, br_color, (br_x+5, br_y+5, corner_size-10, corner_size-10), 
                       math.pi, 3*math.pi/2, 2)
    
    def trigger_strike_effect(self, cue_ball_pos, power):
        """Trigger visual effects when striking the ball"""
        # Start cue stick animation
        self.cue_stick.start_strike_animation()
        
        # Emit impact particles
        particle_count = int(power * 20) + 5
        self.particle_system.emit(cue_ball_pos, count=particle_count,
                                color=(255, 255, 100), 
                                velocity_range=power * 100 + 20,
                                life_span=0.8)
        
        # Screen shake effect could be added here
        
    def get_cue_direction(self):
        """Get current cue direction"""
        return self.cue_stick.get_direction_vector()
    
    def get_power(self):
        """Get current power level"""
        return self.power_meter.power
    
    def toggle_help(self):
        """Toggle help panel visibility"""
        self.show_help = not self.show_help
    
    def update_score(self, player, points):
        """Update player score with animation"""
        if player == 1:
            self.player1_score += points
        else:
            self.player2_score += points
            
        # Could trigger score animation here
    
    def switch_player(self):
        """Switch current player"""
        self.current_player = 3 - self.current_player
        self.game_state = f"Player {self.current_player}'s Turn"
        
        # Emit celebration particles for player switch
        center_pos = (self.screen_width // 2, 100)
        player_color = self.colors['primary'] if self.current_player == 1 else self.colors['secondary']
        self.particle_system.emit(center_pos, count=15, 
                                color=player_color[:3], 
                                velocity_range=60, life_span=1.2)
    
    def reset_scores(self):
        """Reset game scores"""
        self.player1_score = 0
        self.player2_score = 0
        self.current_player = 1
        self.game_state = "Player 1's Turn"
        
        # Reset visual effects
        self.particle_system.particles.clear()
    
    def set_game_state(self, state_text, state_color=None):
        """Set game state with optional color"""
        self.game_state = state_text
        # Color support could be added for different game states
    
    def show_notification(self, message, duration=3.0, color=None):
        """Show temporary notification (could be implemented for game events)"""
        # Implementation for temporary notifications
        pass
    
    def draw_trajectory_preview(self, surface, start_pos, direction, power, max_length=200):
        """Draw trajectory preview line"""
        if power <= 0:
            return
            
        end_pos = start_pos + direction * (power * max_length)
        
        # Draw dashed line with fade effect
        dash_length = 10
        gap_length = 5
        total_length = np.linalg.norm(end_pos - start_pos)
        
        if total_length == 0:
            return
            
        unit_direction = (end_pos - start_pos) / total_length
        
        current_pos = start_pos
        distance_covered = 0
        dash_index = 0
        
        while distance_covered < total_length:
            # Alternate between dash and gap
            if dash_index % 2 == 0:  # Draw dash
                dash_end_distance = min(distance_covered + dash_length, total_length)
                dash_end_pos = start_pos + unit_direction * dash_end_distance
                
                # Fade effect based on distance
                fade_factor = 1.0 - (distance_covered / total_length) * 0.7
                alpha = int(255 * fade_factor * power)
                
                if alpha > 10:
                    color = (*self.colors['light'][:3], alpha)
                    # Draw dash with anti-aliasing effect
                    pygame.draw.line(surface, color[:3], 
                                   current_pos.astype(int), 
                                   dash_end_pos.astype(int), 2)
                
                distance_covered = dash_end_distance
                current_pos = dash_end_pos
            else:  # Skip gap
                distance_covered = min(distance_covered + gap_length, total_length)
                current_pos = start_pos + unit_direction * distance_covered
            
            dash_index += 1
    
    def create_premium_button(self, surface, rect, text, color_scheme='primary', 
                            is_hovered=False, is_pressed=False):
        """Create premium 3D button"""
        x, y, w, h = rect
        
        # Base colors
        if color_scheme == 'primary':
            base_color = self.colors['primary']
        elif color_scheme == 'secondary':
            base_color = self.colors['secondary']
        elif color_scheme == 'danger':
            base_color = self.colors['danger']
        else:
            base_color = self.colors['primary']
        
        # Adjust colors based on state
        if is_pressed:
            main_color = tuple(max(0, c - 40) for c in base_color)
            offset = 1
        elif is_hovered:
            main_color = tuple(min(255, c + 20) for c in base_color)
            offset = 2
        else:
            main_color = base_color
            offset = 3
        
        # Draw button shadow
        shadow_rect = (x + offset, y + offset, w, h)
        pygame.draw.rect(surface, (0, 0, 0, 100), shadow_rect)
        
        # Draw button base
        pygame.draw.rect(surface, main_color, (x, y, w, h))
        
        # Draw button highlight
        highlight_color = tuple(min(255, c + 60) for c in main_color)
        pygame.draw.rect(surface, highlight_color, (x, y, w, h//3))
        
        # Draw button border
        border_color = tuple(max(0, c - 60) for c in main_color)
        pygame.draw.rect(surface, border_color, (x, y, w, h), 2)
        
        # Draw button text
        text_surface = self.ui_font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(x + w//2, y + h//2))
        
        # Text shadow
        shadow_text = self.ui_font.render(text, True, (0, 0, 0))
        surface.blit(shadow_text, (text_rect.x + 1, text_rect.y + 1))
        
        # Main text
        surface.blit(text_surface, text_rect)
        
        return rect  # Return rect for click detection


# Example usage and integration class
class PremiumPoolGame:
    """Example integration of the premium UI system"""
    
    def __init__(self, screen_width=1200, screen_height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Premium Pool Game")
        
        self.ui = PremiumGameUI(screen_width, screen_height)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Demo variables
        self.cue_ball_pos = (screen_width // 2, screen_height // 2)
        self.mouse_angle = 0
        self.power_level = 0
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_h:
                    self.ui.toggle_help()
                elif event.key == pygame.K_r:
                    self.ui.reset_scores()
                elif event.key == pygame.K_SPACE:
                    # Demo strike effect
                    self.ui.trigger_strike_effect(self.cue_ball_pos, self.power_level)
    
    def update_demo(self):
        """Update demo logic"""
        # Get mouse position for cue aiming
        mouse_x, mouse_y = pygame.mouse.get_pos()
        dx = mouse_x - self.cue_ball_pos[0]
        dy = mouse_y - self.cue_ball_pos[1]
        self.mouse_angle = math.atan2(dy, dx)
        
        # Simulate power based on mouse distance
        distance = math.sqrt(dx*dx + dy*dy)
        self.power_level = min(1.0, distance / 200.0)
        
        # Update UI
        self.ui.update(
            cue_ball_pos=self.cue_ball_pos,
            target_angle=self.mouse_angle,
            power=self.power_level,
            game_state=f"Player {self.ui.current_player}'s Turn - Premium Edition"
        )
    
    def draw(self):
        """Draw everything"""
        # Clear screen with dark background
        self.screen.fill((15, 25, 35))
        
        # Draw cue ball
        pygame.draw.circle(self.screen, (255, 248, 220), 
                         [int(pos) for pos in self.cue_ball_pos], 15)
        pygame.draw.circle(self.screen, (200, 200, 200), 
                         [int(pos) for pos in self.cue_ball_pos], 15, 2)
        
        # Draw trajectory preview
        if self.power_level > 0:
            direction = self.ui.get_cue_direction()
            self.ui.draw_trajectory_preview(self.screen, 
                                          np.array(self.cue_ball_pos), 
                                          direction, self.power_level)
        
        # Draw UI
        self.ui.draw(self.screen, self.cue_ball_pos)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update_demo()
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()


# Example of how to use the premium UI
if __name__ == "__main__":
    game = PremiumPoolGame()
    game.run()