import pygame
import pygame.locals
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *

# Constants
WIDTH, HEIGHT = 1024, 768
FPS = 60
FOV = 60  # Field of view in degrees

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_BLUE = (135, 206, 235)
GREEN = (0, 100, 0)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)

# Cessna 172 specifications
CESSNA_MAX_SPEED = 140  # knots
CESSNA_CRUISE_SPEED = 122  # knots
CESSNA_STALL_SPEED = 50  # knots
CESSNA_CLIMB_RATE = 700  # feet per minute
CESSNA_DESCENT_RATE = -500  # feet per minute
CESSNA_MAX_ALTITUDE = 14000  # feet
CESSNA_ROLL_RATE = 60  # degrees per second
CESSNA_PITCH_RATE = 25  # degrees per second
CESSNA_YAW_RATE = 20  # degrees per second

# KIND Airport location (Indianapolis International Airport)
KIND_LAT = 39.7173
KIND_LON = -86.2944
KIND_ELEVATION = 797  # feet above sea level

# Conversion factors
FEET_TO_METERS = 0.3048
KNOTS_TO_MPS = 0.51444

class Aircraft:
    def __init__(self):
        # Position
        self.x = 0  # East position in meters
        self.y = KIND_ELEVATION * FEET_TO_METERS  # Altitude in meters
        self.z = 0  # North position in meters
        
        # Orientation (in degrees)
        self.pitch = 0
        self.roll = 0
        self.heading = 0
        
        # Velocity
        self.airspeed = 0  # meters per second
        self.vertical_speed = 0  # meters per second
        
        # Controls
        self.throttle = 0  # 0 to 1
        self.elevator = 0  # -1 to 1
        self.aileron = 0  # -1 to 1
        self.rudder = 0  # -1 to 1
        self.flaps = 0  # 0 to 3 (0, 10, 20, 30 degrees)
        
        # Engine
        self.rpm = 0
        self.fuel = 100  # percent
        
        # Landing gear
        self.gear_down = True

    def update(self, dt):
        # Update position based on velocity and orientation
        heading_rad = math.radians(self.heading)
        pitch_rad = math.radians(self.pitch)
        
        # Calculate velocity components
        dx = self.airspeed * math.sin(heading_rad) * math.cos(pitch_rad)
        dy = self.airspeed * math.sin(pitch_rad)
        dz = self.airspeed * math.cos(heading_rad) * math.cos(pitch_rad)
        
        # Update position
        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt
        
        # Ground collision check
        if self.y < KIND_ELEVATION * FEET_TO_METERS:
            self.y = KIND_ELEVATION * FEET_TO_METERS
            self.vertical_speed = 0
            self.pitch = max(0, self.pitch)
        
        # Apply control inputs
        self.roll += self.aileron * CESSNA_ROLL_RATE * dt
        self.pitch += self.elevator * CESSNA_PITCH_RATE * dt
        self.heading += self.rudder * CESSNA_YAW_RATE * dt
        
        # Apply banking effect on heading
        self.heading += math.sin(math.radians(self.roll)) * dt * 10
        
        # Normalize heading to 0-360
        self.heading %= 360
        
        # Limit pitch and roll
        self.pitch = max(-30, min(30, self.pitch))
        self.roll = max(-60, min(60, self.roll))
        
        # Apply throttle
        target_airspeed = self.throttle * CESSNA_MAX_SPEED * KNOTS_TO_MPS
        airspeed_diff = target_airspeed - self.airspeed
        self.airspeed += airspeed_diff * dt * 0.5
        
        # Update RPM based on throttle
        target_rpm = 500 + self.throttle * 2300  # Cessna 172 has max RPM of ~2800
        rpm_diff = target_rpm - self.rpm
        self.rpm += rpm_diff * dt * 2
        
        # Update vertical speed based on pitch and airspeed
        target_vs = 0
        if self.pitch > 0:
            # Climb rate depends on airspeed and pitch
            pitch_factor = self.pitch / 10  # normalized pitch effect
            speed_factor = min(1.0, self.airspeed / (CESSNA_CRUISE_SPEED * KNOTS_TO_MPS))
            target_vs = CESSNA_CLIMB_RATE * FEET_TO_METERS / 60 * pitch_factor * speed_factor
        elif self.pitch < 0:
            # Descent rate depends on pitch (negative)
            pitch_factor = abs(self.pitch) / 10
            target_vs = CESSNA_DESCENT_RATE * FEET_TO_METERS / 60 * pitch_factor
        
        # Smoothly adjust vertical speed
        vs_diff = target_vs - self.vertical_speed
        self.vertical_speed += vs_diff * dt * 0.5
        
        # Fuel consumption
        self.fuel -= self.throttle * dt * 0.05  # Simple fuel consumption model
        self.fuel = max(0, self.fuel)
        
        # If fuel is empty, reduce engine power
        if self.fuel <= 0:
            self.rpm = max(0, self.rpm - 500 * dt)
            
        # Natural tendency to level out
        if abs(self.aileron) < 0.1:
            self.roll *= 0.95  # Gradually reduce roll if not actively rolling
        
        # Stall behavior (very simplified)
        if self.airspeed < CESSNA_STALL_SPEED * KNOTS_TO_MPS * 0.9:
            # In a stall, the plane drops and might roll unpredictably
            self.vertical_speed -= 9.8 * dt  # Gravity effect increases
            if abs(self.roll) < 5:
                # Add some random roll during stall
                self.roll += (np.random.random() - 0.5) * 10 * dt

class FlightSimulator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Cessna 172 Flight Simulator")
        self.clock = pygame.time.Clock()
        self.aircraft = Aircraft()
        self.running = True
        self.font = pygame.font.SysFont(None, 24)
        self.paused = False
        
        # Camera
        self.eye_offset = np.array([0, 0.1, -0.3])  # Pilot's eye position relative to aircraft origin
        
        # Mouse controls
        self.mouse_sensitivity = 0.1
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos((WIDTH // 2, HEIGHT // 2))
        
        # Initialize OpenGL
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(FOV, WIDTH / HEIGHT, 0.1, 100000)
        
        # Create runway and terrain
        self.create_world()
        
        # Create cockpit elements
        self.create_cockpit()
        
        # Setup 2D overlay
        self.setup_2d_overlay()
        
        # Start the aircraft positioned on the runway
        self.aircraft.x = 500  # Start a bit down the runway
        self.aircraft.z = 0
        self.aircraft.heading = 230  # Runway heading

    def create_world(self):
        # Create the airport and surrounding terrain
        # This is a simplified representation
        
        # Create runway (approx. 5000 ft long, 100 ft wide)
        runway_length = 5000 * FEET_TO_METERS
        runway_width = 100 * FEET_TO_METERS
        
        # Create runway as a display list for efficiency
        self.runway_dl = glGenLists(1)
        glNewList(self.runway_dl, GL_COMPILE)
        
        # Runway surface
        glBegin(GL_QUADS)
        glColor3f(0.3, 0.3, 0.3)  # Dark gray for runway
        glVertex3f(-runway_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.1, 0)
        glVertex3f(runway_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.1, 0)
        glVertex3f(runway_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.1, -runway_length)
        glVertex3f(-runway_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.1, -runway_length)
        glEnd()
        
        # Runway markings (centerline)
        glBegin(GL_QUADS)
        glColor3f(1.0, 1.0, 1.0)  # White for markings
        
        # Draw dashed centerline
        dash_length = 50 * FEET_TO_METERS
        dash_width = 2 * FEET_TO_METERS
        for i in range(0, int(runway_length), int(dash_length * 2)):
            glVertex3f(-dash_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.2, -i)
            glVertex3f(dash_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.2, -i)
            glVertex3f(dash_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.2, -(i + dash_length))
            glVertex3f(-dash_width/2, KIND_ELEVATION * FEET_TO_METERS + 0.2, -(i + dash_length))
        
        # Draw threshold marks
        for i in range(-int(runway_width/2), int(runway_width/2), int(dash_width * 2)):
            glVertex3f(i, KIND_ELEVATION * FEET_TO_METERS + 0.2, -100 * FEET_TO_METERS)
            glVertex3f(i + dash_width, KIND_ELEVATION * FEET_TO_METERS + 0.2, -100 * FEET_TO_METERS)
            glVertex3f(i + dash_width, KIND_ELEVATION * FEET_TO_METERS + 0.2, -200 * FEET_TO_METERS)
            glVertex3f(i, KIND_ELEVATION * FEET_TO_METERS + 0.2, -200 * FEET_TO_METERS)
        glEnd()
        
        glEndList()
        
        # Create terrain
        terrain_size = 20000  # meters
        grid_size = 1000  # meters
        
        self.terrain_dl = glGenLists(1)
        glNewList(self.terrain_dl, GL_COMPILE)
        
        glBegin(GL_QUADS)
        for x in range(-terrain_size, terrain_size, grid_size):
            for z in range(-terrain_size, terrain_size, grid_size):
                # Randomize terrain height a bit for visual interest
                h1 = np.random.random() * 10 - 5
                h2 = np.random.random() * 10 - 5
                h3 = np.random.random() * 10 - 5
                h4 = np.random.random() * 10 - 5
                
                # Make terrain near airport flat
                dist_to_airport = math.sqrt(x*x + z*z)
                if dist_to_airport < 2000:
                    h1 = h2 = h3 = h4 = 0
                
                # Alternate colors for visual interest
                if (x + z) % (2 * grid_size) == 0:
                    glColor3f(0.0, 0.5, 0.0)  # Green
                else:
                    glColor3f(0.0, 0.6, 0.0)  # Lighter green
                
                glVertex3f(x, KIND_ELEVATION * FEET_TO_METERS + h1, z)
                glVertex3f(x + grid_size, KIND_ELEVATION * FEET_TO_METERS + h2, z)
                glVertex3f(x + grid_size, KIND_ELEVATION * FEET_TO_METERS + h3, z + grid_size)
                glVertex3f(x, KIND_ELEVATION * FEET_TO_METERS + h4, z + grid_size)
        glEnd()
        
        glEndList()

    def create_cockpit(self):
        # Create cockpit elements as a display list
        self.cockpit_dl = glGenLists(1)
        glNewList(self.cockpit_dl, GL_COMPILE)
        
        # Cockpit frame
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)  # Gray
        
        # Forward window frame
        glVertex3f(-0.5, 0.3, -0.2)
        glVertex3f(0.5, 0.3, -0.2)
        glVertex3f(0.5, 0.3, -0.2)
        glVertex3f(0.5, -0.1, -0.2)
        glVertex3f(0.5, -0.1, -0.2)
        glVertex3f(-0.5, -0.1, -0.2)
        glVertex3f(-0.5, -0.1, -0.2)
        glVertex3f(-0.5, 0.3, -0.2)
        
        # Side window frames
        glVertex3f(-0.5, 0.3, -0.2)
        glVertex3f(-0.7, 0.3, 0.0)
        glVertex3f(-0.7, 0.3, 0.0)
        glVertex3f(-0.7, -0.1, 0.0)
        glVertex3f(-0.7, -0.1, 0.0)
        glVertex3f(-0.5, -0.1, -0.2)
        
        glVertex3f(0.5, 0.3, -0.2)
        glVertex3f(0.7, 0.3, 0.0)
        glVertex3f(0.7, 0.3, 0.0)
        glVertex3f(0.7, -0.1, 0.0)
        glVertex3f(0.7, -0.1, 0.0)
        glVertex3f(0.5, -0.1, -0.2)
        glEnd()
        
        # Control yoke
        glBegin(GL_LINES)
        glColor3f(0.2, 0.2, 0.2)  # Dark gray
        
        # Yoke column
        glVertex3f(0.0, -0.2, -0.2)
        glVertex3f(0.0, -0.5, -0.4)
        
        # Yoke wheel
        for i in range(0, 360, 30):
            angle1 = math.radians(i)
            angle2 = math.radians(i + 30)
            x1 = 0.1 * math.cos(angle1)
            z1 = 0.1 * math.sin(angle1)
            x2 = 0.1 * math.cos(angle2)
            z2 = 0.1 * math.sin(angle2)
            
            glVertex3f(x1, -0.5, -0.4 + z1)
            glVertex3f(x2, -0.5, -0.4 + z2)
        glEnd()
        
        # Simple instrument panel
        glBegin(GL_QUADS)
        glColor3f(0.2, 0.2, 0.2)  # Dark gray
        glVertex3f(-0.5, -0.1, -0.2)
        glVertex3f(0.5, -0.1, -0.2)
        glVertex3f(0.5, -0.6, -0.4)
        glVertex3f(-0.5, -0.6, -0.4)
        glEnd()
        
        # Draw instrument outlines
        glBegin(GL_LINES)
        glColor3f(0.7, 0.7, 0.7)  # Light gray
        
        # Airspeed indicator
        self.draw_instrument_circle(-0.35, -0.25, -0.21, 0.1)
        
        # Attitude indicator
        self.draw_instrument_circle(-0.15, -0.25, -0.21, 0.1)
        
        # Altimeter
        self.draw_instrument_circle(0.05, -0.25, -0.21, 0.1)
        
        # Turn coordinator
        self.draw_instrument_circle(0.25, -0.25, -0.21, 0.1)
        
        # Heading indicator
        self.draw_instrument_circle(-0.25, -0.45, -0.3, 0.1)
        
        # Vertical speed indicator
        self.draw_instrument_circle(0.15, -0.45, -0.3, 0.1)
        
        # RPM gauge
        self.draw_instrument_circle(-0.35, -0.45, -0.3, 0.1)
        
        # Fuel gauge
        self.draw_instrument_circle(0.35, -0.45, -0.3, 0.1)
        glEnd()
        
        glEndList()

    def draw_instrument_circle(self, x, y, z, radius):
        # Helper to draw instrument circles
        segments = 20
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments
            glVertex3f(x + radius * math.cos(angle1), y, z + radius * math.sin(angle1))
            glVertex3f(x + radius * math.cos(angle2), y, z + radius * math.sin(angle2))

    def setup_2d_overlay(self):
        # Initialize the 2D rendering surface for HUD and instruments
        self.hud_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def update_2d_overlay(self):
        # Clear the HUD surface
        self.hud_surface.fill((0, 0, 0, 0))
        
        # Draw HUD elements
        # Artificial horizon line
        pygame.draw.line(self.hud_surface, WHITE, (WIDTH // 2 - 100, HEIGHT // 2), (WIDTH // 2 + 100, HEIGHT // 2), 2)
        
        # HUD airspeed
        speed_text = self.font.render(f"SPD: {int(self.aircraft.airspeed / KNOTS_TO_MPS)} kts", True, WHITE)
        self.hud_surface.blit(speed_text, (50, HEIGHT - 100))
        
        # HUD altitude
        alt_feet = int(self.aircraft.y / FEET_TO_METERS)
        alt_text = self.font.render(f"ALT: {alt_feet} ft", True, WHITE)
        self.hud_surface.blit(alt_text, (50, HEIGHT - 70))
        
        # HUD heading
        hdg_text = self.font.render(f"HDG: {int(self.aircraft.heading)}°", True, WHITE)
        self.hud_surface.blit(hdg_text, (50, HEIGHT - 40))
        
        # HUD vertical speed
        vs_fpm = int(self.aircraft.vertical_speed / FEET_TO_METERS * 60)
        vs_text = self.font.render(f"VS: {vs_fpm} fpm", True, WHITE)
        self.hud_surface.blit(vs_text, (WIDTH - 200, HEIGHT - 100))
        
        # HUD bank angle
        roll_text = self.font.render(f"BANK: {int(self.aircraft.roll)}°", True, WHITE)
        self.hud_surface.blit(roll_text, (WIDTH - 200, HEIGHT - 70))
        
        # HUD pitch
        pitch_text = self.font.render(f"PITCH: {int(self.aircraft.pitch)}°", True, WHITE)
        self.hud_surface.blit(pitch_text, (WIDTH - 200, HEIGHT - 40))
        
        # Engine RPM
        rpm_text = self.font.render(f"RPM: {int(self.aircraft.rpm)}", True, WHITE)
        self.hud_surface.blit(rpm_text, (WIDTH // 2 - 100, HEIGHT - 70))
        
        # Fuel gauge
        fuel_text = self.font.render(f"FUEL: {int(self.aircraft.fuel)}%", True, WHITE)
        self.hud_surface.blit(fuel_text, (WIDTH // 2 - 100, HEIGHT - 40))
        
        # Throttle setting
        throttle_text = self.font.render(f"THROT: {int(self.aircraft.throttle * 100)}%", True, WHITE)
        self.hud_surface.blit(throttle_text, (WIDTH // 2 - 100, HEIGHT - 100))
        
        # Stall warning
        if self.aircraft.airspeed < CESSNA_STALL_SPEED * KNOTS_TO_MPS * 1.1:
            stall_text = self.font.render("STALL WARNING", True, RED)
            self.hud_surface.blit(stall_text, (WIDTH // 2 - 80, 50))
            
        # Gear status
        gear_text = self.font.render("GEAR: DOWN" if self.aircraft.gear_down else "GEAR: UP", True, GREEN if self.aircraft.gear_down else RED)
        self.hud_surface.blit(gear_text, (50, HEIGHT - 130))
        
        # Flaps setting
        flaps_positions = ["0°", "10°", "20°", "30°"]
        flaps_text = self.font.render(f"FLAPS: {flaps_positions[self.aircraft.flaps]}", True, WHITE)
        self.hud_surface.blit(flaps_text, (WIDTH - 200, HEIGHT - 130))
        
        # Convert surface to texture for OpenGL
        self.hud_texture_data = pygame.image.tostring(self.hud_surface, "RGBA", True)

    def draw_2d_overlay(self):
        # Switch to 2D orthographic projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, WIDTH, HEIGHT, 0)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for 2D
        glDisable(GL_DEPTH_TEST)
        
        # Enable blend for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Render the HUD as a textured quad
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.hud_texture_data)
        
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Draw textured quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(WIDTH, 0)
        glTexCoord2f(1, 1); glVertex2f(WIDTH, HEIGHT)
        glTexCoord2f(0, 1); glVertex2f(0, HEIGHT)
        glEnd()
        
        glDeleteTextures(1, [texture])
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        
        # Restore 3D projection
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_g:
                    # Toggle landing gear
                    self.aircraft.gear_down = not self.aircraft.gear_down
                elif event.key == pygame.K_f:
                    # Cycle through flaps settings (0, 10, 20, 30 degrees)
                    self.aircraft.flaps = (self.aircraft.flaps + 1) % 4
                elif event.key == pygame.K_r:
                    # Reset aircraft
                    self.__init__()
            
            elif event.type == pygame.MOUSEMOTION:
                # Only use mouse motion if not paused
                if not self.paused:
                    # Get mouse movement
                    dx, dy = event.rel
                    
                    # Apply to aircraft control (simplified yoke control)
                    self.aircraft.elevator = -dy * self.mouse_sensitivity / 10
                    self.aircraft.aileron = dx * self.mouse_sensitivity / 10
                    
                    # Recenter mouse to avoid edge limits
                    pygame.mouse.set_pos((WIDTH // 2, HEIGHT // 2))
        
        # Continuous key inputs
        keys = pygame.key.get_pressed()
        
        # Throttle control
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self.aircraft.throttle = min(1.0, self.aircraft.throttle + 0.01)
        if keys[pygame.K_MINUS]:
            self.aircraft.throttle = max(0.0, self.aircraft.throttle - 0.01)
            
        # Rudder control
        if keys[pygame.K_z]:
            self.aircraft.rudder = -1.0
        elif keys[pygame.K_x]:
            self.aircraft.rudder = 1.0
        else:
            self.aircraft.rudder = 0.0
            
        # Allow keyboard control for elevator/aileron as alternative to mouse
        if keys[pygame.K_UP]:
            self.aircraft.elevator = 1.0
        elif keys[pygame.K_DOWN]:
            self.aircraft.elevator = -1.0
        else:
            # Only reset if mouse isn't controlling
            if abs(self.aircraft.elevator) < 0.1:
                self.aircraft.elevator = 0.0
                
        if keys[pygame.K_LEFT]:
            self.aircraft.aileron = -1.0
        elif keys[pygame.K_RIGHT]:
            self.aircraft.aileron = 1.0
        else:
            # Only reset if mouse isn't controlling
            if abs(self.aircraft.aileron) < 0.1:
                self.aircraft.aileron = 0.0

    def render(self):
        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.5, 0.7, 1.0, 1.0)  # Sky blue clear color
        
        # Set up the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Calculate camera position and orientation
        # Convert aircraft orientation from degrees to radians
        heading_rad = math.radians(self.aircraft.heading)
        pitch_rad = math.radians(self.aircraft.pitch)
        roll_rad = math.radians(self.aircraft.roll)
        
        # Create rotation matrices
        # Heading rotation (around y-axis)
        c_h, s_h = math.cos(heading_rad), math.sin(heading_rad)
        R_heading = np.array([
            [c_h, 0, -s_h],
            [0, 1, 0],
            [s_h, 0, c_h]
        ])
        
        # Pitch rotation (around x-axis)
        c_p, s_p = math.cos(pitch_rad), math.sin(pitch_rad)
        R_pitch = np.array([
            [1, 0, 0],
            [0, c_p, s_p],
            [0, -s_p, c_p]
        ])
        
        # Roll rotation (around z-axis)
        c_r, s_r = math.cos(roll_rad), math.sin(roll_rad)
        R_roll = np.array([
            [c_r, s_r, 0],
            [-s_r, c_r, 0],
            [0, 0, 1]
        ])
        
        # Combine rotations: first heading, then pitch, then roll
        R = R_roll @ R_pitch @ R_heading
        
        # Rotate eye offset
        eye_pos = R @ self.eye_offset
        
        # Add to aircraft position
        eye_x = self.aircraft.x + eye_pos[0]
        eye_y = self.aircraft.y + eye_pos[1]
        eye_z = self.aircraft.z + eye_pos[2]
        
        # Calculate look direction
        # Forward vector
        forward = np.