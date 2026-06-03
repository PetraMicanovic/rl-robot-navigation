"""
Custom Gymnasium environment for mobile robot navigation in a 2D space with static and dynamic obstacles.

Observation space:
    [dx, dy, vx, vy, l1, l2, l3, l4, l5, l6, l7, l8]
    - dx, dy: relative position of the target
    - vx, vy: current agent velocity
    - l1...l8: lidar readings 

Action space:
    [vx, vy] in [-1, 1]^2 - continuous velocity vestor

Reward:
    +10.00: reaching the target
    -10.00: collision with any obstacle
    +0.1 * delta_d: progress toward the target
    -0.01: penality per timestep
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from pathlib import Path

def load_config(config_path = "config.json"):
    """
    Load configuration parameters from a JSON file.

    Parameters:
    config_path: str
        Path to the JSON configuration file

    Returns:
    config: dict
        Dictionary containing all configuration parameters grouped by section
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, 'r') as f:
        return json.load(f)

class RobotNavEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path = "config.json", n_dynamic_obstacles = None, obstacle_speed = None, render_mode = None, use_reward_shaping = True):   
        """
        Initialize the RobotNavEnv environment. Loads all parameters from config.json.

        Parameters:
        config_path: str
            Path to the JSON configuration file
        n_dynamic_obstacles: int or None
            Number of moving obstacles
        obstacle_speed: float or None
            Speed of moving obstacles per timestep
        render_mode: str or None
            Rendering mode
        use_reward_shaping: bool
            If True, adds progress reward toward the target.
            If False, only goal, collision and step penalty rewards aare used.
        """     
        super().__init__()

        config = load_config(config_path)
        environment_config = config["environment"]
        reward_config = config["reward"]

        # Environment parameters
        self.WORLD_SIZE = environment_config["world_size"]
        self.AGENT_RADIUS = environment_config["agent_radius"]
        self.OBSTACLE_RADIUS = environment_config["obstacle_radius"]
        self.TARGET_RADIUS = environment_config["target_radius"]
        self.MAX_SPEED = environment_config["max_speed"]
        self.MAX_STEPS = environment_config["max_steps"]
        self.N_LIDAR_RAYS = environment_config["n_lidar_rays"]
        self.LIDAR_RANGE = environment_config["lidar_range"]

        self.n_dynamic_obstacles = environment_config["n_dynamic_obstacles"]
        self.obstacle_speed = environment_config["obstacle_speed"]

        # Static obstacles (list of x,y, half_width, half_height)
        self.static_obstacles = []
        for obstacle in environment_config["static_obstacles"]:
            self.static_obstacles.append(tuple(obstacle))
        
        # Reward parameters
        self.REWARD_GOAL = reward_config["goal"]
        self.REWARD_COLLISION = reward_config["collision"]
        self.REWARD_PROGRESS = reward_config["progress_scale"]
        self.REWARD_STEP = reward_config["step_penalty"]

        self.render_mode = render_mode
        self.use_reward_shaping = use_reward_shaping

        # Observation space
        observation_size = 4 + self.N_LIDAR_RAYS
        self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape = (observation_size,), dtype = np.float32)

        # Action space
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32)

        # Internal state
        self.agent_position = None
        self.agent_velocity = None
        self.target_position = None
        self.obstacle_positions = None
        self.obstacle_velocities = None
        self.step_count = None
        self.previous_distance = None

        self.renderer = None

    def reset(self, seed = None):
        """
        Reset the environment to a new random initial state. Places the agent, target and all dynamic obstacles at random positions.
        Each dynamic obstacle gets a random initial velocity direction.

        Parameters:
        seed: int or None
            Random seed for reproducibility

        Returns:
        observation: np.ndarray (12,)
            The initial observation vector for the new episode
        info: dict
            Empty dictionary
        """
        super().reset(seed = seed)

        # Place agent and target at random position
        self.agent_position = self._random_free_position()
        self.target_position = self._random_free_position(exclude=[self.agent_position], min_dist=2.0)
        self.agent_velocity = np.zeros(2, dtype=np.float32)

        excluded = [self.agent_position, self.target_position]
        self.obstacle_positions = []
        for _ in range(self.n_dynamic_obstacles):
            self.obstacle_positions.append(self._random_free_position(exclude=excluded, min_dist=1.5))
    
        # Assign random initial directions to dynamic obstacles
        random_angles = self.np_random.uniform(0, 2 * np.pi, self.n_dynamic_obstacles)
        velocity_x = np.cos(random_angles) * self.obstacle_speed
        velocity_y = np.sin(random_angles) * self.obstacle_speed
        self.obstacle_velocities= np.column_stack([velocity_x, velocity_y]).astype(np.float32)

        self.step_count = 0
        self.previous_distance  = np.linalg.norm(self.target_position - self.agent_position)

        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute one timestep in the environment.
        Applies the action to move the agent, moves dynamics obstacles,computes the reward and checks for termination conditions.

        Parameters:
        action: np.ndarray (2,)
            Velocity command [vx, vy] in [-1, 1]^2 

        Returns:
        observation: np.ndarray (12,)
            Updated observation vector after the step
        reward: float 
            Reward received for this timestep
        terminated: bool
            True if episode ended(goal reached or collision)
        truncated: bool
            True if the episode reached the maximum number of steps
        info: dict
            Dictionary with following information:
            - distance_to_target (float): current distance to target
            - step (int): current step count
            - success (bool): whether the agent reached the target
        """
        self.step_count += 1

        # Clip action to valid range and apply to agent
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.agent_velocity = action * self.MAX_SPEED
        self.agent_position = np.clip(
            self.agent_position + self.agent_velocity,
            self.AGENT_RADIUS,
            self.WORLD_SIZE - self.AGENT_RADIUS
        )

        # Move dynamic obstacles and bounce them off walls
        self.obstacle_positions += self.obstacle_velocities
        self._bounce_obstacles()

        # Reward 
        reward = self.REWARD_STEP

        # Progress reward
        current_distance = np.linalg.norm(self.target_position - self.agent_position)
        if self.use_reward_shaping:
            reward += self.REWARD_PROGRESS * (self.previous_distance - current_distance)
        self.previous_distance = current_distance

        terminated = False

        if current_distance < self.TARGET_RADIUS:
            reward += self.REWARD_GOAL
            terminated = True
        elif self._check_collision():
            reward += self.REWARD_COLLISION
            terminated = True

        truncated = self.step_count >= self.MAX_STEPS

        info = {
            "distance_to_target": float(current_distance),
            "step": self.step_count,
            "success": bool(current_distance < self.TARGET_RADIUS)
        }

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """
        Build the normalized observation vector for the current state.

        Parameters:
        None

        Returns:
        observation: np.ndarray (12,)
            Normalized observation vector:
            [dx, dy, vx, vy, l1, l2, l3, l4, l5, l6, l7, l8]
        """
        relative_target = (self.target_position - self.agent_position) / self.WORLD_SIZE
        normalized_velocity = self.agent_velocity / self.MAX_SPEED
        normalized_lidar_readings = self._cast_lidar_rays() / self.LIDAR_RANGE
        observation = np.concatenate([relative_target, normalized_velocity, normalized_lidar_readings]).astype(np.float32)
        return np.clip(observation, -1.0, 1.0)

    def _cast_lidar_rays(self):
        """
        Cast N_LIDAR_RAYS rays from the agent and return the distance to the nearest obstacle or wall in each direction.

        Parameters
        None

        Returns
        readings: np.ndarray (N_LIDAR_RAYS,)
            Array of distances to the nearest obstacle in each ray direction. Values are in [0, LIDAR_RANGE]
        """
        readings = np.full(self.N_LIDAR_RAYS, self.LIDAR_RANGE, dtype=np.float32)
        ray_angles = np.linspace(0, 2 * np.pi, self.N_LIDAR_RAYS, endpoint=False)

        for ray_index, angle in enumerate(ray_angles):
            ray_direction = np.array([np.cos(angle), np.sin(angle)])

            # Check distance to world boundary walls
            wall_distance = self._ray_vs_walls(self.agent_position, ray_direction)
            if wall_distance < readings[ray_index]:
                readings[ray_index] = wall_distance

            # Check distance to each static rectangular obstacle
            for static_obstacle in self.static_obstacles:
                rect_distance = self._ray_vs_rect(self.agent_position, ray_direction, static_obstacle)
                if rect_distance < readings[ray_index]:
                    readings[ray_index] = rect_distance

            # Check distance to each dynamic circular obstacle
            for obstacle_position in self.obstacle_positions:
                circle_distance = self._ray_vs_circle(self.agent_position, ray_direction, obstacle_position, self.OBSTACLE_RADIUS)
                if circle_distance < readings[ray_index]:
                    readings[ray_index] = circle_distance
        return readings

    def _ray_vs_walls(self, ray_origin, ray_direction):
        """
        Compute the distance from a ray origin to the nearest world boundary wall. Uses parametric ray-boundary intersection: finds the t value at
        which the ray hits each wall and returns the smallest positive t.

        Parameters
        ray_origin: np.ndarray (2,)
            Starting point of the ray (agent position)
        ray_direction : np.ndarray (2,)
            Unit direction vector of the ray

        Returns
        float
            Distance to the nearest wall. Returns LIDAR_RANGE if no intersection is found within range.
        """
        intersection_distances = []
        for dimension in range(2):
            # Skip dimensions where the ray is nearly parallel to the wall
            if abs(ray_direction[dimension]) > 1e-9:
                if ray_direction[dimension] > 0:
                    # Ray points toward the far wall (high boundary)
                    t = (self.WORLD_SIZE - self.AGENT_RADIUS - ray_origin[dimension]) / ray_direction[dimension]
                else:
                    # Ray points toward the near wall (low boundary)
                    t = (self.AGENT_RADIUS - ray_origin[dimension]) / ray_direction[dimension]
                if t > 0:
                    intersection_distances.append(t)
        if len(intersection_distances) > 0:
            return min(intersection_distances)  
        else:
            return self.LIDAR_RANGE

    def _ray_vs_rect(self, ray_origin, ray_direction, rect):
        """
        Compute the distance from a ray to an axis-aligned rectangular obstacle.
        Uses the slab method (AABB ray intersection): computes entry and exit t values for each axis and checks for overlap.

        Parameters
        ray_origin: np.ndarray (2,)
            Starting point of the ray
        ray_direction: np.ndarray of shape (2,)
            Unit direction vector of the ray
        rect: tuple of (float, float, float, float)
            Obstacle defined as (center_x, center_y, half_width, half_height)

        Returns
        float
            Distance to the rectangle along the ray. Returns LIDAR_RANGE if the ray does not intersect the rectangle.
        """
        center_x, center_y, half_width, half_height = rect

        # Axis-aligned bounding box boundaries
        bounds = [
            (center_x - half_width, center_x + half_width),   # x-axis slab
            (center_y - half_height, center_y + half_height)  # y-axis slab
        ]

        t_enter = -np.inf
        t_exit = np.inf

        for dimension, (low_bound, high_bound) in enumerate(bounds):
            ray_component = ray_direction[dimension]
            origin_component = ray_origin[dimension]

            if abs(ray_component) < 1e-9:
                # Ray is parallel to this slab — check if origin is inside
                if origin_component < low_bound or origin_component > high_bound:
                    return self.LIDAR_RANGE
            else:
                t1 = (low_bound - origin_component) / ray_component
                t2 = (high_bound - origin_component) / ray_component

                slab_enter = min(t1, t2)
                slab_exit = max(t1, t2)

                if slab_enter > t_enter:
                    t_enter = slab_enter
                if slab_exit < t_exit:
                    t_exit = slab_exit

        # Valid intersection: entry before exit and exit in front of ray
        if t_enter <= t_exit and t_exit > 0:
            if t_enter > 0:
                intersection_t = t_enter
            else:
                intersection_t = t_exit
            return float(np.clip(intersection_t, 0, self.LIDAR_RANGE))

        return self.LIDAR_RANGE

    def _ray_vs_circle(self, ray_origin, ray_direction, circle_center, circle_radius):
        """
        Compute the distance from a ray to a circular obstacle. Solves the quadratic equation derived from substituting the
        parametric ray equation into the circle equation.

        Parameters
        ray_origin: np.ndarray (2,)
            Starting point of the ray
        ray_direction: np.ndarray (2,)
            Unit direction vector of the ray
        circle_center: np.ndarray (2,)
            Center position of the circular obstacle
        circle_radius: float
            Radius of the circular obstacle

        Returns
        float
            Distance to the circle along the ray. Returns LIDAR_RANGE if the ray does not intersect the circle.
        """
        # Vector from circle center to ray origin
        origin_to_center = ray_origin - circle_center

        # Quadratic coefficients: at^2 + bt + c = 0 (a=1 since direction is unit)
        b = 2 * np.dot(origin_to_center, ray_direction)
        c = np.dot(origin_to_center, origin_to_center) - circle_radius ** 2
        discriminant = b ** 2 - 4 * c

        # No real solution means no intersection
        if discriminant < 0:
            return self.LIDAR_RANGE

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / 2.0
        t2 = (-b + sqrt_discriminant) / 2.0

        # Pick the smallest positive t (nearest intersection in front of ray)
        if t1 > 0:
            nearest_t = t1
        else:
            nearest_t = t2

        if nearest_t > 0:
            return float(np.clip(nearest_t, 0, self.LIDAR_RANGE))
        else:
            return self.LIDAR_RANGE

    def _check_collision(self):
        """
        Check whether the agent is currently colliding with anything. Tests collision against world boundary walls, all dynamic circular
        obstacles and all static rectangular obstacles.

        Parameters
        None

        Returns
        bool
            True if a collision is detected, False otherwise.
        """
        agent_margin = self.AGENT_RADIUS

        # Check collision with world boundary walls
        if self.agent_position[0] < agent_margin:
            return True
        if self.agent_position[0] > self.WORLD_SIZE - agent_margin:
            return True
        if self.agent_position[1] < agent_margin:
            return True
        if self.agent_position[1] > self.WORLD_SIZE - agent_margin:
            return True

        # Check collision with each dynamic circular obstacle
        combined_radius = self.AGENT_RADIUS + self.OBSTACLE_RADIUS
        for obstacle_position in self.obstacle_positions:
            distance = np.linalg.norm(self.agent_position - obstacle_position)
            if distance < combined_radius:
                return True

        # Check collision with each static rectangular obstacle
        for center_x, center_y, half_width, half_height in self.static_obstacles:
            # Find the closest point on the rectangle to the agent
            closest_x = np.clip(self.agent_position[0], center_x - half_width, center_x + half_width)
            closest_y = np.clip(self.agent_position[1], center_y - half_height, center_y + half_height)
            closest_point = np.array([closest_x, closest_y])
            distance = np.linalg.norm(self.agent_position - closest_point)
            if distance < self.AGENT_RADIUS:
                return True

        return False

    def _bounce_obstacles(self):
        """
        Reflect dynamic obstacle velocities when they hit world boundaries.
        For each obstacle, if it has crossed a wall boundary, its position is clamped back to the boundary and the velocity in that dimension
        is reversed (elastic bounce).

        Parameters
        None
        """
        low_boundary = self.OBSTACLE_RADIUS
        high_boundary = self.WORLD_SIZE - self.OBSTACLE_RADIUS

        for obstacle_index in range(self.n_dynamic_obstacles):
            for dimension in range(2):
                if self.obstacle_positions[obstacle_index, dimension] < low_boundary:
                    self.obstacle_positions[obstacle_index, dimension] = low_boundary
                    self.obstacle_velocities[obstacle_index, dimension] *= -1
                elif self.obstacle_positions[obstacle_index, dimension] > high_boundary:
                    self.obstacle_positions[obstacle_index, dimension] = high_boundary
                    self.obstacle_velocities[obstacle_index, dimension] *= -1

    def _random_free_position(self, exclude=None, min_dist=1.0):
        """
        Sample a random position that is free of walls, static obstacles and sufficiently far from a list of excluded positions.
        Attempts up to 200 random samples before falling back to the arena center if no valid position is found.

        Parameters
        exclude: list of np.ndarray or None
            List of positions that the new position must be at least min_dist away from
        min_dist: float
            Minimum allowed distance from each excluded position

        Returns
        position: np.ndarray (2,)
            A valid free position in the arena
        """
        wall_margin = max(self.AGENT_RADIUS * 2, 0.5)

        for _ in range(200):
            candidate = self.np_random.uniform(wall_margin, self.WORLD_SIZE - wall_margin, size=2).astype(np.float32)

            # Check minimum distance from all excluded positions
            too_close_to_excluded = False
            if exclude is not None:
                for excluded_position in exclude:
                    distance = np.linalg.norm(candidate - excluded_position)
                    if distance < min_dist:
                        too_close_to_excluded = True
                        break
            if too_close_to_excluded:
                continue

            # Check that candidate is not inside any static obstacle
            inside_obstacle = False
            for center_x, center_y, half_width, half_height in self.static_obstacles:
                x_overlap = abs(candidate[0] - center_x) < half_width + wall_margin
                y_overlap = abs(candidate[1] - center_y) < half_height + wall_margin
                if x_overlap and y_overlap:
                    inside_obstacle = True
                    break
            if inside_obstacle:
                continue

            return candidate

        # Fallback: return arena center if no valid position found after 200 attempts
        return np.array([self.WORLD_SIZE / 2.0, self.WORLD_SIZE / 2.0], dtype=np.float32)

    def render(self):
        """
        Render the current environment state using pygame.
        Draws the arena, static obstacles (grey rectangles), the target(green circle), dynamic obstacles (orange circles) and the agent (blue circle).

        Parameters
        None

        Returns
        None
            When render_mode is 'human' (draws to screen)
        np.ndarray of shape (HEIGHT, WIDTH, 3)
            RGB image array when render_mode is 'rgb_array'
        """
        if self.render_mode is None:
            return

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering. "
                             "Install it with: pip install pygame"
            )

        PIXELS_PER_UNIT = 60
        WINDOW_SIZE = int(self.WORLD_SIZE * PIXELS_PER_UNIT)

        # Initialize pygame and create display surface on first call
        if self.renderer is None:
            pygame.init()
            if self.render_mode == "human":
                self.renderer = pygame.display.set_mode(
                    (WINDOW_SIZE, WINDOW_SIZE)
                )
                pygame.display.set_caption("RL Robot Navigation")
            else:
                self.renderer = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))

        surface = self.renderer
        surface.fill((240, 240, 240))

        def world_to_pixels(world_position):
            """Convert world coordinates to pixel coordinates."""
            pixel_x = int(world_position[0] * PIXELS_PER_UNIT)
            pixel_y = int((self.WORLD_SIZE - world_position[1]) * PIXELS_PER_UNIT)
            return (pixel_x, pixel_y)

        # Draw static rectangular obstacles (grey)
        for center_x, center_y, half_width, half_height in self.static_obstacles:
            rect_pixel_x = int((center_x - half_width) * PIXELS_PER_UNIT)
            rect_pixel_y = int((self.WORLD_SIZE - center_y - half_height) * PIXELS_PER_UNIT)
            rect_width = int(2 * half_width  * PIXELS_PER_UNIT)
            rect_height = int(2 * half_height * PIXELS_PER_UNIT)
            rect = pygame.Rect(rect_pixel_x, rect_pixel_y, rect_width, rect_height)
            pygame.draw.rect(surface, (100, 100, 100), rect)

        # Draw target (green circle)
        pygame.draw.circle(surface, (80, 180, 80), world_to_pixels(self.target_position), int(self.TARGET_RADIUS * PIXELS_PER_UNIT))

        # Draw dynamic obstacles (orange circles)
        for obstacle_position in self.obstacle_positions:
            pygame.draw.circle(surface,(220, 100, 50), world_to_pixels(obstacle_position), int(self.OBSTACLE_RADIUS * PIXELS_PER_UNIT))

        # Draw agent (blue circle)
        pygame.draw.circle(surface, (50, 120, 210), world_to_pixels(self.agent_position), int(self.AGENT_RADIUS * PIXELS_PER_UNIT))

        if self.render_mode == "human":
            pygame.display.flip()
            pygame.time.Clock().tick(self.metadata["render_fps"])
        else:
            raw_pixels = pygame.surfarray.array3d(surface)
            return np.transpose(raw_pixels, axes=(1, 0, 2))

    def close(self):
        """
        Clean up resources when the environment is no longer needed.
        Shuts down the pygame display if it was initialized.

        Parameters
        None
        """
        if self.renderer is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self.renderer = None
