import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import math
import time
import random
from typing import List, Tuple, Set, Optional, Dict


class Node:
    """Represents a node in the grid"""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.g = float('inf')  # Cost from start
        self.h = 0  # Heuristic cost to goal
        self.f = float('inf')  # Total cost
        self.parent: Optional['Node'] = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class PathfindingAgent:
    """Main pathfinding agent with A* and GBFS support"""

    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles: Set[Tuple[int, int]] = set()
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None

        # Metrics
        self.nodes_visited = 0
        self.path_cost = 0
        self.execution_time = 0

        # Visualization state
        self.frontier: Set[Tuple[int, int]] = set()
        self.visited: Set[Tuple[int, int]] = set()
        self.path: List[Tuple[int, int]] = []

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance heuristic"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (4-directional)"""
        x, y = pos
        neighbors = []

        # 4-directional movement (up, down, left, right)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.grid_width and
                    0 <= ny < self.grid_height and
                    (nx, ny) not in self.obstacles):
                neighbors.append((nx, ny))

        return neighbors

    def a_star(self, heuristic_func, start: Tuple[int, int], goal: Tuple[int, int], step_callback=None) -> bool:
        """
        A* search algorithm
        f(n) = g(n) + h(n)
        step_callback: Optional callback function called after each step for visualization
        """
        self.nodes_visited = 0
        self.frontier.clear()
        self.visited.clear()
        self.path.clear()

        start_time = time.time()

        # Initialize nodes
        nodes: Dict[Tuple[int, int], Node] = {}
        start_node = Node(start[0], start[1])
        start_node.g = 0
        start_node.h = heuristic_func(start, goal)
        start_node.f = start_node.g + start_node.h

        nodes[start] = start_node

        # Priority queue: (f_score, counter, node)
        open_set = [(start_node.f, 0, start)]
        counter = 1
        open_set_hash = {start}

        while open_set:
            # Get node with lowest f score
            current_f, _, current_pos = heapq.heappop(open_set)
            open_set_hash.discard(current_pos)

            # Update visualization
            self.frontier = {pos for _, _, pos in open_set}

            # Call visualization callback to show current step
            if step_callback:
                step_callback("exploring", current_pos)

            # Check if goal reached
            if current_pos == goal:
                # Reconstruct path
                self.path = self._reconstruct_path(nodes, current_pos)
                self.path_cost = len(self.path) - 1 if self.path else 0
                self.execution_time = (time.time() - start_time) * 1000
                return True

            # Mark as visited
            self.visited.add(current_pos)
            self.nodes_visited += 1

            current_node = nodes[current_pos]

            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_pos):
                if neighbor_pos in self.visited:
                    continue

                # Calculate tentative g score
                tentative_g = current_node.g + 1

                # Create or update neighbor node
                if neighbor_pos not in nodes:
                    neighbor_node = Node(neighbor_pos[0], neighbor_pos[1])
                    nodes[neighbor_pos] = neighbor_node
                else:
                    neighbor_node = nodes[neighbor_pos]

                # Check if this path is better
                if tentative_g < neighbor_node.g:
                    neighbor_node.parent = current_node
                    neighbor_node.g = tentative_g
                    neighbor_node.h = heuristic_func(neighbor_pos, goal)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    if neighbor_pos not in open_set_hash:
                        heapq.heappush(open_set, (neighbor_node.f, counter, neighbor_pos))
                        counter += 1
                        open_set_hash.add(neighbor_pos)

                        # Visualize adding to frontier
                        self.frontier = {pos for _, _, pos in open_set}
                        if step_callback:
                            step_callback("frontier_add", neighbor_pos)

        # No path found
        self.execution_time = (time.time() - start_time) * 1000
        return False

    def greedy_best_first(self, heuristic_func, start: Tuple[int, int], goal: Tuple[int, int],
                          step_callback=None) -> bool:
        """
        Greedy Best-First Search algorithm
        f(n) = h(n)
        step_callback: Optional callback function called after each step for visualization
        """
        self.nodes_visited = 0
        self.frontier.clear()
        self.visited.clear()
        self.path.clear()

        start_time = time.time()

        # Initialize nodes
        nodes: Dict[Tuple[int, int], Node] = {}
        start_node = Node(start[0], start[1])
        start_node.h = heuristic_func(start, goal)
        start_node.f = start_node.h

        nodes[start] = start_node

        # Priority queue: (h_score, counter, node)
        open_set = [(start_node.h, 0, start)]
        counter = 1
        open_set_hash = {start}

        while open_set:
            # Get node with lowest h score
            current_h, _, current_pos = heapq.heappop(open_set)
            open_set_hash.discard(current_pos)

            # Update visualization
            self.frontier = {pos for _, _, pos in open_set}

            # Call visualization callback to show current step
            if step_callback:
                step_callback("exploring", current_pos)

            # Check if goal reached
            if current_pos == goal:
                # Reconstruct path
                self.path = self._reconstruct_path(nodes, current_pos)
                self.path_cost = len(self.path) - 1 if self.path else 0
                self.execution_time = (time.time() - start_time) * 1000
                return True

            # Mark as visited
            self.visited.add(current_pos)
            self.nodes_visited += 1

            current_node = nodes[current_pos]

            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current_pos):
                if neighbor_pos in self.visited or neighbor_pos in open_set_hash:
                    continue

                # Create neighbor node
                neighbor_node = Node(neighbor_pos[0], neighbor_pos[1])
                neighbor_node.parent = current_node
                neighbor_node.h = heuristic_func(neighbor_pos, goal)
                neighbor_node.f = neighbor_node.h

                nodes[neighbor_pos] = neighbor_node

                heapq.heappush(open_set, (neighbor_node.h, counter, neighbor_pos))
                counter += 1
                open_set_hash.add(neighbor_pos)

                # Visualize adding to frontier
                self.frontier = {pos for _, _, pos in open_set}
                if step_callback:
                    step_callback("frontier_add", neighbor_pos)

        # No path found
        self.execution_time = (time.time() - start_time) * 1000
        return False

    def _reconstruct_path(self, nodes: Dict[Tuple[int, int], Node], current_pos: Tuple[int, int]) -> List[
        Tuple[int, int]]:
        """Reconstruct path from goal to start"""
        path = []
        current_node = nodes.get(current_pos)

        while current_node is not None:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent

        path.reverse()
        return path

    def generate_random_obstacles(self, density: float):
        """Generate random obstacles with given density"""
        self.obstacles.clear()

        total_cells = self.grid_width * self.grid_height
        num_obstacles = int(total_cells * density / 100)

        placed = 0
        attempts = 0
        max_attempts = num_obstacles * 10

        while placed < num_obstacles and attempts < max_attempts:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            pos = (x, y)

            if pos not in self.obstacles and pos != self.start and pos != self.goal:
                self.obstacles.add(pos)
                placed += 1

            attempts += 1


class PathfindingGUI:
    """GUI for the pathfinding agent"""

    # Colors
    COLOR_EMPTY = "#FFFFFF"
    COLOR_OBSTACLE = "#000000"
    COLOR_START = "#4800FF"
    COLOR_GOAL = "#FF0000"
    COLOR_FRONTIER = "#FFFF00"
    COLOR_VISITED = "#10B4F5"
    COLOR_PATH = "#00FF7F"
    COLOR_AGENT = "#FF00FF"
    COLOR_EXPLORING = "#FFA500"  # Orange for currently exploring node

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dynamic Pathfinding Agent")

        # Default settings
        self.grid_width = 30
        self.grid_height = 20
        self.cell_size = 25

        self.agent: Optional[PathfindingAgent] = None
        self.canvas: Optional[tk.Canvas] = None

        # State
        self.mode = "obstacle"  # obstacle, start, goal
        self.algorithm = "A*"
        self.heuristic = "Manhattan"
        self.dynamic_mode = False
        self.is_running = False
        self.agent_position: Optional[Tuple[int, int]] = None
        self.spawn_probability = 0.02
        self.animation_speed = 50  # milliseconds between steps
        self.current_exploring: Optional[Tuple[int, int]] = None

        # Dynamic mode tracking
        self.spawn_attempts = 0
        self.total_obstacles_spawned = 0
        self.total_replans = 0
        self.remaining_path_cache: List[Tuple[int, int]] = []

        self._setup_ui()
        self._create_new_grid()

    def _setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)

        # Grid size controls
        size_frame = ttk.Frame(control_frame)
        size_frame.grid(row=0, column=0, columnspan=3, pady=5)

        ttk.Label(size_frame, text="Width:").grid(row=0, column=0, padx=5)
        self.width_var = tk.StringVar(value="30")
        ttk.Entry(size_frame, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(size_frame, text="Height:").grid(row=0, column=2, padx=5)
        self.height_var = tk.StringVar(value="20")
        ttk.Entry(size_frame, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Button(size_frame, text="New Grid", command=self._create_new_grid).grid(row=0, column=4, padx=5)

        # Algorithm selection
        algo_frame = ttk.Frame(control_frame)
        algo_frame.grid(row=1, column=0, columnspan=3, pady=5)

        ttk.Label(algo_frame, text="Algorithm:").grid(row=0, column=0, padx=5)
        self.algo_var = tk.StringVar(value="A*")
        ttk.Radiobutton(algo_frame, text="A*", variable=self.algo_var, value="A*").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(algo_frame, text="GBFS", variable=self.algo_var, value="GBFS").grid(row=0, column=2, padx=5)

        # Heuristic selection
        heur_frame = ttk.Frame(control_frame)
        heur_frame.grid(row=2, column=0, columnspan=3, pady=5)

        ttk.Label(heur_frame, text="Heuristic:").grid(row=0, column=0, padx=5)
        self.heur_var = tk.StringVar(value="Manhattan")
        ttk.Radiobutton(heur_frame, text="Manhattan", variable=self.heur_var, value="Manhattan").grid(row=0, column=1,
                                                                                                      padx=5)
        ttk.Radiobutton(heur_frame, text="Euclidean", variable=self.heur_var, value="Euclidean").grid(row=0, column=2,
                                                                                                      padx=5)

        # Drawing mode
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=3, column=0, columnspan=3, pady=5)

        ttk.Label(mode_frame, text="Draw Mode:").grid(row=0, column=0, padx=5)
        self.mode_var = tk.StringVar(value="obstacle")
        ttk.Radiobutton(mode_frame, text="Obstacle", variable=self.mode_var, value="obstacle").grid(row=0, column=1,
                                                                                                    padx=5)
        ttk.Radiobutton(mode_frame, text="Start", variable=self.mode_var, value="start").grid(row=0, column=2, padx=5)
        ttk.Radiobutton(mode_frame, text="Goal", variable=self.mode_var, value="goal").grid(row=0, column=3, padx=5)

        # Random generation
        random_frame = ttk.Frame(control_frame)
        random_frame.grid(row=4, column=0, columnspan=3, pady=5)

        ttk.Label(random_frame, text="Obstacle Density (%):").grid(row=0, column=0, padx=5)
        self.density_var = tk.StringVar(value="30")
        ttk.Entry(random_frame, textvariable=self.density_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Button(random_frame, text="Generate Random", command=self._generate_random_map).grid(row=0, column=2,
                                                                                                 padx=5)

        # Dynamic mode
        dynamic_frame = ttk.Frame(control_frame)
        dynamic_frame.grid(row=5, column=0, columnspan=3, pady=5)

        self.dynamic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dynamic_frame, text="Dynamic Mode (Spawn Obstacles)", variable=self.dynamic_var).grid(row=0,
                                                                                                              column=0,
                                                                                                              padx=5)

        # Animation speed
        speed_frame = ttk.Frame(control_frame)
        speed_frame.grid(row=6, column=0, columnspan=3, pady=5)

        ttk.Label(speed_frame, text="Animation Speed (ms):").grid(row=0, column=0, padx=5)
        self.speed_var = tk.StringVar(value="50")
        speed_entry = ttk.Entry(speed_frame, textvariable=self.speed_var, width=8)
        speed_entry.grid(row=0, column=1, padx=5)
        ttk.Label(speed_frame, text="(lower = faster)").grid(row=0, column=2, padx=5)

        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Find Path", command=self._find_path).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear Path", command=self._clear_path).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self._clear_all).grid(row=0, column=2, padx=5)

        # Metrics panel
        metrics_frame = ttk.LabelFrame(main_frame, text="Metrics", padding="10")
        metrics_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)

        self.nodes_visited_label = ttk.Label(metrics_frame, text="Nodes Visited: 0")
        self.nodes_visited_label.grid(row=0, column=0, sticky=tk.W, pady=2)

        self.path_cost_label = ttk.Label(metrics_frame, text="Path Cost: 0")
        self.path_cost_label.grid(row=1, column=0, sticky=tk.W, pady=2)

        self.execution_time_label = ttk.Label(metrics_frame, text="Execution Time: 0 ms")
        self.execution_time_label.grid(row=2, column=0, sticky=tk.W, pady=2)

        self.status_label = ttk.Label(metrics_frame, text="Status: Ready", foreground="green")
        self.status_label.grid(row=3, column=0, sticky=tk.W, pady=2)

        # Legend
        legend_frame = ttk.LabelFrame(main_frame, text="Legend", padding="10")
        legend_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)

        legend_items = [
            ("Start", self.COLOR_START),
            ("Goal", self.COLOR_GOAL),
            ("Obstacle", self.COLOR_OBSTACLE),
            ("Exploring", self.COLOR_EXPLORING),
            ("Frontier", self.COLOR_FRONTIER),
            ("Visited", self.COLOR_VISITED),
            ("Path", self.COLOR_PATH),
            ("Agent", self.COLOR_AGENT)
        ]

        for i, (label, color) in enumerate(legend_items):
            canvas = tk.Canvas(legend_frame, width=20, height=20, bg=color, highlightthickness=1)
            canvas.grid(row=i // 4, column=(i % 4) * 2, padx=5, pady=2)
            ttk.Label(legend_frame, text=label).grid(row=i // 4, column=(i % 4) * 2 + 1, sticky=tk.W, padx=5, pady=2)

        # Canvas frame
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Scrollbars
        self.canvas_scroll_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.canvas_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.canvas_scroll_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.canvas_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def _create_new_grid(self):
        """Create a new grid with specified dimensions"""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())

            if width < 5 or width > 100 or height < 5 or height > 100:
                messagebox.showerror("Error", "Grid dimensions must be between 5 and 100")
                return

            self.grid_width = width
            self.grid_height = height

            # Create new agent
            self.agent = PathfindingAgent(self.grid_width, self.grid_height)

            # Set default start and goal
            self.agent.start = (0, 0)
            self.agent.goal = (self.grid_width - 1, self.grid_height - 1)

            # Create canvas
            if self.canvas:
                self.canvas.destroy()

            canvas_width = self.grid_width * self.cell_size
            canvas_height = self.grid_height * self.cell_size

            self.canvas = tk.Canvas(
                self.root.grid_slaves(row=0, column=0)[0].grid_slaves(row=0, column=1)[0],
                width=min(canvas_width, 800),
                height=min(canvas_height, 600),
                bg=self.COLOR_EMPTY,
                scrollregion=(0, 0, canvas_width, canvas_height)
            )
            self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Configure scrollbars
            self.canvas.config(xscrollcommand=self.canvas_scroll_x.set, yscrollcommand=self.canvas_scroll_y.set)
            self.canvas_scroll_x.config(command=self.canvas.xview)
            self.canvas_scroll_y.config(command=self.canvas.yview)

            # Bind mouse events
            self.canvas.bind("<Button-1>", self._on_canvas_click)
            self.canvas.bind("<B1-Motion>", self._on_canvas_drag)

            # Draw grid
            self._draw_grid()

            self.status_label.config(text="Status: Grid created", foreground="green")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for width and height")

    def _draw_grid(self):
        """Draw the grid on canvas"""
        if not self.canvas or not self.agent:
            return

        self.canvas.delete("all")

        # Draw grid lines
        for i in range(self.grid_width + 1):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.grid_height * self.cell_size, fill="#CCCCCC")

        for i in range(self.grid_height + 1):
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.grid_width * self.cell_size, y, fill="#CCCCCC")

        # Draw visited cells
        for x, y in self.agent.visited:
            if (x, y) != self.current_exploring:
                self._draw_cell(x, y, self.COLOR_VISITED)

        # Draw currently exploring cell (highlighted)
        if self.current_exploring:
            self._draw_cell(self.current_exploring[0], self.current_exploring[1], self.COLOR_EXPLORING)

        # Draw frontier cells
        for x, y in self.agent.frontier:
            self._draw_cell(x, y, self.COLOR_FRONTIER)

        # Draw path (skip agent position so it's visible)
        for x, y in self.agent.path:
            if (x, y) != self.agent.start and (x, y) != self.agent.goal and (x, y) != self.agent_position:
                self._draw_cell(x, y, self.COLOR_PATH)

        # Draw obstacles
        for x, y in self.agent.obstacles:
            self._draw_cell(x, y, self.COLOR_OBSTACLE)

        # Draw start and goal
        if self.agent.start:
            self._draw_cell(self.agent.start[0], self.agent.start[1], self.COLOR_START)

        if self.agent.goal:
            self._draw_cell(self.agent.goal[0], self.agent.goal[1], self.COLOR_GOAL)

        # Draw agent position LAST so it's always visible (if in motion)
        if self.agent_position and self.agent_position != self.agent.goal:
            self._draw_cell(self.agent_position[0], self.agent_position[1], self.COLOR_AGENT)

    def _draw_cell(self, x: int, y: int, color: str):
        """Draw a single cell"""
        x1 = x * self.cell_size + 1
        y1 = y * self.cell_size + 1
        x2 = (x + 1) * self.cell_size - 1
        y2 = (y + 1) * self.cell_size - 1
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def _on_canvas_click(self, event):
        """Handle canvas click event"""
        if not self.agent or self.is_running:
            return

        x = self.canvas.canvasx(event.x) // self.cell_size
        y = self.canvas.canvasy(event.y) // self.cell_size

        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            self._handle_cell_edit(int(x), int(y))

    def _on_canvas_drag(self, event):
        """Handle canvas drag event"""
        if not self.agent or self.is_running:
            return

        x = self.canvas.canvasx(event.x) // self.cell_size
        y = self.canvas.canvasy(event.y) // self.cell_size

        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            self._handle_cell_edit(int(x), int(y))

    def _handle_cell_edit(self, x: int, y: int):
        """Handle editing a cell"""
        pos = (x, y)
        mode = self.mode_var.get()

        if mode == "start":
            self.agent.start = pos
            self.agent.obstacles.discard(pos)
        elif mode == "goal":
            self.agent.goal = pos
            self.agent.obstacles.discard(pos)
        elif mode == "obstacle":
            if pos != self.agent.start and pos != self.agent.goal:
                if pos in self.agent.obstacles:
                    self.agent.obstacles.remove(pos)
                else:
                    self.agent.obstacles.add(pos)

        self._draw_grid()

    def _generate_random_map(self):
        """Generate random obstacles"""
        if not self.agent:
            return

        try:
            density = float(self.density_var.get())
            if density < 0 or density > 80:
                messagebox.showerror("Error", "Density must be between 0 and 80")
                return

            self.agent.generate_random_obstacles(density)
            self._clear_path()
            self._draw_grid()
            self.status_label.config(text="Status: Random map generated", foreground="green")

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for density")

    def _find_path(self):
        """Find path using selected algorithm"""
        if not self.agent or not self.agent.start or not self.agent.goal:
            messagebox.showerror("Error", "Please set start and goal positions")
            return

        if self.is_running:
            return

        # Check if start or goal is on obstacle
        if self.agent.start in self.agent.obstacles or self.agent.goal in self.agent.obstacles:
            messagebox.showerror("Error", "Start or goal is on an obstacle")
            return

        # Reset dynamic mode counters
        self.spawn_attempts = 0
        self.total_obstacles_spawned = 0
        self.total_replans = 0
        self.remaining_path_cache = []

        self.is_running = True
        self.status_label.config(text="Status: Searching...", foreground="orange")
        self.root.update()

        # Get algorithm and heuristic
        algorithm = self.algo_var.get()
        heuristic = self.heur_var.get()

        # Select heuristic function
        if heuristic == "Manhattan":
            heuristic_func = self.agent.manhattan_distance
        else:
            heuristic_func = self.agent.euclidean_distance

        # Run algorithm step by step with visualization
        if algorithm == "A*":
            success = self._run_search_with_visualization(self.agent.a_star, heuristic_func)
        else:  # GBFS
            success = self._run_search_with_visualization(self.agent.greedy_best_first, heuristic_func)

        # Update metrics
        self._update_metrics()

        if success:
            self.status_label.config(text="Status: Path found ✓", foreground="green")

            # Start dynamic mode if enabled
            if self.dynamic_var.get():
                self.agent_position = self.agent.start
                self.is_running = False  # Allow movement to proceed
                self.root.after(100, self._simulate_movement)  # Start movement after short delay
            else:
                self.is_running = False
        else:
            self.status_label.config(text="Status: No path found ✗", foreground="red")
            self.is_running = False
            messagebox.showwarning("No Path", "No path found to the goal")

    def _run_search_with_visualization(self, search_func, heuristic_func) -> bool:
        """Run search algorithm with step-by-step visualization"""
        # Get animation speed
        try:
            self.animation_speed = max(1, int(self.speed_var.get()))
        except ValueError:
            self.animation_speed = 50

        # Store step counter for visualization throttling
        self.step_counter = 0
        self.visualization_interval = max(1, self.animation_speed // 5)  # Update every N steps

        # Create visualization callback
        def step_callback(action, position):
            self.step_counter += 1

            # Update visualization at intervals to avoid blocking
            if self.step_counter % self.visualization_interval == 0:
                if action == "exploring":
                    self.current_exploring = position
                    self.status_label.config(
                        text=f"Status: Exploring ({position[0]}, {position[1]}) | Visited: {self.agent.nodes_visited}",
                        foreground="orange"
                    )
                elif action == "frontier_add":
                    self.status_label.config(
                        text=f"Status: Added to frontier ({position[0]}, {position[1]})",
                        foreground="blue"
                    )

                # Update visualization WITHOUT blocking
                self._draw_grid()
                self.root.update()

        # Run the search with callback (non-blocking)
        success = search_func(heuristic_func, self.agent.start, self.agent.goal, step_callback)

        # Clear exploring highlight
        self.current_exploring = None

        # Draw final result
        self._draw_grid()
        self.root.update()

        return success

    def _simulate_movement(self):
        """Simulate agent movement with dynamic obstacle spawning and efficient re-planning

        Dynamic Mode Features:
        - Obstacles spawn with controlled probability during movement
        - Only re-plans if spawned obstacle blocks the current path
        - Avoids unnecessary re-planning for out-of-path obstacles
        - Efficiently detects path blockage
        """
        if not self.agent.path or self.agent_position == self.agent.goal:
            self.agent_position = None
            self._draw_grid()
            self.status_label.config(text=f"Status: Goal reached! 🎉 (Re-plans: {self.total_replans})", foreground="#27AE60")
            return

        # Find current position index in path
        try:
            current_index = self.agent.path.index(self.agent_position)
        except ValueError:
            self.agent_position = None
            self._draw_grid()
            return

        # Spawning Logic: Generate new obstacles with probability
        if random.random() < self.spawn_probability:
            self._spawn_obstacle()

        # Move to next position
        if current_index + 1 < len(self.agent.path):
            next_pos = self.agent.path[current_index + 1]

            # COLLISION DETECTION: Check if next position is blocked
            if next_pos in self.agent.obstacles:
                # Only re-plan if obstacle is on the remaining path
                # This is the optimization - we don't re-plan for obstacles outside our path
                self._handle_path_blockage(current_index)
                return

            # Move agent
            self.agent_position = next_pos

            # Cache remaining path for efficient check
            self.remaining_path_cache = self.agent.path[current_index + 1:]

            # Update status with current movement
            self.status_label.config(
                text=f"Status: Agent at ({next_pos[0]}, {next_pos[1]}) | Obstacles spawned: {self.total_obstacles_spawned} | Re-plans: {self.total_replans}",
                foreground="#9B59B6"
            )

            # Draw and schedule next move
            self._draw_grid()
            self.root.update()

            if self.agent_position != self.agent.goal:
                self.root.after(200, self._simulate_movement)
            else:
                self.status_label.config(text=f"Status: Goal reached! 🎉 (Re-plans: {self.total_replans})", foreground="#27AE60")
                self.agent_position = None
                self._draw_grid()

    def _spawn_obstacle(self):
        """Spawn a new obstacle with controlled logic

        Ensures spawned obstacles don't block start, goal, or agent position
        Tracks total spawned obstacles for metrics
        """
        # Try to spawn obstacle up to 20 times
        for attempt in range(20):
            obstacle_x = random.randint(0, self.grid_width - 1)
            obstacle_y = random.randint(0, self.grid_height - 1)
            obstacle_pos = (obstacle_x, obstacle_y)

            # Validate obstacle position
            if (obstacle_pos != self.agent.start and
                obstacle_pos != self.agent.goal and
                obstacle_pos != self.agent_position and
                obstacle_pos not in self.agent.obstacles):

                # Spawn the obstacle
                self.agent.obstacles.add(obstacle_pos)
                self.total_obstacles_spawned += 1
                self.spawn_attempts += 1

                # Visual feedback of spawned obstacle
                self.status_label.config(
                    text=f"Status: Obstacle spawned at ({obstacle_pos[0]}, {obstacle_pos[1]})",
                    foreground="#F39C12"
                )
                break

    def _handle_path_blockage(self, current_index: int):
        """Handle obstacle blocking current path - Optimized re-planning

        Only re-plans if the obstacle blocks the current remaining path
        Avoids unnecessary re-planning for obstacles not on path
        """
        self.status_label.config(
            text="Status: ⚠️  Path blocked! Detecting collision...",
            foreground="#E74C3C"
        )
        self.root.update()

        # Check if any obstacle is on the remaining path
        remaining_path = self.agent.path[current_index:]
        blocked_nodes = [pos for pos in remaining_path if pos in self.agent.obstacles]

        if not blocked_nodes:
            # No obstacles on remaining path - continue moving
            return

        # OPTIMIZATION: Only re-plan if blockage is close (next 3 nodes)
        # This prevents unnecessary re-planning for distant blockages
        if len(blocked_nodes) > 0 and current_index + 3 < len(remaining_path):
            # Distant blockage - continue with current path
            next_pos = self.agent.path[current_index + 1]
            if next_pos not in self.agent.obstacles:
                return

        # Re-planning Mechanism: Calculate new path from current position
        self.status_label.config(
            text="Status: 🔄 Re-planning new path...",
            foreground="#E67E22"
        )
        self.root.update()

        old_start = self.agent.start
        self.agent.start = self.agent_position

        # Get algorithm and heuristic
        algorithm = self.algo_var.get()
        heuristic = self.heur_var.get()

        if heuristic == "Manhattan":
            heuristic_func = self.agent.manhattan_distance
        else:
            heuristic_func = self.agent.euclidean_distance

        # Re-run search from current position
        if algorithm == "A*":
            success = self.agent.a_star(heuristic_func, self.agent.start, self.agent.goal)
        else:
            success = self.agent.greedy_best_first(heuristic_func, self.agent.start, self.agent.goal)

        self.agent.start = old_start
        self.total_replans += 1

        if not success:
            self.status_label.config(
                text="Status: ✗ Path blocked! No escape route found",
                foreground="#E74C3C"
            )
            self.agent_position = None
            self._draw_grid()
            messagebox.showwarning("Path Blocked",
                f"Cannot reach goal from position ({self.agent_position[0]}, {self.agent_position[1]})\n"
                f"Total obstacles spawned: {self.total_obstacles_spawned}\n"
                f"Total re-plans attempted: {self.total_replans}")
            return

        self._update_metrics()
        self.status_label.config(
            text=f"Status: ✓ New path calculated (Re-plans: {self.total_replans})",
            foreground="#27AE60"
        )

        # Continue movement with new path
        if len(self.agent.path) > 1:
            self.agent_position = self.agent.path[1]
            self.remaining_path_cache = self.agent.path[1:]
            self._draw_grid()
            self.root.update()
            self.root.after(200, self._simulate_movement)
        else:
            self.agent_position = self.agent.goal
            self._draw_grid()
            self.status_label.config(
                text=f"Status: Goal reached! 🎉 (Re-plans: {self.total_replans})",
                foreground="#27AE60"
            )


    def _clear_path(self):
        """Clear path visualization"""
        if not self.agent:
            return

        self.agent.frontier.clear()
        self.agent.visited.clear()
        self.agent.path.clear()
        self.agent.nodes_visited = 0
        self.agent.path_cost = 0
        self.agent.execution_time = 0
        self.agent_position = None
        self.current_exploring = None

        # Reset dynamic mode tracking
        self.spawn_attempts = 0
        self.total_obstacles_spawned = 0
        self.total_replans = 0
        self.remaining_path_cache = []

        self._update_metrics()
        self._draw_grid()
        self.status_label.config(text="Status: Path cleared ✓", foreground="green")

    def _clear_all(self):
        """Clear everything"""
        if not self.agent:
            return

        self.agent.obstacles.clear()

        # Reset dynamic mode tracking
        self.spawn_attempts = 0
        self.total_obstacles_spawned = 0
        self.total_replans = 0
        self.remaining_path_cache = []

        self._clear_path()
        self.status_label.config(text="Status: All cleared ✓", foreground="green")

    def _update_metrics(self):
        """Update metrics display"""
        if not self.agent:
            return

        self.nodes_visited_label.config(text=f"Nodes Visited: {self.agent.nodes_visited}")
        self.path_cost_label.config(text=f"Path Cost: {self.agent.path_cost}")
        self.execution_time_label.config(text=f"Execution Time: {self.agent.execution_time:.2f} ms")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PathfindingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
