import numpy as np
import os
from scipy.spatial import KDTree

from .node import Node
from .inventory_manager.inventory import Inventory, WeightInitialization

def manhattan_distance(p1 : tuple, p2 : tuple) -> float: 
    return np.abs(p2[1] - p1[1]) + np.abs(p2[0] - p1[0])

class AStar:
    def __init__(self, G, start : tuple, goal : tuple) -> None:
        self.admissible_heuristic = manhattan_distance
        self.goal = goal
        self.start = start
        self.get_neighbors = G.get_neighbors
        self.G = G  # Store reference to graph to check obstacles

    def reconstruct_path(self, came_from : dict, current : tuple) -> list:
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self):
        """
        low level search 
        """
        initial_state = self.start
        step_cost = 1
        
        # Check if start or goal is an obstacle
        if self.G.get_if_obstacle(initial_state) or self.G.get_if_obstacle(self.goal):
            print(f"Start or goal is an obstacle")
            print(f"Start: {self.G.get_if_obstacle(initial_state)}")
            print(f"Goal: {self.G.get_if_obstacle(self.goal)}")
            return False
        
        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        g_score = {} 
        g_score[initial_state] = 0

        f_score = {} 

        f_score[initial_state] = self.admissible_heuristic(initial_state, initial_state)

        while open_set:
            temp_dict = {open_item:f_score.setdefault(open_item, float("inf")) for open_item in open_set}
            current = min(temp_dict, key=temp_dict.get)

            # If current state is goal, return path
            if self.goal == current:
                return self.reconstruct_path(came_from, current)

            # Add current to closed and remove from open sets
            open_set -= {current}
            closed_set |= {current}

            # Get list of neighbors from current node - use ignore_robots=False to respect obstacles
            neighbor_list = self.get_neighbors(current, False)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                if neighbor not in open_set:
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(neighbor, self.goal)
        if self.start == (12, 30) and self.goal == (2, 38):
            print(f"No path found")
        return False

class Graph:
    def __init__(self, num_robots : int, file_name: str = None, initial_warehouse_capacity: float = 25.0, num_skus: int = 10, weight_init_method: str = "random") -> None:
        self.__num_robots = num_robots
        self.__occupancy_graph, self.__obstacle_graph = self.__load_graph(file_name, initial_warehouse_capacity, num_skus, weight_init_method)
        self.__aisle_start, self.__driveway_start = self.__get_aisle_driveway_start()
        
        # Initialize distance matrix
        self.__distance_matrix = None
        if file_name:
            self.__load_or_compute_distance_matrix(file_name)

        self.__sku_KD_trees = {}
        self.__build_sku_KD_trees(num_skus)

    def query_sku_KD_trees(self, sku_id: int, location: tuple, num_neighbors: int) -> list:
        if self.__sku_KD_trees[sku_id] is None:
            if num_neighbors == 1:
                return [0]
            else:
                return [(0, 0)]
        else:
            if self.__sku_KD_trees[sku_id].n > 1: 
                return self.__sku_KD_trees[sku_id].query(location, k=num_neighbors, p=1)
            else:
                if num_neighbors == 1:
                    return [0]
                else:
                    return [(0, 0)]
            # print(f"SKU KD Tree values: {self.__sku_KD_trees[sku_id].query(location, k=num_neighbors, p=1)}")

    def update_sku_KD_trees(self, sku_id: int) -> None:
        sku_locations = self.warehouse.get_sku_instances(sku_id)
        if sku_locations == []:
            self.__sku_KD_trees[sku_id] = None
        else:
            self.__sku_KD_trees[sku_id] = KDTree(sku_locations)

    def __build_sku_KD_trees(self, num_skus: int) -> None:
        for sku_id in range(1, num_skus + 1):
            sku_locations = self.warehouse.get_sku_instances(sku_id)
            if sku_locations == []:
                self.__sku_KD_trees[sku_id] = None
            else:
                self.__sku_KD_trees[sku_id] = KDTree(sku_locations)

    def __load_graph(self, filename : str, initial_warehouse_capacity : float, num_skus : int, weight_init_method : str):
        """This method takes in a map file, parses the metadata and map data, 
        then saves a map representation, warehouse and driveway item representation, 
        height, width, and initializes robot start locations.

        Args:
            filename (str): Filename of the map
            weight_init_method (str): Method for initializing SKU weights
            initial_warehouse_capacity (float): Initial fill percentage for the warehouse

        Raises:
            Exception: If the number of robots the user wants to generate exceeds
            the maximum number of robots defined in the map file, raise an exception.
        """
        occupancy_graph = []
        obstacle_graph = []
        cost = 4
        # Read-in map file
        f = open(filename, "r")
        
        # Read the map file's metadata
        map_data = f.readline().split(" ")
        for i in range(len(map_data)):
            map_data[i] = int(map_data[i].replace("\n", "").replace(",", ""))
        
        self.height = map_data[0]
        self.width = map_data[1]
        num_warehouse_locations = map_data[2]
        num_driveway_locations = map_data[3]
        max_num_robots = map_data[4]
        
        # Read the number of empty spaces in the map
        empty_points = []
        f.readline()
        for i, line in enumerate(f):
            for j, character in enumerate(line):
                if character == ".":
                    empty_points.append((i, j))
                    
        # Init lists for warehouse and driveway locations, possible robot start locations, and obstacles            
        warehouse_locations = []
        driveway_locations = []
        robot_start_locations = []
        self.obstacles = []
        self.aisle_locations = []  # Track aisle locations (e)
        self.station_locations = []  # Track inbound/outbound stations (s)
        self.all_locations = []
                
        # Loop through each character in the map, generating the graph list with Node objects, and saving information into the above lists
        f = open(filename, "r")
        f.readline()
        f.readline()
        for i, line in enumerate(f):
            row = []
            obstacle_row = []
            for j, character in enumerate(line):
                self.all_locations.append((i, j))
                if character == "@":
                    self.obstacles.append((i, j))
                    row.append(Node(cost, occupied=False, obstacle=True))
                    obstacle_row.append(1)
                elif character == "e":
                    obstacle_row.append(0)
                    row.append(Node(cost, occupied=False, obstacle=False))
                    self.aisle_locations.append((i, j))
                elif character == "s":
                    obstacle_row.append(0)
                    row.append(Node(cost, occupied=False, obstacle=False))
                    self.station_locations.append((i, j))
                elif character == ".":
                    obstacle_row.append(0)
                    row.append(Node(cost, occupied=False, obstacle=False))
                    # if (i, j) in empty_points[:num_warehouse_locations]:
                    #     warehouse_locations.append((i, j))
                    # elif (i, j) in empty_points[(-1*num_driveway_locations):]:
                    #     driveway_locations.append((i, j))
                    # else:
                    #     pass
                elif character == "r":
                    obstacle_row.append(0)
                    row.append(Node(cost, occupied=False, obstacle=False))
                    robot_start_locations.append((i, j))
            occupancy_graph.append(row)
            obstacle_graph.append(obstacle_row)
        
        # Initialize Warehouse and Driveway as Inventory objects
        weight_init = WeightInitialization.RANDOM if weight_init_method == "random" else WeightInitialization.UNIFORM
        
        self.warehouse = Inventory(
            num_skus=num_skus,
            warehouse_locations=self.aisle_locations,
            fill_percentage=initial_warehouse_capacity,
            weight_init=weight_init
        )
        
        self.driveway = Inventory(
            num_skus=num_skus,
            warehouse_locations=self.station_locations,
            fill_percentage=0.0,
            weight_init=weight_init
        )
        
        if self.__num_robots > max_num_robots:
            raise Exception("Number of robots exceeds maximum number of robots for map")
        
        # Randomly place N robots into the environment's M start locations
        for i in range(self.__num_robots):
            location = robot_start_locations[np.random.choice(len(robot_start_locations))]
            occupancy_graph[location[0]][location[1]].set_occupied(True)
            robot_start_locations.remove(location)
        
        return np.asarray(occupancy_graph), np.asarray(obstacle_graph)
    
    def __get_aisle_driveway_start(self):
        on_aisle = True
        for i, row in enumerate(self.__obstacle_graph):
            if on_aisle:
                if 1 in row:
                    continue
                else:
                    beginning_aisle = i-1
                    on_aisle = False
            else:
                if 1 in row:
                    beginning_driveway = i
                    break
                else:
                    continue
            
        return beginning_aisle, beginning_driveway
    
    def get_neighbors(self, node: tuple, ignore_robots : bool = False) -> list:
        """Returns list of non-occupied neighbor nodes in order (N, E, S, W).

        Args:
            node (tuple): _description_

        Raises:
            Exception: _description_

        Returns:
            list: _description_
        """
        if node[0] >= self.__occupancy_graph.shape[0] or node[1] >= self.__occupancy_graph.shape[1] or node[0] < 0 or node[1] < 0:
            raise Exception("Node %s is out of range of graph with shape %s" % (node, self.__occupancy_graph.shape))
        
        north = (node[0]-1, node[1])
        east = (node[0], node[1]+1)
        south = (node[0]+1, node[1])
        west = (node[0], node[1]-1)
        
        if ignore_robots: 
            if north[0] < 0  or self.get_if_obstacle(north):
                north = None
                
            if east[1] == self.__occupancy_graph.shape[1] or self.get_if_obstacle(east):
                east = None
                
            if south[0] == self.__occupancy_graph.shape[0] or self.get_if_obstacle(south):
                south = None   
            
            if west[1] < 0 or self.get_if_obstacle(west):
                west = None            
        else:
            if north[0] < 0  or self.__get_if_occupied(north) or self.get_if_obstacle(north):
                north = None
                
            if east[1] == self.__occupancy_graph.shape[1] or self.__get_if_occupied(east) or self.get_if_obstacle(east):
                east = None
                
            if south[0] == self.__occupancy_graph.shape[0] or self.__get_if_occupied(south) or self.get_if_obstacle(south):
                south = None   
            
            if west[1] < 0 or self.__get_if_occupied(west) or self.get_if_obstacle(west):
                west = None
            
        neighbors = [north, east, south, west]
        # Remove neighbors which go into obstacles or beyond the border of the map
        neighbors = [x for x in neighbors if x is not None]
        return neighbors
    
    def __get_if_occupied(self, node: tuple) -> bool:
        return self.__occupancy_graph[node[0], node[1]].get_occupied()
    
    def get_if_obstacle(self, node: tuple) -> bool:
        return self.__occupancy_graph[node[0], node[1]].get_obstacle()
    
    def get_aisle_occupancy(self) -> list:
        aisle_occupied = []
        for col in range(self.__occupancy_graph.shape[1]):
            # Skip if the column is an obstacle
            if self.__obstacle_graph[self.__aisle_start, col] == 1:
                continue
            agent_count = np.count_nonzero(np.array([x.get_occupied() for x in self.__occupancy_graph[0:self.__aisle_start, col]]))
            aisle_occupied.append(int(agent_count))
        return aisle_occupied
    
    def get_driveway_occupancy(self) -> list:
        driveway_occupied = []
        for col in range(self.__occupancy_graph.shape[1]):
            # Skip if the column is an obstacle
            if self.__obstacle_graph[self.__driveway_start, col] == 1:
                continue
            agent_count = np.count_nonzero(np.array([x.get_occupied() for x in self.__occupancy_graph[self.__driveway_start:, col]]))
            driveway_occupied.append(int(agent_count))
        return driveway_occupied
    
    def get_all_occupied(self) -> list:
        occupied = []
        for row in range(self.__occupancy_graph.shape[0]):
            for col in range(self.__occupancy_graph.shape[1]):
                if self.__occupancy_graph[row, col].get_occupied():
                    occupied.append((row, col))
        return occupied
    
    def get_all_unoccupied(self) -> list:
        unoccupied = []
        for row in range(self.__occupancy_graph.shape[0]):
            for col in range(self.__occupancy_graph.shape[1]):
                if not self.__occupancy_graph[row, col].get_occupied():
                    unoccupied.append((row, col))
        return unoccupied
    
    def get_all_non_obstacles(self) -> list:
        empty_space = []
        for row in range(self.__occupancy_graph.shape[0]):
            for col in range(self.__occupancy_graph.shape[1]):
                if not self.__occupancy_graph[row, col].get_obstacle():
                    empty_space.append((row, col))
        return empty_space
    
    def set_occupied(self, node : tuple, occupied : bool) -> None:
        self.__occupancy_graph[node[0], node[1]].set_occupied(occupied)
    
    def get_cost(self, loc1, loc2): 
        return self.__occupancy_graph[loc2[0], loc2[1]].get_cost()
        
    def get_graph_size(self):
        return self.__occupancy_graph.shape
    
    def draw_tile(self, id, style):
        r = " . "
        if 'number' in style and id in style['number']: r = " %-2d" % style['number'][id]
        if 'point_to' in style and style['point_to'].get(id, None) is not None:
            (x1, y1) = id
            (x2, y2) = style['point_to'][id]
            if x2 == x1 + 1: r = " > "
            if x2 == x1 - 1: r = " < "
            if y2 == y1 + 1: r = " v "
            if y2 == y1 - 1: r = " ^ "
        if 'path' in style and id in style['path']:   r = " @ "
        if 'start' in style and id == style['start']: r = " A "
        if 'goal' in style and id == style['goal']:   r = " Z "
        if id in self.obstacles: r = "###"
        return r

    def draw_grid(self, **style):
        print("___" * self.width)
        for y in range(self.height):
            for x in range(self.width):
                print("%s" % self.draw_tile((x, y), style), end="")
            print()
        print("~~~" * self.width)
        
    def draw_graph(self):
        graph = ""
        for row in range(self.__occupancy_graph.shape[0]):
            line = ""
            for col in range(self.__occupancy_graph.shape[1]):
                if self.__occupancy_graph[row, col].get_obstacle():
                    line += "@"
                elif self.__occupancy_graph[row, col].get_occupied():
                    line += "r"
                else:
                    line += "."
            line += "\n"
            graph += line
        print(graph)
            
    def draw_graph_with_highlight(self, positions : list) -> None:
        graph = ""
        for row in range(self.__occupancy_graph.shape[0]):
            line = ""
            for col in range(self.__occupancy_graph.shape[1]):
                if self.__occupancy_graph[row, col].get_obstacle():
                    line += "@"
                elif self.__occupancy_graph[row, col].get_occupied():
                    line += "r"
                elif (row, col) in positions:
                    line += "0"
                else:
                    line += "."
            line += "\n"
            graph += line
        print(graph)
                    
    def get_aisle_locations(self) -> list:
        """Returns list of aisle locations (e) in the map."""
        return self.aisle_locations

    def get_station_locations(self) -> list:
        """Returns list of inbound/outbound station locations (s) in the map."""
        return self.station_locations
                    
    def __load_or_compute_distance_matrix(self, map_file: str) -> None:
        """Load the distance matrix from file if it exists, otherwise compute and save it.
        
        Args:
            map_file (str): Path to the map file
        """
        # Get the directory and base filename
        dir_name = os.path.dirname(map_file)
        base_name = os.path.splitext(os.path.basename(map_file))[0]
        matrix_file = os.path.join(dir_name, f"{base_name}_distances.npy")
        
        # Try to load existing matrix
        if os.path.exists(matrix_file):
            self.__distance_matrix = np.load(matrix_file)
            return
            
        # Compute new matrix
        print("Computing distance matrix...")
        n = self.height * self.width
        self.__distance_matrix = np.full((n, n), np.inf)
        
        # Set diagonal to 0
        np.fill_diagonal(self.__distance_matrix, 0)
        
        # Get all non-obstacle locations
        valid_locations = self.get_all_non_obstacles()
        
        # Create a temporary graph without robot positions for distance computation
        temp_occupancy = self.__occupancy_graph.copy()
        # Clear all robot positions
        for row in range(self.__occupancy_graph.shape[0]):
            for col in range(self.__occupancy_graph.shape[1]):
                if not self.__occupancy_graph[row, col].get_obstacle():
                    temp_occupancy[row, col].set_occupied(False)
        
        # Compute distances between all pairs
        for i, start in enumerate(valid_locations):
            if i % 10 == 0:
                print(f"Computing distances for location {i}/{len(valid_locations)}")
            
            for goal in valid_locations:
                if start == goal:
                    continue
                    
                # Convert 2D coordinates to 1D index
                start_idx = start[0] * self.width + start[1]
                goal_idx = goal[0] * self.width + goal[1]
                
                # Skip if we already computed this pair
                if self.__distance_matrix[goal_idx, start_idx] != np.inf:
                    self.__distance_matrix[start_idx, goal_idx] = self.__distance_matrix[goal_idx, start_idx]
                    continue
                
                # Create a temporary graph for A* search without robot positions
                class TempGraph:
                    def __init__(self, occupancy_graph, obstacle_graph, height, width):
                        self.__occupancy_graph = occupancy_graph
                        self.__obstacle_graph = obstacle_graph
                        self.height = height
                        self.width = width
                    
                    def get_if_obstacle(self, node):
                        return self.__occupancy_graph[node[0], node[1]].get_obstacle()
                    
                    def get_neighbors(self, node, ignore_robots=False):
                        if node[0] >= self.__occupancy_graph.shape[0] or node[1] >= self.__occupancy_graph.shape[1] or node[0] < 0 or node[1] < 0:
                            raise Exception("Node %s is out of range of graph with shape %s" % (node, self.__occupancy_graph.shape))
                        
                        north = (node[0]-1, node[1])
                        east = (node[0], node[1]+1)
                        south = (node[0]+1, node[1])
                        west = (node[0], node[1]-1)
                        
                        if ignore_robots: 
                            if north[0] < 0  or self.get_if_obstacle(north):
                                north = None
                                
                            if east[1] == self.__occupancy_graph.shape[1] or self.get_if_obstacle(east):
                                east = None
                                
                            if south[0] == self.__occupancy_graph.shape[0] or self.get_if_obstacle(south):
                                south = None   
                            
                            if west[1] < 0 or self.get_if_obstacle(west):
                                west = None            
                        else:
                            if north[0] < 0  or self.__get_if_occupied(north) or self.get_if_obstacle(north):
                                north = None
                                
                            if east[1] == self.__occupancy_graph.shape[1] or self.__get_if_occupied(east) or self.get_if_obstacle(east):
                                east = None
                                
                            if south[0] == self.__occupancy_graph.shape[0] or self.__get_if_occupied(south) or self.get_if_obstacle(south):
                                south = None   
                            
                            if west[1] < 0 or self.__get_if_occupied(west) or self.get_if_obstacle(west):
                                west = None
                        
                        neighbors = [north, east, south, west]
                        neighbors = [x for x in neighbors if x is not None]
                        return neighbors
                    
                    def __get_if_occupied(self, node):
                        return self.__occupancy_graph[node[0], node[1]].get_occupied()
                
                temp_graph = TempGraph(temp_occupancy, self.__obstacle_graph, self.height, self.width)
                
                # Compute path using A*
                if start == (12, 30) and goal == (2, 38):
                    print(f"Computing path from {start} to {goal}")
                path = AStar(temp_graph, start, goal).search()
                if start == (12, 30) and goal == (2, 38):
                    print(f"Path: {path}")
                if path:
                    self.__distance_matrix[start_idx, goal_idx] = len(path) - 1  # -1 because path includes start node
                else:
                    # Check if either location is an obstacle
                    if self.get_if_obstacle(start):
                        print(f"  Start location {start} is an obstacle")
                    if self.get_if_obstacle(goal):
                        print(f"  Goal location {goal} is an obstacle")
                    
        # Save the matrix
        np.save(matrix_file, self.__distance_matrix)
        print("Distance matrix saved to", matrix_file)
        
    def get_distance(self, start: tuple, goal: tuple) -> float:
        """Get the shortest path distance between two locations.
        
        Args:
            start (tuple): Starting location (row, col)
            goal (tuple): Goal location (row, col)
            
        Returns:
            float: Shortest path distance, or inf if no path exists
        """
        if self.__distance_matrix is None:
            raise ValueError("Distance matrix not initialized. Make sure to provide a map file when creating the Graph.")
            
        start_idx = start[0] * self.width + start[1]
        goal_idx = goal[0] * self.width + goal[1]
        
        return self.__distance_matrix[start_idx, goal_idx]
    
    def get_distance_matrix(self) -> np.ndarray:
        return self.__distance_matrix
                    