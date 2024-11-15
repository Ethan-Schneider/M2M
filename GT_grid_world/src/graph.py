import numpy as np

from .node import Node
from .inventory import Inventory

class Graph:
    def __init__(self, num_robots : int, file_name: str = None, DOF:int = 4, deterministic = True, *args) -> None:
        if (DOF != 4) and (DOF != 8):
            raise Exception("DOF must be 4 or 8")
        else:
            self.__DOF = DOF
        self.__deterministic = deterministic
        self.__num_robots = num_robots
        self.__graph = self.__load_graph(file_name, args[0], args[1])
        
    def __load_graph(self, filename : str, warehouse_strategy : str, initial_warehouse_capacity : float):
        """This method takes in a map file, parses the metadata and map data, 
        then saves a map representation, warehouse and driveway item representation, 
        height, width, and initializes robot start locations.

        Args:
            filename (str): Filename of the map
            warehouse_strategy (str): Strategy for how to generate the initial warehouse inventory

        Raises:
            Exception: If the number of robots the user wants to generate exceeds
            the maximum number of robots defined in the map file, raise an exception.
        """
        graph = []
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
                
        # Loop through each character in the map, generating the graph list with Node objects, and saving information into the above lists
        f = open(filename, "r")
        f.readline()
        f.readline()
        for i, line in enumerate(f):
            row = []
            for j, character in enumerate(line):
                if character == "@":
                    self.obstacles.append((i, j))
                    row.append(Node(cost, occupied=False, obstacle=True))
                elif character == ".":
                    row.append(Node(cost, occupied=False, obstacle=False))
                    if (i, j) in empty_points[:num_warehouse_locations]:
                        warehouse_locations.append((i, j))
                    elif (i, j) in empty_points[(-1*num_driveway_locations):]:
                        driveway_locations.append((i, j))
                    else:
                        pass
                elif character == "r":
                    row.append(Node(cost, occupied=False, obstacle=False))
                    robot_start_locations.append((i, j))
            graph.append(row)
        
        # Initialize Warehouse and Driveway as Warehouse objects 
        # Note: it will populate the warehouse with the given strategy
        # TODO: Currently populating driveway in the same state as the warehouse, should consider changing  
        self.warehouse = Inventory(warehouse_locations, warehouse_strategy, initial_warehouse_capacity)
        self.driveway = Inventory(driveway_locations, warehouse_strategy, initial_warehouse_capacity)
        
        
        if self.__num_robots > max_num_robots:
            raise Exception("Number of robots exceeds maximum number of robots for map")
        
        # Randomly place N robots into the environment's M start locations
        for i in range(self.__num_robots):
            location = robot_start_locations[np.random.choice(len(robot_start_locations))]
            graph[location[0]][location[1]].set_occupied(True)
            robot_start_locations.remove(location)
        
        return np.asarray(graph)
    
    def get_neighbors(self, node: tuple, ignore_robots : bool = False) -> list:
        """Returns list of non-occupied neighbor nodes in order (N, E, S, W).

        Args:
            node (tuple): _description_

        Raises:
            Exception: _description_

        Returns:
            list: _description_
        """
        if node[0] >= self.__graph.shape[0] or node[1] >= self.__graph.shape[1] or node[0] < 0 or node[1] < 0:
            raise Exception("Node %s is out of range of graph with shape %s" % (node, self.__graph.shape))
        
        north = (node[0]-1, node[1])
        east = (node[0], node[1]+1)
        south = (node[0]+1, node[1])
        west = (node[0], node[1]-1)
        
        if ignore_robots: 
            if north[0] < 0  or self.__get_if_obstacle(north):
                north = None
                
            if east[1] == self.__graph.shape[1] or self.__get_if_obstacle(east):
                east = None
                
            if south[0] == self.__graph.shape[0] or self.__get_if_obstacle(south):
                south = None   
            
            if west[1] < 0 or self.__get_if_obstacle(west):
                west = None            
        else:
            if north[0] < 0  or self.__get_if_occupied(north):
                north = None
                
            if east[1] == self.__graph.shape[1] or self.__get_if_occupied(east):
                east = None
                
            if south[0] == self.__graph.shape[0] or self.__get_if_occupied(south):
                south = None   
            
            if west[1] < 0 or self.__get_if_occupied(west):
                west = None
            
        neighbors = [north, east, south, west]
        # Remove neighbors which go into obstacles or beyond the border of the map
        neighbors = [x for x in neighbors if x is not None]
        return neighbors
    
    def __get_if_occupied(self, node: tuple) -> bool:
        return self.__graph[node[0], node[1]].get_occupied()
    
    def __get_if_obstacle(self, node: tuple) -> bool:
        return self.__graph[node[0], node[1]].get_occupied()
    
    def get_all_occupied(self) -> list:
        occupied = []
        for row in range(self.__graph.shape[0]):
            for col in range(self.__graph.shape[1]):
                if self.__graph[row, col].get_occupied():
                    occupied.append((row, col))
        return occupied
    
    def get_all_unoccupied(self) -> list:
        unoccupied = []
        for row in range(self.__graph.shape[0]):
            for col in range(self.__graph.shape[1]):
                if not self.__graph[row, col].get_occupied():
                    unoccupied.append((row, col))
        return unoccupied
    
    def get_all_non_obstacles(self) -> list:
        empty_space = []
        for row in range(self.__graph.shape[0]):
            for col in range(self.__graph.shape[1]):
                if not self.__graph[row, col].get_obstacle():
                    empty_space.append((row, col))
        return empty_space
    
    def set_occupied(self, node : tuple, occupied : bool) -> None:
        self.__graph[node[0], node[1]].set_occupied(occupied)
    
    def get_cost(self, loc1, loc2): 
        return self.__graph[loc2[0], loc2[1]].get_cost()
        
    
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
        for row in range(self.__graph.shape[0]):
            for col in range(self.__graph.shape[1]):
                if self.__graph.shape[row, col].get_obstacle():
                    print("@")
                elif self.__graph.shape[row, col].get_occupied():
                    print("r")
                else:
                    print(".")
            print("\n")
                    

def main():
    g = Graph(20, 20, 4, [(1, 19), (1, 18), (1, 17), 
                          (3, 19), (3, 18), (3, 17), 
                          (5, 19), (5, 18), (5, 17), 
                          (7, 19), (7, 18), (7, 17),
                          (9, 19), (9, 18), (9, 17)])
    g.draw_grid()

if __name__ == "__main__":
    main()