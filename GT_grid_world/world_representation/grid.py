import numpy as np
from world_representation import node

class Graph:
    def __init__(self, ROWS: int, COLS: int, DOF:int, obstacles: list, deterministic = True) -> None:
        self.width = COLS
        self.height = ROWS
        self.obstacles = obstacles
        
        if (DOF != 4) and (DOF != 8):
            raise Exception("DOF must be 4 or 8")
        else:
            self.__DOF = DOF
        self.__deterministic = deterministic
        self.__graph = self.__initialize_graph(ROWS, COLS, obstacles)
        
    def __initialize_graph(self, ROWS, COLS, obstacles):
        graph = []
        cost = 4
        for i, row in enumerate(range(ROWS)):
            row = []
            for j, col in enumerate(range(COLS)):
                if (i, j) in obstacles:
                    obstacle = True
                else:
                    obstacle = False
                row.append(node.Node(cost, occupied=False, obstacle=obstacle))
            graph.append(row)
        graph = np.asarray(graph)
        return graph
    
    def get_neighbors(self, node: tuple) -> list:
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
        if north[0] < 0  or self.__get_if_occupied(north) :
            north = None
            
        east = (node[0], node[1]+1)
        if east[1] == self.__graph.shape[1] or self.__get_if_occupied(east):
            east = None
            
        south = (node[0]+1, node[1])
        if south[0] == self.__graph.shape[0] or self.__get_if_occupied(south):
            south = None   
        
        west = (node[0], node[1]-1)
        if west[1] < 0 or self.__get_if_occupied(west):
            west = None
        
        neighbors = [north, east, south, west]
        neighbors = [x for x in neighbors if x is not None]
        
        return neighbors
    
    def __get_if_occupied(self, node: tuple) -> bool:
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
    
    def get_cost(self, loc1, loc2): 
        return self.__graph[loc2[0], loc2[1]].get_cost()

def main():
    g = Graph(8, 8, 4, [(5, 4)])
    print(g.get_neighbors((6, 7)))

if __name__ == "__main__":
    main()