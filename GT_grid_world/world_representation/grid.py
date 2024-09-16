import numpy as np
import node

class Graph:
    def __init__(self, ROWS: int, COLS: int, DOF:int, obstacles: list, deterministic = True, file_name: str = None) -> None:
        self.width = COLS
        self.height = ROWS
        self.obstacles = obstacles
        
        if (DOF != 4) and (DOF != 8):
            raise Exception("DOF must be 4 or 8")
        else:
            self.__DOF = DOF
        self.__deterministic = deterministic
        if not file_name:
            self.__graph = self.__initialize_graph(ROWS, COLS, obstacles)
        else:
            self.__graph = self.__load_graph(file_name)
        
    def __load_graph(self, filename):
        graph = []
        cost = 4
        
        f = open("maps/" + filename, "r")
        for i, line in enumerate(f):
            pass
    
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

def main():
    g = Graph(20, 20, 4, [(1, 19), (1, 18), (1, 17), 
                          (3, 19), (3, 18), (3, 17), 
                          (5, 19), (5, 18), (5, 17), 
                          (7, 19), (7, 18), (7, 17),
                          (9, 19), (9, 18), (9, 17)])
    g.draw_grid()

if __name__ == "__main__":
    main()