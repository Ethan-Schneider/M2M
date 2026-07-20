class Node:
    def __init__(self, cost, occupied = False, obstacle = False) -> None:
        self.__cost = cost
        self.__obstacle = obstacle
        self.__occupied = occupied
    
    def set_occupied(self, occupied : bool) -> None:
        self.__occupied = occupied
            
    def get_occupied(self) -> bool:
        return self.__occupied
    
    def get_obstacle(self) -> bool:
        return self.__obstacle
    
    def get_cost(self):
        return self.__cost
            
    