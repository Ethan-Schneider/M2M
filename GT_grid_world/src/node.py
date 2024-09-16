import numpy as np

class Node:
    def __init__(self, cost, occupied = False, obstacle = False) -> None:
        self.__cost = cost
        self.__obstacle = obstacle
        if self.__obstacle:
            self.__occupied = True
        else:
            self.__occupied = occupied
            
    def get_occupied(self) -> bool:
        return self.__occupied
    
    def get_cost(self):
        return self.__cost
            
    def __repr__(self):
        return str(self.__occupied)        
    
    def __str__(self):
        return str(self.__occupied)
            
            
    