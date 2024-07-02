import numpy as np
import json

from item import Item, ItemCategory

class Warehouse:
    def __init__(self) -> None:
        self.__inventory = {}
        self.__init_inventory()
        
    def __init_inventory(self):
        max_pos = 2659
        
        for i in range(max_pos + 1):
            self.__inventory[i] = 0
    
    def printInventory(self):
        print(self.__inventory)
        
    def readData(self):
        pass
    
    def writeData(self):
        pass
        
    def findEmpty(self):
        empty = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] == 0:
                empty.append(i)
        return empty
    
    def find(self):
        pass
    
    def add(self, pos, item: Item):
        self.__inventory[pos] = item
    
    def remote(self, pos):
        pass
    
    
if __name__=="__main__":
    warehouse = Warehouse()
    
    positions = list(range(2659+1))
    
    for i in range(int(len(positions) - np.floor(len(positions)*.2))):
        pos = np.random.choice(positions, replace=False)
        positions.pop(positions.index(pos))
        item = Item(np.random.randint(0, 99999), ItemCategory(np.random.randint(low=1, high=len(ItemCategory)+1)).name)
        warehouse.add(pos, item)
    print(warehouse.printInventory())
    print(warehouse.findEmpty())
    
