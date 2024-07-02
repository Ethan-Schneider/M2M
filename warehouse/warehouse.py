import numpy as np
import json

from item import ItemCategory

class Warehouse:
    def __init__(self) -> None:
        self.__inventory = {}
        self.__init_inventory()
        
    def __init_inventory(self):
        # Defined by current warehouse layout
        max_pos = 2659
        
        for i in range(max_pos + 1):
            self.__inventory[i] = ItemCategory(1).name
    
    def printInventory(self):
        print(self.__inventory)
        
    def readData(self):
        with open("warehouse/data/file.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        self.__inventory = {i: ItemCategory[data[x]].name for i, x in enumerate(data)}
    
    def writeData(self):
        output = {x: self.__inventory[x] for x in range(len(self.__inventory))}
        with open("warehouse/data/file.json", "w+", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        
    def findEmpty(self):
        empty = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] == ItemCategory(1).name:
                empty.append(i)
        return empty
    
    def find(self, target):
        locations = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] == 0:
                continue
            if self.__inventory[item] == target:
                locations.append(i)
        return locations
    
    def add(self, pos, item: str):
        self.__inventory[pos] = item
    
    def remove(self, pos):
        self.__inventory[pos] = ItemCategory(1).name
    
    
if __name__=="__main__":
    warehouse = Warehouse()
    
    # Init Warehouse
    # positions = list(range(2659+1))
    
    # for i in range(int(len(positions) - np.floor(len(positions)*.2))):
    #     pos = np.random.choice(positions, replace=False)
    #     positions.pop(positions.index(pos))
    #     item = ItemCategory(np.random.randint(low=2, high=len(ItemCategory)+1)).name
    #     warehouse.add(pos, item)
    
    warehouse.readData()
    
    print(warehouse.findEmpty())
    
    print(warehouse.find(ItemCategory(2).name))
    print(len(warehouse.find(ItemCategory(2).name)))
    
    # warehouse.writeData()
    
