import json
import numpy as np

from .item import ItemCategory, Item

class Inventory:
    def __init__(self, *args) -> None:
        
        self.__inventory = {}
        
        if isinstance(args[0], int):
            self.max_pos = args[0]
            self.__init_inventory(args[0])
        elif isinstance(args[0], list):
            self.max_pos = len(args[0])
            self.__init_inventory_from_map(args[0])
            
        if isinstance(args[1], str):
            if args[1] == "uniform":
                self.__uniformly_init_inventory(args[2])
            else:
                raise Exception("Unknown initialization strategy for Inventory object, please use from the list: \n - uniform")
        
    def __init_inventory(self) -> None:
        for i in range(self.max_pos):
            self.__inventory[i] = ItemCategory(1).name
            
    def __init_inventory_from_map(self, locations : list) -> None:
        """Initializes the inventory dictionary with the coordinates and Empty item

        Args:
            locations (list): List of locations on the map where items can be stored
        """
        for location in locations:
            self.__inventory[location] = ItemCategory(1).name
            
    def __uniformly_init_inventory(self, percentage : float) -> None:
        """Initializes the warehouse by uniformly sampling locations and placing items (also chosen uniformly) in those locations

        Args:
            percentage (float): Percentage of the warehouse that should be filled with items
        """
        num_locations = np.ceil(len(self.__inventory)*(percentage/100))
        locations = np.random.choice(len(self.__inventory), int(num_locations), False)
        
        for location in locations:
            self.__inventory[list(self.__inventory.keys())[location]] = ItemCategory(np.random.choice(np.arange(2, len(ItemCategory)+1))).name
    
    def printInventory(self):
        print(self.__inventory)
        
    def readData(self):
        with open("GT_grid_world/data/file.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        self.__inventory = {i: ItemCategory[data[x]].name for i, x in enumerate(data)}
    
    def writeData(self) -> list:
        output = {x: self.__inventory[x] for x in range(len(self.__inventory))}
        with open("GT_grid_world/data/file.json", "w+", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    
    def findEmpty(self) -> list:
        empty = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] == ItemCategory(1).name:
                empty.append(list(self.__inventory.keys())[i])
        return empty
    
    def findFull(self) -> list:
        locations = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] != ItemCategory(1).name:
                locations.append(list(self.__inventory.keys())[i])
        return locations
    
    def find(self, target) -> list:
        locations = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] == 0:
                continue
            if self.__inventory[item] == target:
                locations.append(list(self.__inventory.keys())[i])
        return locations
    
    def add(self, pos, item: str) -> None:
        self.__inventory[pos] = item
    
    def remove(self, pos) -> None:
        self.__inventory[pos] = ItemCategory(1).name