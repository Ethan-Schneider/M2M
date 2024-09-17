import json
import numpy as np

from .item import ItemCategory


class Inventory:
    def __init__(self, *args) -> None:
        
        self.__inventory = {}
        
        if isinstance(args[0], int):
            self.__init_inventory(args[0])
        elif isinstance(args[0], list):
            self.__init_inventory_from_map(args[0])
        
    def __init_inventory(self, max_pos) -> None:
        for i in range(self.max_pos):
            self.__inventory[i] = ItemCategory(1).name
            
    def __init_inventory_from_map(self, locations : list) -> None:
        for location in locations:
            self.__inventory[location] = ItemCategory(1).name
    
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
                empty.append(i)
        return empty
    
    def findFull(self) -> list:
        locations = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] != ItemCategory(1).name:
                locations.append(i)
        return locations
    
    def find(self, target) -> list:
        locations = []
        for i, item in enumerate(self.__inventory):
            if self.__inventory[item] == 0:
                continue
            if self.__inventory[item] == target:
                locations.append(i)
        return locations
    
    def add(self, pos, item: str) -> None:
        self.__inventory[pos] = item
    
    def remove(self, pos) -> None:
        self.__inventory[pos] = ItemCategory(1).name