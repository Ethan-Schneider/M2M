import json
import numpy as np

from .item import ItemCategory

class Warehouse:
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
    
    
def CRG(J: set, W: Warehouse, DW: Warehouse, N: int, inbound_to_outbound: float, last_task_id: int, strategy: str = "uniform", item : list = []) -> set:
    """_summary_

    Args:
        J (set): _description_
        W (Warehouse): _description_
        DW (Warehouse): _description_
        N (int): _description_
        strategy (str, optional): _description_. Defaults to uniform.
        item (list, optional): _description_. Defaults to [].

    Returns:
        set: _description_
    """
    inbound_probability = inbound_to_outbound/(inbound_to_outbound + 1)
    outbound_probability = 1 - inbound_probability
    tasks_to_generate = np.random.choice([0, 1], size=N, p=[outbound_probability, inbound_probability])
    J_new = set([])
    
    #Generate tasks uniformly throughout the warehouse without inventory information
    if strategy == "uniform":
        for task in tasks_to_generate:
            #Generate inbound task
            if task == 1:
                J_new = J_new | set([(last_task_id + 1, np.random.choice(np.arange(DW.max_pos)), np.random.choice(np.arange(W.max_pos)))])
                last_task_id += 1
            #Generate outbound task
            elif task == 0:
                J_new = J_new | set([(last_task_id + 1, np.random.choice(np.arange(W.max_pos)), np.random.choice(np.arange(DW.max_pos)))])
                last_task_id += 1
    return J_new, last_task_id
    
if __name__=="__main__":
    warehouse = Warehouse()
    
    # Init Warehouse
    # positions = list(range(2659+1))
    
    # for i in range(int(len(positions) - np.floor(len(positions)*.2))):
    #     pos = np.random.choice(positions, replace=False)
    #     positions.pop(positions.index(pos))
    #     item = ItemCategory(np.random.randint(low=2, high=len(ItemCategory)+1)).name
    #     warehouse.add(pos, item)
    
    # warehouse.readData()
    
    # print(warehouse.findEmpty())
    
    # print(warehouse.find(ItemCategory(2).name))
    # print(len(warehouse.find(ItemCategory(2).name)))
    
    # warehouse.writeData()
    
