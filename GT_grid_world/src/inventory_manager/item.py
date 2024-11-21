from enum import Enum, auto, unique

@unique
class ItemCategory(Enum):
    EMPTY = auto()
    Shoes = auto()
    Jackets = auto()
    Pants = auto()
    Shampoo = auto()
    Soap = auto()
    
    
    
class Item():
    def __init__(self, loc : tuple = (-1, -1), item_SKU : int = -1) -> None:
        if item_SKU != -1:
            self.__SKU = item_SKU
        else:
            # TODO: Randomly choose item to initialize object to
            pass
        
        # TODO: Check whether the given loc already has an item, and if so, return error or randomize location
        if loc != (-1, -1):
            self.__location = loc
        else:
            # TODO: Randomize location
            pass
        
        # History of the path taken for the item
        path_history = [self.__location]
        
    def update_location(self, new_loc : tuple) -> None:
        self.__location = new_loc
        
    def set_location_to_robot(self, robot_loc : tuple) -> None:
        self.__location = robot_loc
        
    def return_item_location(self) -> tuple:
        return self.__location
    
    def return_item_SKU(self) -> int:
        return self.__SKU