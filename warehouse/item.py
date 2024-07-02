from enum import Enum, auto, unique

@unique
class ItemCategory(Enum):
    Shoes = auto()
    Jackets = auto()
    Pants = auto()
    Shampoo = auto()
    Soap = auto()

class Item:
    def __init__(self, itemID: int, item_category: ItemCategory) -> None:
        self.__itemID = itemID
        self.category = item_category
        
        
if __name__=="__main__":
    print(ItemCategory.Shoes.value)
    print(list(ItemCategory))
    print(ItemCategory(1).name)