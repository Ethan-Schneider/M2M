from enum import Enum, auto, unique

@unique
class ItemCategory(Enum):
    EMPTY = auto()
    Shoes = auto()
    Jackets = auto()
    Pants = auto()
    Shampoo = auto()
    Soap = auto()