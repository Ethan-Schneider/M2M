from enum import Enum, auto
from typing import Optional
import numpy as np

class WeightInitialization(Enum):
    RANDOM = auto()
    UNIFORM = auto()
    CUSTOM = auto()

class SKU:
    def __init__(self, sku_id: int, 
                 appearance_weight: Optional[float] = None,
                 tasking_weight: Optional[float] = None,
                 weight_init: WeightInitialization = WeightInitialization.RANDOM) -> None:
        """
        Initialize a SKU (Stock Keeping Unit) with its properties.
        
        Args:
            sku_id: Unique identifier for the SKU
            appearance_weight: Weight determining how frequently this SKU appears in the warehouse
            tasking_weight: Weight determining how likely this SKU is to be in a generated task
            weight_init: Method to initialize weights if not provided
        """
        self.__sku_id = sku_id
        
        if appearance_weight is not None and tasking_weight is not None:
            self.__appearance_weight = appearance_weight
            self.__tasking_weight = tasking_weight
        else:
            if weight_init == WeightInitialization.RANDOM:
                self.__appearance_weight = np.random.uniform(0.1, 1.0)
                self.__tasking_weight = np.random.uniform(0.1, 1.0)
            elif weight_init == WeightInitialization.UNIFORM:
                self.__appearance_weight = 1.0
                self.__tasking_weight = 1.0
            else:  # CUSTOM
                raise ValueError("Custom weight initialization requires both appearance_weight and tasking_weight to be provided")
    
    @property
    def sku_id(self) -> int:
        return self.__sku_id
    
    @property
    def appearance_weight(self) -> float:
        return self.__appearance_weight
    
    @property
    def tasking_weight(self) -> float:
        return self.__tasking_weight
    
    def __str__(self) -> str:
        return f"SKU(ID: {self.__sku_id}, Appearance Weight: {self.__appearance_weight}, Tasking Weight: {self.__tasking_weight})"
    
    def __repr__(self) -> str:
        return self.__str__()