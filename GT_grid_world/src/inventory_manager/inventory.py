import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from .item import SKU, WeightInitialization

class Inventory:
    def __init__(self, num_skus: int, warehouse_locations: List[tuple], 
                 fill_percentage: float = 0.0,
                 weight_init: WeightInitialization = WeightInitialization.RANDOM,
                 custom_weights: Optional[Dict[int, Tuple[float, float]]] = None) -> None:
        """
        Initialize the inventory system.
        
        Args:
            num_skus: Number of unique SKUs to create
            warehouse_locations: List of valid locations in the warehouse
            fill_percentage: Percentage of warehouse to fill (0.0 to 100.0)
            weight_init: Method to initialize SKU weights
            custom_weights: Dictionary mapping SKU IDs to (appearance_weight, tasking_weight) tuples
        """
        self.__skus: Dict[int, SKU] = {}  # SKU ID -> SKU mapping
        self.__location_to_sku: Dict[tuple, int] = {}  # Location -> SKU ID mapping
        self.__sku_instances: Dict[int, List[tuple]] = {}  # SKU ID -> List of locations
        self.__warehouse_locations = warehouse_locations
        
        # Initialize SKUs
        for sku_id in range(1, num_skus + 1):
            if custom_weights and sku_id in custom_weights:
                appearance_weight, tasking_weight = custom_weights[sku_id]
                self.__skus[sku_id] = SKU(sku_id, appearance_weight, tasking_weight)
            else:
                self.__skus[sku_id] = SKU(sku_id, weight_init=weight_init)
            self.__sku_instances[sku_id] = []
        
        if fill_percentage > 0:
            self.__uniformly_init_inventory(fill_percentage)
    
    def __uniformly_init_inventory(self, percentage: float) -> None:
        """
        Initialize the warehouse by placing SKU instances based on their appearance weights.
        
        Args:
            percentage: Percentage of the warehouse that should be filled with items
        """
        num_locations = int(np.ceil(len(self.__warehouse_locations) * (percentage / 100)))
        available_locations = self.__warehouse_locations.copy()
        
        # Calculate total appearance weight for sampling
        total_weight = sum(sku.appearance_weight for sku in self.__skus.values())
        weights = [sku.appearance_weight / total_weight for sku in self.__skus.values()]
        sku_ids = list(self.__skus.keys())
        
        # Sample locations and SKUs
        for _ in range(num_locations):
            if not available_locations:
                break
                
            # Choose a location
            loc_idx = np.random.choice(len(available_locations))
            location = available_locations.pop(loc_idx)
            
            # Choose an SKU based on appearance weights
            sku_id = np.random.choice(sku_ids, p=weights)
            
            # Place the SKU instance
            self.add_sku_instance(sku_id, location)
    
    def add_sku_instance(self, sku_id: int, location: tuple) -> None:
        """
        Add a SKU instance to a specific location.
        
        Args:
            sku_id: The SKU ID to add
            location: The location to add the SKU instance to
        """
        if sku_id not in self.__skus:
            raise ValueError(f"SKU {sku_id} does not exist")
        if location not in self.__warehouse_locations:
            raise ValueError(f"Location {location} is not a valid warehouse location")
        if location in self.__location_to_sku:
            raise ValueError(f"Location {location} is already occupied")
            
        self.__sku_instances[sku_id].append(location)
        self.__location_to_sku[location] = sku_id
    
    def remove_sku_instance(self, location: tuple) -> None:
        """
        Remove a SKU instance from a specific location.
        
        Args:
            location: The location to remove the SKU instance from
        """
        if location not in self.__location_to_sku:
            raise ValueError(f"No SKU instance at location {location}")
            
        sku_id = self.__location_to_sku[location]
        self.__sku_instances[sku_id].remove(location)
        del self.__location_to_sku[location]
    
    def get_sku_at_location(self, location: tuple) -> Optional[SKU]:
        """Get the SKU at a specific location."""
        if location not in self.__location_to_sku:
            return None
        return self.__skus[self.__location_to_sku[location]]
    
    def get_empty_locations(self) -> List[tuple]:
        """Get all empty locations in the warehouse."""
        return [loc for loc in self.__warehouse_locations if loc not in self.__location_to_sku]
    
    def get_full_locations(self) -> List[tuple]:
        """Get all locations that contain SKU instances."""
        return list(self.__location_to_sku.keys())
    
    def get_sku_instances(self, sku_id: int) -> List[tuple]:
        """Get all locations containing instances of a specific SKU."""
        if sku_id not in self.__skus:
            raise ValueError(f"SKU {sku_id} does not exist")
        return self.__sku_instances[sku_id].copy()
    
    def get_all_skus(self) -> Dict[int, SKU]:
        """Get all SKUs in the inventory."""
        return self.__skus.copy()
    
    def get_tasking_weights(self) -> Dict[int, float]:
        """Get the tasking weights for all SKUs."""
        return {sku_id: sku.tasking_weight for sku_id, sku in self.__skus.items()}
    
    def get_sku_instance_count(self, sku_id: int) -> int:
        """Get the number of instances of a specific SKU in the warehouse."""
        if sku_id not in self.__skus:
            raise ValueError(f"SKU {sku_id} does not exist")
        return len(self.__sku_instances[sku_id])
    
    def save_to_file(self, filename: str) -> None:
        """Save the current inventory state to a file."""
        data = {
            'skus': {str(sku_id): {
                'appearance_weight': sku.appearance_weight,
                'tasking_weight': sku.tasking_weight
            } for sku_id, sku in self.__skus.items()},
            'sku_instances': {str(sku_id): locations 
                            for sku_id, locations in self.__sku_instances.items()},
            'location_to_sku': {str(loc): sku_id 
                              for loc, sku_id in self.__location_to_sku.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load_from_file(self, filename: str) -> None:
        """Load inventory state from a file."""
        with open(filename, 'r') as f:
            data = json.load(f)
            
        self.__skus.clear()
        self.__sku_instances.clear()
        self.__location_to_sku.clear()
        
        # Load SKUs
        for sku_id_str, sku_data in data['skus'].items():
            sku_id = int(sku_id_str)
            self.__skus[sku_id] = SKU(
                sku_id,
                sku_data['appearance_weight'],
                sku_data['tasking_weight']
            )
            self.__sku_instances[sku_id] = []
        
        # Load SKU instances
        for sku_id_str, locations in data['sku_instances'].items():
            sku_id = int(sku_id_str)
            for location in locations:
                loc_tuple = tuple(location)
                self.__sku_instances[sku_id].append(loc_tuple)
                self.__location_to_sku[loc_tuple] = sku_id