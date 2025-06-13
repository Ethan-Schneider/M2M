import numpy as np
from typing import Set, Tuple
from .graph import Graph
from .inventory_manager.inventory import Inventory
from .utils import *
from .analysis.statistics import Stats
    
def CRG(S: Stats, t: int, J: Set[Tuple], G: Graph, N: int, inbound_to_outbound: float, 
        last_task_id: int, max_task_number: int, inventory: Inventory,
        strategy: str = "uninformed_uniform") -> Tuple[Set[Tuple], int]:
    """
    Case Request Generator that creates new tasks based on the current inventory state.
    
    Args:
        S: Statistics object for tracking metrics
        t: Current timestep
        J: Current set of tasks
        G: Graph representing the warehouse
        N: Number of tasks to generate
        inbound_to_outbound: Ratio of inbound to outbound tasks
        last_task_id: ID of the last generated task
        max_task_number: Maximum number of tasks allowed
        inventory: Inventory system
        strategy: Task generation strategy ("uninformed_uniform" or "informed_uniform")
    
    Returns:
        Tuple containing:
        - Set of new tasks (task_id, start_location, goal_location)
        - Updated last_task_id
    """
    
    #TODO: Add logic for skipping task generation is no more tasks can be generated on the map (e.g. every location has some task assigned to it, 
    # all warehouse locations are full of items so no more inbound tasks can be generated, etc.)
    
    # BUG: This causes error with tasks having the same start or goal location bc it is allowing so many tasks to be generated that it is less likely that the router will have to deal with a same location bug
    # if len(J) > max_task_number:
    #     return J, last_task_id
    
    inbound_probability = inbound_to_outbound / (inbound_to_outbound + 1)
    outbound_probability = 1 - inbound_probability
    tasks_to_generate = np.random.choice([0, 1], size=int(N), p=[outbound_probability, inbound_probability])
    J_new = set()
    
    # Get current task locations to avoid conflicts
    current_task_locations = set()
    for task in J:
        current_task_locations.add(task[1])  # start location
        current_task_locations.add(task[2])  # goal location
    print(strategy)
    if strategy == "uninformed_uniform":
        for task in tasks_to_generate:
            if task == 1:  # Inbound task
                # Pickup from station, deliver to empty aisle location
                pickup_location = G.get_station_locations()[np.random.choice(len(G.get_station_locations()), 1)[0]]
                empty_locations = [loc for loc in G.get_aisle_locations() 
                                 if loc not in current_task_locations]
                if not empty_locations:
                    continue
                delivery_location = empty_locations[np.random.choice(len(empty_locations), 1)[0]]
                
                J_new.add((last_task_id + 1, pickup_location, delivery_location))
                last_task_id += 1
                
            elif task == 0:  # Outbound task
                # Pickup from aisle with items, deliver to station
                full_locations = inventory.get_full_locations()
                available_locations = [loc for loc in full_locations 
                                     if loc not in current_task_locations]
                if not available_locations:
                    continue
                    
                pickup_location = available_locations[np.random.choice(len(available_locations), 1)[0]]
                station_locations = [loc for loc in G.get_station_locations() 
                                   if loc not in current_task_locations]
                if not station_locations:
                    continue
                delivery_location = station_locations[np.random.choice(len(station_locations), 1)[0]]
                
                J_new.add((last_task_id + 1, pickup_location, delivery_location))
                last_task_id += 1
    
    elif strategy == "informed_uniform":
        # Get tasking weights for all SKUs
        tasking_weights = inventory.get_tasking_weights()
        total_weight = sum(tasking_weights.values())
        weights = [w / total_weight for w in tasking_weights.values()]
        sku_ids = list(tasking_weights.keys())
        
        for task in tasks_to_generate:
            # Select SKU based on tasking weights
            sku_id = np.random.choice(sku_ids, p=weights)
            
            if task == 1:  # Inbound task
                # Pickup from station, deliver to empty aisle location
                pickup_location = G.get_station_locations()[np.random.choice(len(G.get_station_locations()), 1)[0]]
                empty_locations = [loc for loc in G.get_aisle_locations() 
                                 if loc not in current_task_locations]
                if not empty_locations:
                    continue
                delivery_location = empty_locations[np.random.choice(len(empty_locations), 1)[0]]
                
                J_new.add((last_task_id + 1, pickup_location, delivery_location))
                last_task_id += 1
                
            elif task == 0:  # Outbound task
                # Pickup from aisle location containing the selected SKU
                sku_locations = inventory.get_sku_instances(sku_id)
                available_locations = [loc for loc in sku_locations 
                                     if loc not in current_task_locations]
                if not available_locations:
                    continue
                    
                pickup_location = available_locations[np.random.choice(len(available_locations), 1)[0]]
                station_locations = [loc for loc in G.get_station_locations() 
                                   if loc not in current_task_locations]
                if not station_locations:
                    continue
                delivery_location = station_locations[np.random.choice(len(station_locations), 1)[0]]
                
                J_new.add((last_task_id + 1, pickup_location, delivery_location))
                last_task_id += 1
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'uninformed_uniform' or 'informed_uniform'")
    
    return J_new, last_task_id
    