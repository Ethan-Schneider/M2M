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
    Each task is defined as (task_id, S_n, D_n) where:
    - S_n is the frozenset of possible start locations
    - D_n is the frozenset of possible destination locations
    
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
        - Set of new tasks (task_id, start_locations_frozenset, goal_locations_frozenset)
        - Updated last_task_id
    """
    
    # Check if we've reached the maximum number of tasks
    if len(J) >= max_task_number:
        return set(), last_task_id
    
    inbound_probability = inbound_to_outbound / (inbound_to_outbound + 1)
    outbound_probability = 1 - inbound_probability
    tasks_to_generate = np.random.choice([0, 1], size=int(N), p=[outbound_probability, inbound_probability])
    J_new = set()
    
    # Get current task locations to avoid conflicts
    current_task_locations = set()
    for task in J:
        # For each task, add all possible start and goal locations to current_task_locations
        current_task_locations.update(task[1])  # start locations set
        current_task_locations.update(task[2])  # goal locations set
    
    print(strategy)
    if strategy == "uninformed_uniform":
        for task in tasks_to_generate:
            if task == 1:  # Inbound task
                # Start locations are all driveway nodes
                start_locations = frozenset(G.get_station_locations())
                
                # Goal locations are all empty aisle locations
                goal_locations = frozenset(loc for loc in G.get_aisle_locations() 
                                        if loc not in current_task_locations)
                
                if not goal_locations:
                    continue
                
                J_new.add((last_task_id + 1, start_locations, goal_locations))
                last_task_id += 1
                
            elif task == 0:  # Outbound task
                # Start locations are all aisle locations containing SKUs
                start_locations = frozenset(inventory.get_full_locations())
                
                # Goal locations are all driveway nodes
                goal_locations = frozenset(G.get_station_locations())
                
                if not start_locations:
                    continue
                
                J_new.add((last_task_id + 1, start_locations, goal_locations))
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
                # Start locations are all driveway nodes
                start_locations = frozenset(G.driveway.get_empty_locations())
                
                # Goal locations are all empty aisle locations
                goal_locations = frozenset(G.warehouse.get_empty_locations())
                
                if not goal_locations:
                    continue
                
                J_new.add((last_task_id + 1, start_locations, goal_locations))
                last_task_id += 1
                
            elif task == 0:  # Outbound task
                # Start locations are all locations containing the selected SKU
                start_locations = frozenset(G.warehouse.get_sku_instances(sku_id))
                
                # Goal locations are all driveway nodes
                goal_locations = frozenset(G.driveway.get_empty_locations())
                if not start_locations:
                    continue
                
                J_new.add((last_task_id + 1, start_locations, goal_locations))
                last_task_id += 1
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'uninformed_uniform' or 'informed_uniform'")
    
    return J_new, last_task_id
    