import numpy as np
import random
from typing import Set, Tuple
from .graph import Graph
from .agent import AgentLoader
from .inventory_manager.inventory import Inventory
from .utils import *
from .analysis.statistics import Stats
    
def CRG(S: Stats, t: int, J: Set[Tuple], G: Graph, Rs: AgentLoader, N: int, inbound_to_outbound: float, 
        last_task_id: int, max_task_number: int, inventory: Inventory,
        strategy: str = "uninformed_uniform", deadline_generation_method: str = "constant",
        deadline_offset: float = 30) -> Tuple[Set[Tuple], int]:
    """
    Case Request Generator that creates new tasks based on the current inventory state.
    Each task is defined as (task_id, S_n, D_n, deadline) where:
    - S_n is the frozenset of possible start locations
    - D_n is the frozenset of possible destination locations
    - deadline is an integer (time by which the task should be completed)
    
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
        deadline_generation_method: Method for generating deadlines ("constant" or other)
    
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

    # Deadline generation
    def get_deadline(current_time: int) -> int:
        if deadline_generation_method == "constant":
            return current_time + deadline_offset
        elif deadline_generation_method == "normal":
            # Normal distribution, mean 30, stddev 5
            deadline_offset = np.random.normal(loc=deadline_offset, scale=3)
            return current_time + int(round(deadline_offset))
        elif deadline_generation_method == "bimodal":
            # Bimodal: 10% chance of N(20, 3), 90% chance of N(50, 3)
            if np.random.rand() < 0.1:
                deadline_offset = np.random.normal(loc=deadline_offset - 10, scale=3)
            else:
                deadline_offset = np.random.normal(loc=deadline_offset + 10, scale=3)
            return current_time + int(round(deadline_offset))
        elif deadline_generation_method == "none":
            # Generate tasks with an effective deadline of inf
            return 9999999
        else:
            # Default fallback
            return current_time + 30

    if strategy == "uninformed_uniform":
        for task in tasks_to_generate:
            if task == 1:  # Inbound task
                # Start locations are all driveway nodes
                start_locations = frozenset(G.get_station_locations())
                
                # Goal locations are all empty aisle locations
                goal_locations = frozenset(loc for loc in G.get_aisle_locations())
                
                if not goal_locations:
                    continue
                
                deadline = get_deadline(t)
                J_new.add((last_task_id + 1, start_locations, goal_locations, deadline))
                S.add_task_release(last_task_id + 1, t)
                S.add_task_deadline(last_task_id + 1, deadline)
                last_task_id += 1
                
            elif task == 0:  # Outbound task
                # Start locations are all aisle locations containing SKUs
                start_locations = frozenset(inventory.get_full_locations())
                
                # Goal locations are all driveway nodes
                goal_locations = frozenset(G.get_station_locations())
                
                if not start_locations:
                    continue
                
                deadline = get_deadline(t)
                J_new.add((last_task_id + 1, start_locations, goal_locations, deadline))
                S.add_task_release(last_task_id + 1, t)
                S.add_task_deadline(last_task_id + 1, deadline)
                last_task_id += 1
    
    elif strategy == "informed_uniform":
        outbound_tasks = []
        inbound_tasks = []
        # Get tasking weights for all SKUs
        tasking_weights = inventory.get_tasking_weights()
        total_weight = sum(tasking_weights.values())
        weights = [w / total_weight for w in tasking_weights.values()]
        sku_ids = list(tasking_weights.keys())

        for task in tasks_to_generate:
            # Select SKU based on tasking weights
            sku_id = np.random.choice(sku_ids, p=weights)
            
            if task == 1:  # Inbound task
                # Choose random start location from empty driveway locations
                available_start_locations = set(G.driveway.get_empty_locations())
                if not available_start_locations:
                    continue
                chosen_start_location = random.choice(list(available_start_locations))
                G.driveway.add_sku_instance(sku_id, chosen_start_location)

                # Goal locations are all empty aisle locations
                available_goal_locations = set(G.warehouse.get_empty_locations())
                if not available_goal_locations:
                    continue
                
                deadline = get_deadline(t)
                J_new.add((last_task_id + 1, frozenset([chosen_start_location]), frozenset(available_goal_locations), deadline))
                S.add_task_release(last_task_id + 1, t)
                S.add_task_deadline(last_task_id + 1, deadline)
                last_task_id += 1
                inbound_tasks.append(last_task_id)
                
            elif task == 0:  # Outbound task
                # Start locations are all locations containing the selected SKU (excluding already assigned locations)
                available_start_locations = set(G.warehouse.get_sku_instances(sku_id))
                
                # Goal locations are all driveway nodes (excluding already assigned locations)
                available_goal_locations = set(G.driveway.get_empty_locations())
                
                if not available_start_locations or not available_goal_locations:
                    print(f"No available start or goal locations for SKU {sku_id}: Outbound task")
                    print(f"Number of available start locations: {len(available_start_locations)}")
                    print(f"Number of available goal locations: {len(available_goal_locations)}")
                    continue
                
                deadline = get_deadline(t)
                J_new.add((last_task_id + 1, frozenset(available_start_locations), frozenset(available_goal_locations), deadline))
                S.add_task_release(last_task_id + 1, t)
                S.add_task_deadline(last_task_id + 1, deadline)
                last_task_id += 1
                outbound_tasks.append(last_task_id)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'uninformed_uniform' or 'informed_uniform'")
    
    # print(f"Outbound tasks: {outbound_tasks}")
    # print(f"Inbound tasks: {inbound_tasks}")
    # print(f"Number of outbound tasks: {len(outbound_tasks)}")
    # print(f"Number of inbound tasks: {len(inbound_tasks)}")
    return J_new, last_task_id, outbound_tasks, inbound_tasks
    