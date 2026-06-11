import numpy as np
import random
from typing import Set, Tuple, Dict
from .graph import Graph
from .agent import AgentLoader
from .inventory_manager.inventory import Inventory
from .utils import *
from .analysis.statistics import Stats
from .logging_config import get_logger

_log = get_logger("crg")

# Task type codes used throughout M2M:
#   0 = outbound (warehouse -> driveway)
#   1 = inbound  (driveway  -> warehouse)
#   2 = shuffle  (warehouse -> warehouse, shelf-to-shelf rearrangement)
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1
TASK_TYPE_SHUFFLE = 2


def CRG(S: Stats, t: int, J: Dict[int, Tuple], G: Graph, Rs: AgentLoader, N: int, inbound_to_outbound: float, 
        last_task_id: int, max_task_number: int, inventory: Inventory,
        strategy: str = "uninformed_uniform", deadline_generation_method: str = "constant",
        deadline_offset: float = 30, improvement_task_assign_strategy: str = "c_lns",
        initial_inventory : float = 25.0,
        shuffle_percentage: float = 0.0) -> Tuple[Dict[int, Tuple], int]:
    """
    Case Request Generator that creates new tasks based on the current inventory state.
    Each task is defined as (task_id, S_n, D_n, deadline, sku_id, type) where:
    - S_n is the frozenset of possible start locations
    - D_n is the frozenset of possible destination locations
    - deadline is an integer (time by which the task should be completed)
    - sku_id identifies which SKU the task moves
    - type is 0 (outbound), 1 (inbound), or 2 (shuffle / shelf-to-shelf)

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
        strategy: Task generation strategy
        deadline_generation_method: Method for generating deadlines
        shuffle_percentage: Per-task probability of generating a shuffle (type=2) task
            instead of an IB/OB task. 0.0 (default) disables shuffle generation. The
            proper ratio balancer for rearrangement lives in roadmap section 3.2;
            this flag is the 1.4-skeleton on/off switch.

    Returns:
        Tuple containing the updated J, last_task_id, and lists of newly created
        outbound and inbound task ids. Shuffle task ids are not returned in either
        list (the ratio balancer in 3.2 will take ownership of that bookkeeping).
    """

    outbound_tasks: list = []
    inbound_tasks: list = []

    if len(J) >= max_task_number:
        return J, last_task_id, outbound_tasks, inbound_tasks
    
    inbound_probability = inbound_to_outbound / (inbound_to_outbound + 1)
    outbound_probability = 1 - inbound_probability
    tasks_to_generate = np.random.choice([0, 1], size=int(N), p=[outbound_probability, inbound_probability])

    shuffle_percentage = float(np.clip(shuffle_percentage, 0.0, 1.0))

    if strategy == "uninformed_uniform":
        for task in tasks_to_generate:
            if shuffle_percentage > 0.0 and random.random() < shuffle_percentage:
                # Shuffle (type=2) doesn't have a meaningful "uninformed" SKU
                # selection because it has to pick a SKU that already exists in
                # the warehouse. Pick one uniformly at random from SKUs that
                # currently have at least 2 instances so a non-degenerate move
                # exists.
                J, success, last_task_id = _generate_shuffle_task(
                    deadline_generation_method, deadline_offset,
                    sku_id=None, t=t, last_task_id=last_task_id,
                    G=G, J=J, S=S,
                )
                if success:
                    continue
                # Fall through to IB/OB if shuffle generation failed (e.g. no
                # SKU has >=2 instances). Don't double-roll the IB/OB die --
                # the caller already chose between IB/OB for this slot.
            if task == 1:  # Inbound task
                # Start locations are all driveway nodes
                start_locations = frozenset(G.get_station_locations())
                
                # Goal locations are all empty aisle locations
                goal_locations = frozenset(loc for loc in G.get_aisle_locations())
                
                if not goal_locations:
                    continue
                
                deadline = get_deadline(t, deadline_generation_method, deadline_offset)
                J[last_task_id + 1] = (start_locations, goal_locations, deadline, 0, 1)
                S.add_task_release(last_task_id + 1, t)
                S.add_task_deadline(last_task_id + 1, deadline)
                last_task_id += 1
                inbound_tasks.append(last_task_id)
                
            elif task == 0:  # Outbound task
                # Start locations are all aisle locations containing SKUs
                start_locations = frozenset(inventory.get_full_locations())
                
                # Goal locations are all driveway nodes
                goal_locations = frozenset(G.get_station_locations())
                
                if not start_locations:
                    continue
                
                deadline = get_deadline(t, deadline_generation_method, deadline_offset)
                J[last_task_id + 1] = (start_locations, goal_locations, deadline, 0, 0)
                S.add_task_release(last_task_id + 1, t)
                S.add_task_deadline(last_task_id + 1, deadline)
                last_task_id += 1
                outbound_tasks.append(last_task_id)
    
    elif strategy == "informed_uniform":
        # Get tasking weights for all SKUs
        tasking_weights = inventory.get_tasking_weights()
        total_weight = sum(tasking_weights.values())
        weights = [w / total_weight for w in tasking_weights.values()]
        sku_ids = list(tasking_weights.keys())
        
        num_skus = S.get_num_skus()
        allocated_skus = {sku_id: 0 for sku_id in sku_ids}

        for task_id, task_info in J.items():
            sku_id = task_info[3]
            allocated_skus[sku_id] += 1

        for task in tasks_to_generate:
            attempts = 0
            while True:
                J, success, last_task_id = generate_task(deadline_generation_method, deadline_offset, sku_ids, weights, task, t, last_task_id, allocated_skus, G, J, S,
                                                          shuffle_percentage=shuffle_percentage)
                if success:
                    break
                attempts += 1
                if attempts > 50:
                    break
                
    elif strategy == "feedback_control":
        # Get tasking weights for all SKUs
        tasking_weights = inventory.get_tasking_weights()
        total_weight = sum(tasking_weights.values())
        weights = [w / total_weight for w in tasking_weights.values()]
        sku_ids = list(tasking_weights.keys())
        
        allocated_skus = {sku_id: 0 for sku_id in sku_ids}
        
        k = 0
        if initial_inventory <= 50.0:
            k = 1.0
        else:
            k = 0.05
            
        p_min = 0.15
        p_max = 0.85
        
        current_inventory = G.warehouse.get_fullness_percentage()*100
            
        p_in = np.clip(0.5 +  k*(initial_inventory - current_inventory), p_min, p_max)
        p_out = 1 - p_in
        
        tasks_to_generate = np.random.choice([0, 1], size=int(N), p=[p_out, p_in])
        
        # print(f"P In: {p_in} P Out: {p_out}")
        # print(f"Current Inventory: {current_inventory}")
        # print(f"tasks to generate: {tasks_to_generate}")
        
        for task in tasks_to_generate:
            attempts = 0
            while True:
                J, success, last_task_id = generate_task(deadline_generation_method, deadline_offset, sku_ids, weights, task, t, last_task_id, allocated_skus, G, J, S,
                                                          shuffle_percentage=shuffle_percentage)
                if success:
                    break
                attempts += 1
                if attempts > 50:
                    break
            if len(J) >= max_task_number:
                break
        
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'uninformed_uniform' or 'informed_uniform'")
    
    # print(f"Outbound tasks: {outbound_tasks}")
    # print(f"Inbound tasks: {inbound_tasks}")
    # print(f"Number of outbound tasks: {len(outbound_tasks)}")
    # print(f"Number of inbound tasks: {len(inbound_tasks)}")
    return J, last_task_id, outbound_tasks, inbound_tasks


def generate_task(deadline_generation_method : str, deadline_offset : float, sku_ids, weights, task, t : int, last_task_id : int, allocated_skus : dict, G : Graph, J : set, S : Stats,
                  shuffle_percentage: float = 0.0) -> bool:
    # Select SKU based on tasking weights. The chosen SKU is reused for both the
    # IB/OB path and the shuffle short-circuit so the SKU mix matches the
    # caller's tasking weights instead of being uniform across SKUs.
    sku_id = int(np.random.choice(sku_ids, p=weights))

    if shuffle_percentage > 0.0 and random.random() < shuffle_percentage:
        J, success, last_task_id = _generate_shuffle_task(
            deadline_generation_method, deadline_offset,
            sku_id=sku_id, t=t, last_task_id=last_task_id,
            G=G, J=J, S=S,
        )
        if success:
            return J, True, last_task_id
        # Fall through to IB/OB on failure (e.g. <2 instances of this SKU).

    if task == 1:  # Inbound task
        # Choose random start location from empty driveway locations
        available_start_locations = set(G.driveway.get_empty_locations())
        if not available_start_locations:
            return J, False, last_task_id
        
        num_outbound_tasks = len([task for task in list(J.values()) if task[4] == 0])
        num_available_start_locations = len(available_start_locations) - num_outbound_tasks
        
        # if num_available_start_locations == 0:
        #     return J, False, last_task_id
        
        chosen_start_location = random.choice(list(available_start_locations))
        G.driveway.add_sku_instance(sku_id, chosen_start_location)

        # Goal locations are all empty aisle locations
        available_goal_locations = set(G.warehouse.get_empty_locations())
        if not available_goal_locations:
            return J, False, last_task_id
        
        num_inbound_tasks = len([task for task in list(J.values()) if task[4] == 1])
        num_available_goal_locations = len(available_goal_locations) - num_inbound_tasks
        # if num_available_goal_locations == 0:
        #     return J, False, last_task_id
        
        deadline = get_deadline(t, deadline_generation_method, deadline_offset)
        J[last_task_id + 1] = (frozenset([chosen_start_location]), frozenset(available_goal_locations), deadline, sku_id, 1)

        S.add_task_release(last_task_id + 1, t)
        S.add_task_deadline(last_task_id + 1, deadline)
        last_task_id += 1
        
        return J, True, last_task_id
        
    elif task == 0:  # Outbound task
        
        # If there exists a number of outbound tasks for a sku equal to the number of instances of that sku in the warehouse, skip
        if allocated_skus[sku_id] >= G.warehouse.get_sku_instance_count(sku_id):
            return J, False, last_task_id
        
        # Start locations are all locations containing the selected SKU (excluding already assigned locations)
        available_start_locations = set(G.warehouse.get_sku_instances(sku_id))
        
        num_outbound_tasks_with_sku = len([task for task in list(J.values()) if (task[4] == 0 and task[2] == sku_id)])
        num_available_start_locations = len(available_start_locations) - num_outbound_tasks_with_sku
        
        # if num_available_start_locations == 0:
        #     return J, False, last_task_id
        
        # Goal locations are all driveway nodes (excluding already assigned locations)
        available_goal_locations = set(G.driveway.get_empty_locations())
        
        num_outbound_tasks = len([task for task in list(J.values()) if task[4] == 0])
        num_available_goal_locations = len(available_goal_locations) - num_outbound_tasks
        
        # if num_available_goal_locations == 0:
        #     return J, False, last_task_id
        
        
        if not available_start_locations or not available_goal_locations:
            print(f"No available start or goal locations for SKU {sku_id}: Outbound task")
            print(f"Number of available start locations: {len(available_start_locations)}")
            print(f"Number of available goal locations: {len(available_goal_locations)}")
            return J, False, last_task_id
        
        deadline = get_deadline(t, deadline_generation_method, deadline_offset)
        J[last_task_id + 1] = (frozenset(available_start_locations), frozenset(available_goal_locations), deadline, sku_id, 0)

        S.add_task_release(last_task_id + 1, t)
        S.add_task_deadline(last_task_id + 1, deadline)
        last_task_id += 1
        
        return J, True, last_task_id
        
        
def _generate_shuffle_task(deadline_generation_method: str, deadline_offset: float,
                           sku_id, t: int, last_task_id: int,
                           G: Graph, J: dict, S: Stats):
    """
    Generate a single shuffle (type=2, shelf-to-shelf) task and append it to J.

    Picks a SKU that has at least one warehouse instance and at least one
    empty warehouse cell. Both endpoints live in the warehouse: start_locs is
    the frozenset of all current instances of the chosen SKU and goal_locs is
    the frozenset of all empty warehouse cells. The allocator chooses which
    specific (start, goal) pair to commit to.

    The caller may pass ``sku_id=None`` to ask the helper to pick a SKU
    uniformly at random from those with >=1 warehouse instance (used by the
    "uninformed_uniform" strategy). If a specific ``sku_id`` is passed and that
    SKU has no warehouse instances, the call returns success=False so the
    caller can fall back to IB/OB.

    Returns:
        (J, success, last_task_id) -- mirrors generate_task's contract.
    """
    available_goal_locations = set(G.warehouse.get_empty_locations())
    if not available_goal_locations:
        # Warehouse is full. Shelf-to-shelf is impossible until something exits.
        _log.debug("Shuffle skipped at t=%s: no empty warehouse cells", t)
        return J, False, last_task_id

    if sku_id is None:
        # Caller wants us to pick a SKU. Restrict to SKUs with >=1 warehouse
        # instance so the start_locs frozenset is non-empty.
        candidate_skus = [
            s for s in G.warehouse.get_all_skus().keys()
            if G.warehouse.get_sku_instance_count(s) >= 1
        ]
        if not candidate_skus:
            _log.debug("Shuffle skipped at t=%s: warehouse has no SKUs", t)
            return J, False, last_task_id
        sku_id = int(random.choice(candidate_skus))
    else:
        sku_id = int(sku_id)
        if sku_id not in G.warehouse.get_all_skus():
            # Caller specified a SKU the warehouse does not know about. Treat
            # as "no shuffle possible for this slot" rather than raising; CRG
            # callers fall through to IB/OB on False.
            return J, False, last_task_id
        if G.warehouse.get_sku_instance_count(sku_id) < 1:
            return J, False, last_task_id

    available_start_locations = set(G.warehouse.get_sku_instances(sku_id))
    if not available_start_locations:
        return J, False, last_task_id

    # Source must differ from goal. start_locs comes from warehouse-FULL cells,
    # goal_locs comes from warehouse-EMPTY cells, so the two sets are disjoint
    # by construction. Still, guard against the (impossible) overlap so a
    # future change to either set can't silently produce degenerate tasks.
    if available_start_locations & available_goal_locations:
        raise AssertionError(
            "Shuffle start_locs and goal_locs overlap; warehouse occupancy "
            "bookkeeping is inconsistent"
        )

    deadline = get_deadline(t, deadline_generation_method, deadline_offset)
    new_task_id = last_task_id + 1
    J[new_task_id] = (
        frozenset(available_start_locations),
        frozenset(available_goal_locations),
        deadline,
        sku_id,
        TASK_TYPE_SHUFFLE,
    )
    S.add_task_release(new_task_id, t)
    S.add_task_deadline(new_task_id, deadline)
    last_task_id = new_task_id
    _log.debug("Generated shuffle task id=%s sku=%s |start|=%s |goal|=%s deadline=%s",
               new_task_id, sku_id, len(available_start_locations),
               len(available_goal_locations), deadline)
    return J, True, last_task_id


# Deadline generation
def get_deadline(current_time: int, deadline_generation_method : str, deadline_offset : float) -> int:
    if deadline_generation_method == "constant":
        return current_time + deadline_offset
    elif deadline_generation_method == "normal":
        # Normal distribution, mean 30, stddev 5
        deadline = np.random.normal(loc=deadline_offset, scale=3)
        return current_time + int(round(deadline))
    elif deadline_generation_method == "bimodal":
        # Bimodal: 10% chance of N(20, 3), 90% chance of N(50, 3)
        if np.random.rand() < 0.1:
            deadline = np.random.normal(loc=deadline_offset - 10, scale=3)
        else:
            deadline = np.random.normal(loc=deadline_offset + 10, scale=3)
        return current_time + int(round(deadline))
    elif deadline_generation_method == "none":
        # Generate tasks with an effective deadline of inf
        return 9999999
    else:
        # Default fallback
        return current_time + 30
    