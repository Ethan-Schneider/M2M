import numpy as np
from .graph import Graph
from .inventory_manager.item import ItemCategory
from .utils import *
    
def CRG(J: set, G : Graph, N: int, inbound_to_outbound: float, last_task_id: int, max_task_number : int, strategy: str = "uniform", seed : int = 0) -> tuple[set, int]:
    """_summary_

    Args:
        J (set): _description_
        G (Graph): _description_
        N (int): _description_
        inbound_to_outbound (float): _description_
        last_task_id (int): _description_
        max_task_number (int): _description_
        strategy (str, optional): _description_. Defaults to "uniform".

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        tuple[set, int]: _description_
    """
    
    #TODO: Add logic for skipping task generation is no more tasks can be generated on the map (e.g. every location has some task assigned to it, 
    # all warehouse locations are full of items so no more inbound tasks can be generated, etc.)
    
    # BUG: This causes error with tasks having the same start or goal location bc it is allowing so many tasks to be generated that it is less likely that the router will have to deal with a same location bug
    # if len(J) > max_task_number:
    #     return J, last_task_id
    
    inbound_probability = inbound_to_outbound/(inbound_to_outbound + 1)
    outbound_probability = 1 - inbound_probability
    tasks_to_generate = np.random.choice([0, 1], size=int(N), p=[outbound_probability, inbound_probability])
    J_new = set([]) 
    
    #Generate tasks uniformly throughout the warehouse without inventory information
    if strategy == "uninformed_uniform":
        for task in tasks_to_generate:
            #Generate inbound task
            if task == 1:
                J_new.add((last_task_id + 1, np.random.choice(np.arange(G.driveway.max_pos)), np.random.choice(np.arange(G.warehouse.max_pos))))
                last_task_id += 1
            #Generate outbound task
            elif task == 0:
                J_new = J_new | set([(last_task_id + 1, np.random.choice(np.arange(G.warehouse.max_pos)), np.random.choice(np.arange(G.driveway.max_pos)))])
                last_task_id += 1
    elif strategy == "informed_uniform":
        # TODO: Implement informed_uniform strategy: where the algorithm will uniformly sample an item from ItemCategory for inbound or outbound, then uniformly sample from 
        # the warehouse for that item or for empty spaces to put that item.  

        # Generate items for i/o tasks
        items = []
        for _ in tasks_to_generate:
            items.append(ItemCategory(np.random.choice(np.arange(2, len(ItemCategory)+1))).name)
        
        # Generate list of locations involved in a current task
        current_task_locations = set()
        for task in J:
            current_task_locations.add(task[1])
            current_task_locations.add(task[2])
            
        # print(f"Current Task Locations: {current_task_locations}")

        # Construct list of current task locations in the driveway and aisle
        driveway_locations = []
        aisle_locations = []
        # Iterate over every currently existing task
        for task in J:
            # Get the start and goal locations for the task
            start_loc = get_task_start_location(J, task[0])
            goal_loc = get_task_goal_location(J, task[0])
            
            # Check which position is in the driveway and aisle, and append them accordingly
            if start_loc[0] >= 4:
                driveway_locations.append(start_loc[1])
                aisle_locations.append(goal_loc[1])
            else:
                driveway_locations.append(goal_loc[1])
                aisle_locations.append(start_loc[1])
        
        for i, item in enumerate(items):
            if tasks_to_generate[i] == 0:
                # Generate locations for item pickup that are not part of a task yet
                locations_for_item = list(set(G.warehouse.find(item)) - current_task_locations)
                if not locations_for_item:
                    continue
                
                while True:
                # Uniformly choose a pickup location
                    pickup_location = locations_for_item[np.random.choice(len(locations_for_item), 1)[0]]

                    aisle_loc = pickup_location[1]
                    if np.count_nonzero(aisle_locations == aisle_loc) >= 2:
                        continue
                    else:
                        break
                
                # Generate locations for item dropoff that are not part of a task
                locations_for_dropoff = list(set(G.driveway.findEmpty()) - current_task_locations)
                if not locations_for_dropoff:
                    continue
                
                while True:
                # Uniformly choose a dropoff location

                    dropoff_location = locations_for_dropoff[np.random.choice(len(locations_for_dropoff), 1)[0]]

                    driveway_loc = dropoff_location[1]
                    if np.count_nonzero(driveway_locations == driveway_loc) >= 2:
                        continue
                    else:
                        break
                
                J_new.add((last_task_id, pickup_location, dropoff_location))
                last_task_id += 1
                
                current_task_locations.add(pickup_location)
                current_task_locations.add(dropoff_location)
                
            elif tasks_to_generate[i] == 1:
                # Generate locations for item pickup that are not part of a task yet
                locations_for_item = list(set(G.driveway.findEmpty()) - current_task_locations)
                if not locations_for_item:
                    continue

                while True:
                    # Uniformly choose a pickup location
                    pickup_location = locations_for_item[np.random.choice(len(locations_for_item), 1)[0]]

                    driveway_loc = pickup_location[1]
                    if np.count_nonzero(driveway_locations == driveway_loc) >= 2:
                        continue
                    else:
                        break
                
                # Generate locations for item dropoff that are not part of a task
                locations_for_dropoff = list(set(G.warehouse.findEmpty()) - current_task_locations)
                if not locations_for_dropoff:
                    continue
                
                while True:
                    # Uniformly choose a dropoff location
                    dropoff_location = locations_for_dropoff[np.random.choice(len(locations_for_dropoff), 1)[0]]

                    aisle_loc = dropoff_location[1]
                    if np.count_nonzero(aisle_locations == aisle_loc) >= 2:
                        continue
                    else:
                        break
                
                J_new.add((last_task_id, pickup_location, dropoff_location))
                last_task_id += 1
                
                current_task_locations.add(pickup_location)
                current_task_locations.add(dropoff_location)
            else:
                raise Exception("Error: Item has been designated neither an inbound or outbound task.")
        
        
    else:
        raise Exception("Unknown strategy, " + strategy + ", given, please select one of the chosen task generation strategies: \n - uninformed_uniform \n - informed_uniform")
    # print(f"current task locations: {current_task_locations}")
    return J_new, last_task_id
    