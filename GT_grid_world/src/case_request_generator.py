import numpy as np
from .graph import Graph
from .inventory_manager.item import ItemCategory
from .utils import *
from .analysis.statistics import Stats
    
def CRG(S : Stats, t : int, J: set, G : Graph, N: int, inbound_to_outbound: float, last_task_id: int, max_task_number : int, strategy: str = "uniform", seed : int = 0) -> tuple[set, int]:
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
                # For inbound tasks, pickup from station and deliver to aisle
                pickup_location = np.random.choice(G.get_station_locations())
                delivery_location = np.random.choice(G.get_aisle_locations())
                J_new.add((last_task_id + 1, pickup_location, delivery_location))
                last_task_id += 1
            #Generate outbound task
            elif task == 0:
                # For outbound tasks, pickup from aisle and deliver to station
                pickup_location = np.random.choice(G.get_aisle_locations())
                delivery_location = np.random.choice(G.get_station_locations())
                J_new.add((last_task_id + 1, pickup_location, delivery_location))
                last_task_id += 1
    elif strategy == "informed_uniform":
        # Generate items for i/o tasks
        items = []
        for _ in tasks_to_generate:
            items.append(ItemCategory(np.random.choice(np.arange(2, len(ItemCategory)+1))).name)
        
        # Generate list of locations involved in a current task
        current_task_locations = set()
        for task in J:
            current_task_locations.add(task[1])
            current_task_locations.add(task[2])
            
        # Construct list of current task locations in the aisle and stations
        aisle_locations = []
        station_locations = []
        # Iterate over every currently existing task
        for task in J:
            # Get the start and goal locations for the task
            start_loc = get_task_start_location(J, task[0])
            goal_loc = get_task_goal_location(J, task[0])
            
            # Check which position is in the aisle and station, and append them accordingly
            if start_loc in G.get_aisle_locations():
                aisle_locations.append(start_loc)
                station_locations.append(goal_loc)
            else:
                aisle_locations.append(goal_loc)
                station_locations.append(start_loc)
        
        for i, item in enumerate(items):
            if tasks_to_generate[i] == 0:  # Outbound task
                # Generate locations for item pickup from aisle that are not part of a task yet
                locations_for_item = list(set(G.get_aisle_locations()) - current_task_locations)
                if not locations_for_item:
                    continue
                
                while True:
                    # Uniformly choose a pickup location from aisle
                    pickup_location = locations_for_item[np.random.choice(len(locations_for_item), 1)[0]]
                    if pickup_location not in aisle_locations:
                        break
                
                # Generate locations for item dropoff at station that are not part of a task
                locations_for_dropoff = list(set(G.get_station_locations()) - current_task_locations)
                if not locations_for_dropoff:
                    continue
                
                while True:
                    # Uniformly choose a dropoff location from stations
                    dropoff_location = locations_for_dropoff[np.random.choice(len(locations_for_dropoff), 1)[0]]
                    if dropoff_location not in station_locations:
                        break
                
                J_new.add((last_task_id, pickup_location, dropoff_location))
                S.add_task_release(last_task_id, t)
                last_task_id += 1
                
                current_task_locations.add(pickup_location)
                current_task_locations.add(dropoff_location)
                
            elif tasks_to_generate[i] == 1:  # Inbound task
                # Generate locations for item pickup from station that are not part of a task yet
                locations_for_item = list(set(G.get_station_locations()) - current_task_locations)
                if not locations_for_item:
                    continue

                while True:
                    # Uniformly choose a pickup location from stations
                    pickup_location = locations_for_item[np.random.choice(len(locations_for_item), 1)[0]]
                    if pickup_location not in station_locations:
                        break
                
                # Generate locations for item dropoff at aisle that are not part of a task
                locations_for_dropoff = list(set(G.get_aisle_locations()) - current_task_locations)
                if not locations_for_dropoff:
                    continue
                
                while True:
                    # Uniformly choose a dropoff location from aisles
                    dropoff_location = locations_for_dropoff[np.random.choice(len(locations_for_dropoff), 1)[0]]
                    if dropoff_location not in aisle_locations:
                        break
                
                J_new.add((last_task_id, pickup_location, dropoff_location))
                S.add_task_release(last_task_id, t)
                last_task_id += 1
                
                current_task_locations.add(pickup_location)
                current_task_locations.add(dropoff_location)
            else:
                raise Exception("Error: Item has been designated neither an inbound or outbound task.")
        
        
    else:
        raise Exception("Unknown strategy, " + strategy + ", given, please select one of the chosen task generation strategies: \n - uninformed_uniform \n - informed_uniform")
    # print(f"current task locations: {current_task_locations}")
    return J_new, last_task_id
    