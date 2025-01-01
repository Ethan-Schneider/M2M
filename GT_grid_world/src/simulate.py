from .analysis.statistics import Stats
from .graph import Graph
from .utils import *

def simulate(S : Stats, G : Graph, Rs : list, robot_sequences : list, Ra : list, J : set, to_delivery : set, to_pickup : set, free_agents : set):

    robot_new_states = []
    # Update state of robots
    for robot in Rs:
        # If robot sequence is stationary, leave the robot in place (wait action)
        # BUG: crashes when router fails to find a solution 
        if len(robot_sequences[robot[0]]) == 0:
            robot_new_states.append(robot[-1])
            continue
        # If robot does have a sequence of actions, pop next state and update
        else:
            old_state = robot[-1]
            new_state = robot_sequences[robot[0]].pop(0)
            Rs[robot[0]] = (robot[0], new_state)
            robot_new_states.append(new_state)
            
            # Update Graph occupied states
            G.set_occupied(old_state, False)
            G.set_occupied(new_state, True)
            
            if robot[0] in to_pickup:
                S.update_actual_pickup_distance(get_assigned_task_id(Ra, robot[0]), euclidian_distance(old_state, new_state))
                S.update_actual_pickup_duration(get_assigned_task_id(Ra, robot[0]), euclidian_distance(old_state, new_state))
            elif robot[0] in to_delivery:
                S.update_actual_distance(get_assigned_task_id(Ra, robot[0]), euclidian_distance(old_state, new_state))
                S.update_actual_duration(get_assigned_task_id(Ra, robot[0]), euclidian_distance(old_state, new_state))
            else:
                pass
    
    S.append_number_of_collisions(len(Ra) - len(set(Ra)))
    
    # Update Statistics for the total paths taken
    S.add_paths(robot_new_states)
    
    # Update Statistics for Asile and Driveway Occupancy
    S.add_aisle_occupancy(G.get_aisle_occupancy())
    S.add_driveway_occupancy(G.get_driveway_occupancy())
    # S.add_aisle_occupancy(G.get_aisle_occupied())
    # S.add_driveway_occupancy(G.get_driveway_occupied())
    
    # Update free_agents, to_pickup, and to_delivery
    # TODO: Update warehouse and driveway inventory when a robot has reached goal location (i.e. for both for loops below) 

    robot_ids = []
    # Update to_pickup -> to_delivery
    for robot in to_pickup:
        task_id = -1
        # Find task_id assignment
        for assignment in Ra:
            if assignment[1] == robot:
                task_id = assignment[0]
                break
        start_loc = (-1, -1)
        # Find start location
        for task in J:
            if task_id == task[0]:
                start_loc = task[1]
                break
        # Find robot.state
        robot_state = Rs[robot][1]
        # Check if robot.state == start_location, if so remove robot from to_pickup and add to_delivery
        if robot_state == start_loc:
            robot_ids.append(robot)
            
            # Update Statistics
            S.add_completed_to_pickup_task_id(task_id)
            
    for id in robot_ids:
        to_pickup.remove(id)
        to_delivery.add(id)  
            
            
    robot_ids = []
    # Update to_delivery -> free_agents
    for robot in to_delivery:
        task_id = -1
        #Find task_id assignment
        current_allocation = -1
        for assignment in Ra:
            if assignment[1] == robot:
                current_allocation = assignment
                task_id = assignment[0]
                break
        goal_loc = (-1, -1)
        curent_task = (-1, -1)
        # Find goal location
        for task in J:
            if task_id == task[0]:
                current_task = task
                goal_loc = task[2]
                break
        # Find robot.state
        robot_state = Rs[robot][1]
        # Check if robot.state == goal_location, if so remove robot from to_delivery and add to free_agents, then remove task from J
        if robot_state == goal_loc:
            robot_ids.append(robot)
            J.remove(current_task)
            Ra.remove(current_allocation)
            # Update Statistics For Completed Tasks
            S.add_completed_task_id(current_task[0])
            
    for id in robot_ids:
        to_delivery.remove(id)
        free_agents.add(id)
            
    return Rs, Ra, J, to_pickup, to_delivery, free_agents