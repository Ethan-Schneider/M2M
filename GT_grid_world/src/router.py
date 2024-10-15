from src.path_finding_algorithms.external_algorithms.EECBS import eecbs
from src.path_finding_algorithms.external_algorithms.PBS import pbs

def pathPlan(G, Rs : list, Ra : list, J : set, path_planning_strategy : str, to_pickup : set, to_delivery : set, free_agents : set) -> list:
    if path_planning_strategy == "ecbs":
        states = [robot[1] for robot in Rs]

        goal_locations = []
        for robot in Rs:
            robot_id = robot[0]
            
            # If robot is going to pickup, set goal location to the task's start location
            if robot_id in to_pickup:
                # Get assigned task_id
                task_id = -1
                for assignment in Ra:
                    if assignment[1] == robot_id:
                        task_id = assignment[0]
                # Get assigned task's start location
                start_loc = (-1, -1)
                for task in J:
                    if task[0] == task_id:
                        start_loc = task[1]
                goal_locations.append((robot_id, start_loc))
                
            # If robot is going to delivery, set goal location to the task's goal location
            elif robot_id in to_delivery:
                # Get assigned task_id
                task_id = -1
                for assignment in Ra:
                    if assignment[1] == robot_id:
                        task_id = assignment[0]
                        
                # Get assigned task's goal location
                goal_loc = (-1, -1)
                for task in J:
                    if task[0] == task_id:
                        goal_loc = task[2]
                goal_locations.append((robot_id, goal_loc))
                
            # If robot is a free_agent, set goal location to current state
            else:
                goal_locations.append((robot_id, robot[1]))
                
        goal_locations.sort()
        goal_locations = [x[1] for x in goal_locations]
        sequences = eecbs.test_cpp_func("GT_grid_world/src/maps/symbotic_small", len(Rs), 60, 1.2, states, goal_locations)
        
        # Remove first item in sequences, as they are the robot's current location
        for sequence in sequences:
            sequence.pop(0)
        
        return sequences
    elif path_planning_strategy == "pbs":
        states = [robot[1] for robot in Rs]

        goal_locations = []
        for robot in Rs:
            robot_id = robot[0]
            
            # If robot is going to pickup, set goal location to the task's start location
            if robot_id in to_pickup:
                # Get assigned task_id
                task_id = -1
                for assignment in Ra:
                    if assignment[1] == robot_id:
                        task_id = assignment[0]
                # Get assigned task's start location
                start_loc = (-1, -1)
                for task in J:
                    if task[0] == task_id:
                        start_loc = task[1]
                goal_locations.append((robot_id, start_loc))
                
            # If robot is going to delivery, set goal location to the task's goal location
            elif robot_id in to_delivery:
                # Get assigned task_id
                task_id = -1
                for assignment in Ra:
                    if assignment[1] == robot_id:
                        task_id = assignment[0]
                        
                # Get assigned task's goal location
                goal_loc = (-1, -1)
                for task in J:
                    if task[0] == task_id:
                        goal_loc = task[2]
                goal_locations.append((robot_id, goal_loc))
                
            # If robot is a free_agent, set goal location to current state
            else:
                goal_locations.append((robot_id, robot[1]))
                
        goal_locations.sort()
        goal_locations = [x[1] for x in goal_locations]
        sequences = pbs.test_cpp_func("GT_grid_world/src/maps/symbotic_small", len(Rs), 60, 1.2, states, goal_locations)
        
        # Remove first item in sequences, as they are the robot's current location
        for sequence in sequences:
            sequence.pop(0)
        
        return sequences