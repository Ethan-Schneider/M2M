import numpy as np
from .statistics import Stats
from .graph import Graph
from .utils import get_assigned_task_id, get_robot_state, get_task_start_location, get_task_goal_location, a_star

def TaskAllocation(S : Stats, G : Graph, Rs : list, Ra : list, J : set, task_assignment_strategy : str, to_pickup : set, free_agents : set, hash_map : dict = {}):
    # make a set of all tasks that are not currently allocated
    
    # if statement for which strategy will be used
    
    # ------ closest robot to closest task
    # compute the distance from a task to every robot's current location
    
    if task_assignment_strategy == "closest_robot":
        # TODO: Fix closest_robot stategy: Bug involves index of the cost_tensor not matching the task_id number
        # Algorithm assigns the closest task to the closest agent at every time-step
        # Assigns a single-task robot pairing, NOT a sequence of tasks for each robot
        
        assigned_task_ids = set([x[0] for x in Ra])
        task_ids = set([x[0] for x in J])
        
        unassigned_task_ids = task_ids - assigned_task_ids
        
        assigned_robot_ids = set()
        for assignment in Ra:
            assigned_robot_ids |= set([assignment[-1]])
        unassigned_robot_ids = set(np.arange(0, len(Rs))) - assigned_robot_ids
        
        if len(unassigned_robot_ids) < 0:
            return Ra
        elif len(unassigned_task_ids) < 0:
            return Ra
        else:
            pass
        
        # Init cost_tensor with values
        cost_tensor = []

        # Iterate over all robots and task to generate the cost_tensor
        for robot in Rs:
            array = []
            for task in J:
                # TODO: Replace L1 norm with A* search for distance to task
                array.append(np.linalg.norm(np.asarray(task[1]) - np.asarray(robot[1]), ord=1))
            cost_tensor.append(array)
            
        cost_tensor = np.asarray(cost_tensor)
        
        # Stop assigned tasks from being assigned twice
        for i in assigned_task_ids:
            cost_tensor[:, i] = np.inf
        
        # Stop assgined robots from being assigned twice
        for i in assigned_robot_ids:
            cost_tensor[i, :] = np.inf
            
        # If all robots or all tasks are allocated, return the allocation
        if np.all(cost_tensor == np.inf):
            return Ra, to_pickup, free_agents

        # Iterate for every 
        min_value_indexes = []
        
        for _ in range(len(Rs)):
            min_value_indexes.append(np.unravel_index(np.argmin(cost_tensor), cost_tensor.shape))
            robot = min_value_indexes[-1][0]
            task = min_value_indexes[-1][1]
            
            cost_tensor[robot, :] = np.inf
            cost_tensor[:, task] = np.inf
            Ra.append((task, robot))
            
            if np.all(cost_tensor == np.inf):
                break
            
        for task in Ra:
            robot_id = task[-1]
            if robot_id in free_agents:
                free_agents.remove(robot_id)
                to_pickup.add(robot_id)
            else:
                continue

        return Ra, to_pickup, free_agents
    
    elif task_assignment_strategy == "random":
        assigned_task_ids = set([x[0] for x in Ra])
        task_ids = set([x[0] for x in J])
        
        unassigned_task_ids = task_ids - assigned_task_ids
        
        assigned_robot_ids = set()
        for assignment in Ra:
            assigned_robot_ids |= set([assignment[-1]])
        unassigned_robot_ids = set(np.arange(0, len(Rs))) - assigned_robot_ids
        
        if len(unassigned_robot_ids) <= 0:
            return Ra, to_pickup, free_agents
        elif len(unassigned_task_ids) <= 0:
            return Ra, to_pickup, free_agents
        else:
            pass
        
        for unassigned_robot_id in unassigned_robot_ids:
            task = np.random.choice(list(unassigned_task_ids))
            
            Ra.append((task, unassigned_robot_id))
            unassigned_task_ids.remove(task)
            if len(unassigned_task_ids) == 0:
                break
        
        
        for task in Ra:
            robot_id = task[-1]
            if robot_id in free_agents:
                free_agents.remove(robot_id)
                to_pickup.add(robot_id)
                
                S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
                S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
            else:
                continue

        return Ra, to_pickup, free_agents
    
    elif task_assignment_strategy == "a_star": 
        
        # path = a_star(G, (45, 12), (114, 81))
        # print(len(path))
        
        # Get unassigned tasks and unassigned robots
        assigned_task_ids = set([x[0] for x in Ra])
        task_ids = set([x[0] for x in J])
        
        unassigned_task_ids = task_ids - assigned_task_ids
        
        assigned_robot_ids = set()
        for assignment in Ra:
            assigned_robot_ids |= set([assignment[-1]])
        unassigned_robot_ids = set(np.arange(0, len(Rs))) - assigned_robot_ids
        
        
        if len(unassigned_robot_ids) <= 0:
            return Ra, to_pickup, free_agents, True
        elif len(unassigned_task_ids) <= 0:
            return Ra, to_pickup, free_agents, True
        else:
            pass
        
        # Create cost matrix
        cost_matrix = []
        for unassigned_robot_id in unassigned_robot_ids:
            robot_cost = []
            for unassigned_task_id in unassigned_task_ids:
                # In-case a_star breaks and does not return a solution, try again
                # TODO: Debug a_start implementation for why this occurs so infrequently
                path = a_star(G, get_robot_state(Rs, unassigned_robot_id), get_task_start_location(J, unassigned_task_id))
                
                if not path:
                    robot_cost.append(np.inf)
                else:
                    robot_cost.append(len(path))
            cost_matrix.append(robot_cost)
        
        # Assign tasks from cost matrix
        is_solution = True
        cost_matrix = np.asarray(cost_matrix, dtype=float)    
        try: 
            for unassigned_robot_id in unassigned_robot_ids:
                # try: 
                min_index_row, min_index_col = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                Ra.append((list(unassigned_task_ids)[min_index_col], list(unassigned_robot_ids)[min_index_row]))
                S.add_estimated_pickup_distance(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
                S.add_estimated_pickup_duration(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
                while True:
                    estimated_task_path = a_star(G, get_task_start_location(J, list(unassigned_task_ids)[min_index_col]), get_task_goal_location(J, list(unassigned_task_ids)[min_index_col]))
                    if estimated_task_path:
                        break
                    elif not estimated_task_path:
                        # G.draw_graph()
                        print(get_task_start_location(J, list(unassigned_task_ids)[min_index_col])) 
                        print(get_task_goal_location(J, list(unassigned_task_ids)[min_index_col]))
                        is_solution = False
                        break
                if is_solution:
                    S.add_estimated_distance(list(unassigned_task_ids)[min_index_col], len(estimated_task_path) - 1)
                    S.add_estimated_duration(list(unassigned_task_ids)[min_index_col], len(estimated_task_path) - 1)
                else:
                    S.add_estimated_distance(list(unassigned_task_ids)[min_index_col], np.Infinity)
                    S.add_estimated_duration(list(unassigned_task_ids)[min_index_col], np.Infinity)
                is_solution = True
                
                cost_matrix[:, min_index_col] = np.inf
                cost_matrix[min_index_row, :] = np.inf
                
                unassigned_task_ids.remove(list(unassigned_task_ids)[min_index_col])
                if len(unassigned_task_ids) == 0:
                    break
        except:
            return Ra, to_pickup, free_agents, False
            # except:
            #     print("++++++++++++++++++++++++++++++++Skipping Allocation++++++++++++++++++++++++++++++++")
            #     for task in Ra:
            #         robot_id = task[-1]
            #         if robot_id in free_agents:
            #             free_agents.remove(robot_id)
            #             to_pickup.add(robot_id)
                        
            #             S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
            #             S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
                        
            #             S.add_actual_duration(get_assigned_task_id(Ra, robot_id))
            #             S.add_actual_pickup_duration(get_assigned_task_id(Ra, robot_id))
            #         else:
            #             continue
            #     return Ra, to_pickup, free_agents

        # Update statistics, free agents, and to_pickup
        for task in Ra:
            robot_id = task[-1]
            if robot_id in free_agents:
                free_agents.remove(robot_id)
                to_pickup.add(robot_id)
                
                S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
                S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
                
                S.add_actual_duration(get_assigned_task_id(Ra, robot_id))
                S.add_actual_pickup_duration(get_assigned_task_id(Ra, robot_id))
            else:
                continue
            
            
        return Ra, to_pickup, free_agents, True
    
    elif hash_map != {}: 
        
        # path = a_star(G, (45, 12), (114, 81))
        # print(len(path))
        
        # Get unassigned tasks and unassigned robots
        assigned_task_ids = set([x[0] for x in Ra])
        task_ids = set([x[0] for x in J])
        
        unassigned_task_ids = task_ids - assigned_task_ids
        
        assigned_robot_ids = set()
        for assignment in Ra:
            assigned_robot_ids |= set([assignment[-1]])
        unassigned_robot_ids = set(np.arange(0, len(Rs))) - assigned_robot_ids
        
        
        if len(unassigned_robot_ids) <= 0:
            return Ra, to_pickup, free_agents, True
        elif len(unassigned_task_ids) <= 0:
            return Ra, to_pickup, free_agents, True
        else:
            pass
        
        # Create cost matrix
        cost_matrix = []
        for unassigned_robot_id in unassigned_robot_ids:
            robot_cost = []
            for unassigned_task_id in unassigned_task_ids:
                # In-case a_star breaks and does not return a solution, try again
                # TODO: Debug a_start implementation for why this occurs so infrequently
                path = hash_map[get_robot_state(Rs, unassigned_robot_id)][get_task_start_location(J, unassigned_task_id)]
                
                if not path:
                    robot_cost.append(np.inf)
                else:
                    robot_cost.append(len(path))
            cost_matrix.append(robot_cost)
        
        # Assign tasks from cost matrix
        is_solution = True
        cost_matrix = np.asarray(cost_matrix, dtype=float)    
        try: 
            for unassigned_robot_id in unassigned_robot_ids:
                # try: 
                min_index_row, min_index_col = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                Ra.append((list(unassigned_task_ids)[min_index_col], list(unassigned_robot_ids)[min_index_row]))
                S.add_estimated_pickup_distance(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
                S.add_estimated_pickup_duration(list(unassigned_task_ids)[min_index_col], np.min(cost_matrix) - 1)
                while True:
                    estimated_task_path = a_star(G, get_task_start_location(J, list(unassigned_task_ids)[min_index_col]), get_task_goal_location(J, list(unassigned_task_ids)[min_index_col]))
                    if estimated_task_path:
                        break
                    elif not estimated_task_path:
                        # G.draw_graph()
                        print(get_task_start_location(J, list(unassigned_task_ids)[min_index_col])) 
                        print(get_task_goal_location(J, list(unassigned_task_ids)[min_index_col]))
                        is_solution = False
                        break
                if is_solution:
                    S.add_estimated_distance(list(unassigned_task_ids)[min_index_col], len(estimated_task_path) - 1)
                    S.add_estimated_duration(list(unassigned_task_ids)[min_index_col], len(estimated_task_path) - 1)
                else:
                    S.add_estimated_distance(list(unassigned_task_ids)[min_index_col], np.Infinity)
                    S.add_estimated_duration(list(unassigned_task_ids)[min_index_col], np.Infinity)
                is_solution = True
                
                cost_matrix[:, min_index_col] = np.inf
                cost_matrix[min_index_row, :] = np.inf
                
                unassigned_task_ids.remove(list(unassigned_task_ids)[min_index_col])
                if len(unassigned_task_ids) == 0:
                    break
        except:
            return Ra, to_pickup, free_agents, False
            # except:
            #     print("++++++++++++++++++++++++++++++++Skipping Allocation++++++++++++++++++++++++++++++++")
            #     for task in Ra:
            #         robot_id = task[-1]
            #         if robot_id in free_agents:
            #             free_agents.remove(robot_id)
            #             to_pickup.add(robot_id)
                        
            #             S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
            #             S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
                        
            #             S.add_actual_duration(get_assigned_task_id(Ra, robot_id))
            #             S.add_actual_pickup_duration(get_assigned_task_id(Ra, robot_id))
            #         else:
            #             continue
            #     return Ra, to_pickup, free_agents

        # Update statistics, free agents, and to_pickup
        for task in Ra:
            robot_id = task[-1]
            if robot_id in free_agents:
                free_agents.remove(robot_id)
                to_pickup.add(robot_id)
                
                S.add_actual_distance(get_assigned_task_id(Ra, robot_id))
                S.add_actual_pickup_distance(get_assigned_task_id(Ra, robot_id))
                
                S.add_actual_duration(get_assigned_task_id(Ra, robot_id))
                S.add_actual_pickup_duration(get_assigned_task_id(Ra, robot_id))
            else:
                continue
            
            
        return Ra, to_pickup, free_agents, True