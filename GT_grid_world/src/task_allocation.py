import numpy as np

def TaskAllocation(Rs : list, Ra : list, J : set, task_assignment_strategy : str):
    # make a set of all tasks that are not currently allocated
    
    # if statement for which strategy will be used
    
    # ------ closest robot to closest task
    # compute the distance from a task to every robot's current location
    
    if task_assignment_strategy == "closest_robot":
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
            return Ra

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
            
        
        return Ra
