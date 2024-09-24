import numpy as np

def TaskAllocation(Rs : list, Ra : list, J : set, task_assignment_strategy : str):
    # make a set of all tasks that are not currently allocated
    
    # if statement for which strategy will be used
    
    # ------ closest robot to closest task
    # compute the distance from a task to every robot's current location
    
    if task_assignment_strategy == "closest_robot":
        # Algorithm assigns the closest task to the closest agent at every time-step
        
        assigned_task_ids = set([x[0] for x in Ra])
        task_ids = set([x[0] for x in J])
        
        unassigned_task_ids = task_ids - assigned_task_ids
        # Get list of unassigned task locations 
        unassigned_tasks = [x for x in J]
        
        # Compare each unassigned task location to each agent's location to determine which robot to allocate to
        
        # return list of assignments 
        