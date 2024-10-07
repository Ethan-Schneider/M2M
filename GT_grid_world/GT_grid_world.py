import os
import numpy as np

from src import graph, simulate, task_allocation, case_request_generator, item, router
    
def execute(I: tuple, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            task_assignment_strategy : str = "closest_robot",
            path_planning_strategy : str = "cbs"):
    # Unpack Robot State (Rs) and Graph (G)
    Rs, G = I
    # Initilize empty set of tasks, task is defined as (id, start_loc, goal_loc)
    J = set()
    # Initilize robot allocation to empty: allocation is defined as (task_id, robot_id)
    Ra = []
    robot_path_sequences = []
    
    last_task_id = 0
    
    for t in range(T):
        # Check if new tasks need to be generated
        if t%frequency == 0:
            if frequency >= 1.: 
                N = 1
            elif frequency < 1.: 
                N = frequency**-1
            else:
                N = 0
            # Generate new tasks
            J_new, last_task_id = case_request_generator.CRG(J, G, N, inbound_to_outbound_ratio, last_task_id, case_request_strategy)
            # Append the new tasks to the list of tasks
            J |= J_new
            
        # Assign unassigned tasks to robots
        Ra = task_allocation.TaskAllocation(Rs, Ra, J, task_assignment_strategy)
        print(Ra)
        robot_path_sequences = router.pathPlan(G, Rs, Ra, J, path_planning_strategy)
        
        
def main():
    G = graph.Graph(8, "GT_grid_world/src/maps/symbotic_small", 4, True, "uniform", 25.)
    
    #Initilize state of robots (robot_id, state, free_agent)
    Rs_init = []
    robot_id = 0
    for location in G.get_all_occupied():
        Rs_init.append((robot_id, location, True))
        robot_id += 1 
    
    # Init values for task frequency and ratio, total number of timesteps, etc. 
    frequency = 1
    inbound_outbound_ratio = 1.0
    T = 5
    task_generation_strategy = "informed_uniform"
    
    task_assignment_strategy = "closest_robot"
    
    path_planning_strategy = "cbs"
    
    # Execute online algorithm
    execute((Rs_init, G), frequency, inbound_outbound_ratio, T, 
            task_generation_strategy, task_assignment_strategy, path_planning_strategy)        


if __name__=="__main__":
    main()