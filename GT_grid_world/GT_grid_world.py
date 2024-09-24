import os
import numpy as np

from src import graph, router, simulate, task_allocation, case_request_generator, item

def main():
    #Initilize state of robots
    Rs_init = [(0, (80, 100), True, -1)]
    
    G = graph.Graph(8, "GT_grid_world/src/maps/symbotic_small", 4, True, "uniform", 25.)
    # Init values for task frequency and ratio, total number of timesteps, etc. 
    frequency = 2
    inbound_outbound_ratio = 1.0
    T = 35
    task_generation_strategy = "informed_uniform"
    
    task_assignment_strategy = "closest_robot"
    
    # Execute online algorithm
    execute((Rs_init, G), frequency, inbound_outbound_ratio, T, task_generation_strategy)
    
    
def execute(I: tuple, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            task_assignment_strategy : str = "closest_robot"):
    # Unpack Robot State (Rs) and Graph (G)
    Rs, G = I
    # Initilize empty set of tasks
    J = set()
    # Initilize robot allocation to empty
    Ra = []
    
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
        
        
    print(J)
    print(len(J))

if __name__=="__main__":
    main()