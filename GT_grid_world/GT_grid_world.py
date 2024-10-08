import os
import numpy as np

from src import graph, simulate, task_allocation, case_request_generator, item, router, visualize
    
def execute(I: tuple, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            task_assignment_strategy : str = "closest_robot",
            path_planning_strategy : str = "ecbs"):
    # Unpack Robot State (Rs) and Graph (G)
    Rs, G = I
    # Initilize empty set of tasks, task is defined as (id, start_loc, goal_loc)
    J = set()
    # Initilize robot allocation to empty: allocation is defined as (task_id, robot_id)
    Ra = []
    
    # Init robot path over the course of the simulation
    robot_path_sequences = []
    for robot in Rs:
        robot_path_sequences.append([robot[1]])
    
    last_task_id = 0
    
    to_pickup = set([])
    to_delivery = set([])
    free_agents = set([robot[0] for robot in Rs])
    
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
            
        print("========================== Prior to TA ==========================")
        # Assign unassigned tasks to robots
        Ra, to_pickup, free_agents = task_allocation.TaskAllocation(Rs, Ra, J, task_assignment_strategy, to_pickup, free_agents)
        print("========================== Prior to MAPF ==========================")

        robot_sequences = router.pathPlan(G, Rs, Ra, J, path_planning_strategy, to_pickup, to_delivery, free_agents)
        
        print("========================== Prior to Simulating ==========================")
        
        Rs, Ra, J, to_pickup, to_delivery, free_agents, robot_path_sequences = simulate.simulate(Rs, robot_sequences, Ra, J, to_delivery, to_pickup, free_agents, robot_path_sequences)
        
    return robot_path_sequences
        
def main():
    G = graph.Graph(8, "GT_grid_world/src/maps/symbotic_small", 4, True, "uniform", 25.)
    
    #Initilize state of robots (robot_id, state)
    Rs_init = []
    robot_id = 0
    for location in G.get_all_occupied():
        Rs_init.append((robot_id, location))
        robot_id += 1 
    
    # Init values for task frequency and ratio, total number of timesteps, etc. 
    frequency = 1
    inbound_outbound_ratio = 1.0
    T = 40
    task_generation_strategy = "informed_uniform"
    
    task_assignment_strategy = "random"
    
    path_planning_strategy = "ecbs"
    
    # Execute online algorithm
    paths = execute((Rs_init, G), frequency, inbound_outbound_ratio, T, 
            task_generation_strategy, task_assignment_strategy, path_planning_strategy)      
    
    visualize.main((G.width, G.height), G.obstacles, paths)
    
    print("Paths: ", paths)  


if __name__=="__main__":
    main()