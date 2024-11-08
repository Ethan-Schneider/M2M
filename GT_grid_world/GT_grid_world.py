import time

from src import graph, simulate, task_allocation, case_request_generator, router, visualize, statistics
    
def execute(S : statistics.Stats, map : str, I: tuple, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            max_task_number : int = 20,
            task_assignment_strategy : str = "closest_robot",
            path_planning_strategy : str = "ecbs"):
    # Unpack Robot State (Rs) and Graph (G)
    Rs, G = I
    # Initilize empty set of tasks, task is defined as (id, start_loc, goal_loc)
    J = set()
    # Initilize robot allocation to empty: allocation is defined as (task_id, robot_id)
    Ra = []
    
    last_task_id = 0
    
    to_pickup = set([])
    to_delivery = set([])
    free_agents = set([robot[0] for robot in Rs])
    # Commit
    for t in range(T):
        print("============================= T : " + str(t) + "=============================")
        # Check if new tasks need to be generated
        print("=============================" + "Task Generation"+ "=============================")
        if t%frequency == 0:
            if frequency >= 1.: 
                N = 1
            elif frequency < 1.: 
                N = frequency**-1
            else:
                N = 0
            # Generate new tasks
            tik = time.time()
            J_new, last_task_id = case_request_generator.CRG(J, G, N, inbound_to_outbound_ratio, last_task_id, max_task_number, case_request_strategy)
            tok = time.time()
            S.add_total_CRG_time(tok-tik)
            # Append the new tasks to the list of tasks
            J |= J_new
        # total_locations = set()
        # print("Tasks: ")
        # for task in J:
        #     print(task)
        #     if task[1] in total_locations:
        #         print("REPEATED LOCATION: ", task[1])
        #     if task[2] in total_locations:
        #         print("REPEATED LOCATION: ", task[2])
        #     total_locations.add(task[1])
        #     total_locations.add(task[2])
        # print(len(total_locations))
        # Assign unassigned tasks to robots
        tik = time.time()
        print("=============================" + "Task Allocation"+ "=============================")
        Ra, to_pickup, free_agents = task_allocation.TaskAllocation(S, G, Rs, Ra, J, task_assignment_strategy, to_pickup, free_agents)
        tok = time.time()
        S.add_total_TA_time(tok-tik)
        
        
        print("=============================" +"Routing"+ "=============================")
        tik = time.time()
        robot_sequences = router.pathPlan(G, map, Rs, Ra, J, path_planning_strategy, to_pickup, to_delivery, free_agents)
        tok = time.time()
        S.add_total_PF_time(tok-tik)
        if not robot_sequences:
            print("No valid path plan, exiting ... ")
            return 
        
        
        print("=============================" +"Taking Step"+ "=============================")
        tik = time.time()
        Rs, Ra, J, to_pickup, to_delivery, free_agents = simulate.simulate(S, G, Rs, robot_sequences, Ra, J, to_delivery, to_pickup, free_agents)
        tok = time.time()
        S.add_total_SIM_time(tok-tik)
        print(G.get_all_occupied())
    return
        
def main():
    # Init values for task frequency and ratio, total number of timesteps, etc. 
    frequency = 1
    inbound_outbound_ratio = 1.0
    initial_inventory_amount = 25.

    task_generation_strategy = "informed_uniform"
    max_current_tasks = 40
    
    task_assignment_strategy = "a_star"
    
    path_planning_strategy = "pbs"
    
    map = "GT_grid_world/src/maps/symbotic_large_w_top"
    
    num_robots = 40
    DOF = 4
    
    # Init graph with number of robot, map file, DOF, deterministic, warehouse initialization strategy, warehouse initial capacity number
    G = graph.Graph(num_robots, map, DOF, True, "uniform", initial_inventory_amount)
    
    T = 3600
    S = statistics.Stats(num_robots, T)
    
    #Initilize state of robots (robot_id, state)
    Rs_init = []
    robot_id = 0
    for location in G.get_all_occupied():
        Rs_init.append((robot_id, location))
        robot_id += 1 
    
    init_locations = [state[-1] for state in Rs_init]
    S.add_paths(init_locations)
    
    
    tik = time.time()
    # Execute online algorithm
    execute(S, map, (Rs_init, G), frequency, inbound_outbound_ratio, T, 
            task_generation_strategy, max_current_tasks, task_assignment_strategy, path_planning_strategy)    
    tok = time.time()
    S.set_total_runtime(tok-tik)  
    
    S.trim_data()
    S.print_statistics()
    S.task_length()
    S.output_graphs()
    
    # print("============================Visualizing Output============================")
    # visualize.main((G.width, G.height), G.obstacles, S.return_full_paths(), str(data/videos/path_planning_strategy) + "_" + str(T) + "_" + str(task_assignment_strategy) + ".mp4", speed=4)


if __name__=="__main__":
    main()