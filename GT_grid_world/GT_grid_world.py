import time

from src import graph, simulate, task_allocation, case_request_generator, router
from src.analysis import visualize, statistics

def execute(S : statistics.Stats, map : str, I: tuple, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            max_task_number : int = 20,
            task_assignment_strategy : str = "cost_matrix",
            path_planning_strategy : str = "ecbs", time_limit : int = 99999, hash_map : dict = {}):
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
    robot_sequences = []
    for _ in range(len(Rs)):
        robot_sequences.append([])
    
    global_tik = time.time()
    
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
            print(N)
            # Generate new tasks
            tik = time.time()
            J_new, last_task_id = case_request_generator.CRG(J, G, N, inbound_to_outbound_ratio, last_task_id, max_task_number, case_request_strategy)
            tok = time.time()
            S.add_total_CRG_time(tok-tik)
            # Append the new tasks to the list of tasks
            J |= J_new
            
        tik = time.time()
        print(len(J))
        print("=============================" + "Task Allocation"+ "=============================")
        Ra, to_pickup, free_agents = task_allocation.TaskAllocation(S, G, Rs, Ra, J, task_assignment_strategy, to_pickup, free_agents, map, robot_sequences)

        print("Ra length: ", len(Ra))
        
        tok = time.time()
        S.add_total_TA_time(tok-tik)
        S.append_task_allocation(Ra, J, Rs)
        
        print("=============================" +"Routing"+ "=============================")
        tik = time.time()
        # TODO: Refactor this / clean the logic b/c during first 40 timesteps it replans the plan
        # What should happen: 
        for sequence in robot_sequences:
            if sequence == []:
                robot_sequences = router.pathPlan(G, map, Rs, Ra, J, path_planning_strategy, to_pickup, to_delivery, free_agents, robot_sequences)
                break
            else:
                pass
        tok = time.time()
        S.add_total_PF_time(tok-tik)
        if not robot_sequences:
            print("Robot Sequences: ", robot_sequences)
            print("No valid path plan, exiting ... ")
            return 
        
        
        print("=============================" +"Taking Step"+ "=============================")
        tik = time.time()
        Rs, Ra, J, to_pickup, to_delivery, free_agents = simulate.simulate(S, G, Rs, robot_sequences, Ra, J, to_delivery, to_pickup, free_agents)
        tok = time.time()
        S.add_total_SIM_time(tok-tik)
        print(G.get_all_occupied())
        
        global_tok = time.time()
        
        if (global_tok - global_tik) >= time_limit:
            return
    return
        
def main():
    # Init values for task frequency and ratio, total number of timesteps, etc. 
    frequency = 0.1
    inbound_outbound_ratio = 1.0
    initial_inventory_amount = 25.

    task_generation_strategy = "informed_uniform"
    max_current_tasks = 50
    
    task_assignment_strategy = "a_star"
    
    path_planning_strategy = "pbs"
    
    map = "data/maps/symbotic_large_w_top"
    
    num_robots = 40
    DOF = 4
    
    # Init graph with number of robot, map file, DOF, deterministic, warehouse initialization strategy, warehouse initial capacity number
    G = graph.Graph(num_robots, map, DOF, True, "uniform", initial_inventory_amount)
    
    T = 1000
    
    output_file = "data/raw_data/" + str(T) + "_" + str(task_assignment_strategy) + "_" +str(path_planning_strategy) + ".json"
    S = statistics.Stats(num_robots, T, output_file)
    
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
    # S.print_statistics()
    # S.task_length()
    S.save_data()
    S.output_graphs()
    
    # print("============================Visualizing Output============================")
    # visualize.main((G.width, G.height), G.obstacles, S.return_full_paths(), 'data/videos/' + str(path_planning_strategy) + "_" + str(T) + "_" + str(task_assignment_strategy) + ".mp4", speed=4)

def arg_main(S : statistics.Stats, G : graph.Graph, num_robots : int, T : int, task_generation_strategy : str, task_assignment_strategy : str, path_planning_strategy : str, map : str, time_limit : int = 99999, visualize : bool = False):
    frequency = 0.1
    inbound_outbound_ratio = 1.0 
    
    max_current_tasks = num_robots
    
    #Initilize state of robots (robot_id, state)
    Rs_init = []
    robot_id = 0
    for location in G.get_all_occupied():
        Rs_init.append((robot_id, location))
        robot_id += 1 
    
    init_locations = [state[-1] for state in Rs_init]
    S.add_paths(init_locations)
    
    # hash_map = utils.construct_distance_hashmap(G)
    
    tik = time.time()
    # Execute online algorithm
    execute(S, map, (Rs_init, G), frequency, inbound_outbound_ratio, T, 
            task_generation_strategy, max_current_tasks, task_assignment_strategy, path_planning_strategy, time_limit)    
    tok = time.time()
    S.set_total_runtime(tok-tik)  
    
    # S.print_statistics()
    # S.task_length()
    S.save_data()
    
    folder_name = str(T) + "_" + str(task_generation_strategy) + "_" + str(task_assignment_strategy) + "_" +str(path_planning_strategy) + "_" + str(num_robots)
    S.output_graphs(folder_name)

    if visualize:
        print("============================Visualizing Output============================")
        visualize.main((G.width, G.height), G.obstacles, S.return_full_paths(), 'data/videos/' + str(path_planning_strategy) + "_" + str(T) + "_" + str(task_assignment_strategy) + ".mp4", speed=4)

    return 

def entry():
    initial_inventory_amount = 25.
    
    map = "data/maps/symbotic_small"
    
    time_limit = 14500
    
    T = [5000]*120
    # num_robots = list(range(7, 100))
    num_robots = [25]
    DOF = 4
    task_generation_strategy = "informed_uniform"
    task_assignment_strategy = "lns"
    path_planning_strategy = "ecbs"
    
    visualize = False
    
    for i in range(len(num_robots)):
        output_file = "data/raw_data/" + str(T[i]) + "_" + str(task_generation_strategy) + "_" + str(task_assignment_strategy) + "_" +str(path_planning_strategy) + "_" + str(num_robots[i]) + ".json"      
        S = statistics.Stats(num_robots[i], T[i], output_file)
        G = graph.Graph(num_robots[i], map, DOF, True, "uniform", initial_inventory_amount)
        
        arg_main(S, G, num_robots[i], T[i], task_generation_strategy, task_assignment_strategy, path_planning_strategy, map, time_limit)

if __name__=="__main__":
    entry()