import time
import numpy as np

from src import graph, simulate, task_allocation, case_request_generator, router, agent
from src.analysis import visualize, statistics

def execute(S : statistics.Stats, map : str, Rs : agent.AgentLoader, G : graph.Graph, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            max_task_number : int = 20,
            task_assignment_strategy : str = "cost_matrix", task_sequences : bool = False,
            path_planning_strategy : str = "ecbs", time_limit : int = 99999, hash_map : dict = {}):
    # Initilize empty set of tasks, task is defined as (id, start_loc, goal_loc)
    J = set()
    # Initilize robot allocation to empty: allocation is defined as (task_id, robot_id)
    # If task_sequences, then Ra is defined as ([task_ids], robot_id)

    last_task_id = 0
    
    global_tik = time.time()
    
    for t in range(T):
        print("============================= T : " + str(t) + "=============================")
        # Check if new tasks need to be generated
        print("=============================" + "Task Generation"+ "=============================")
        if t%frequency == 0:
            if len(J) < max_task_number:
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
            
        tik = time.time()

        print("=============================" + "Task Allocation"+ "=============================")
        num = 0
        for agent in Rs.agents:
            if agent.path_sequence == []:
                num += 1
        if num == len(Rs.agents):
            # Unallocate all agents from their tasks
            for agent in Rs.agents:
                agent.task_sequence = []
                agent.status = 0
        
        # Check if all tasks are allocated, if so, skip
        total = 0
        for agent in Rs.agents:
            total += len(agent.task_sequence)
            
        if total < max_task_number:
            Rs = task_allocation.TaskAllocation(S, G, Rs, J, task_assignment_strategy, task_sequences, map)

        tok = time.time()
        S.add_total_TA_time(tok-tik)
        S.append_task_allocation(Rs, J)

        # Check if any agent is allocated the same tasks
        for agent in Rs.agents:
            task_ids = set()
            for task_id in agent.task_sequence:
                if task_id in task_ids:
                    raise ValueError(f"Task {task_id} is allocated to multiple agents.")
                task_ids.add(task_id)
        
        if task_assignment_strategy != "lns_fully_informed":
                
            print("=============================" +"Routing"+ "=============================")
            tik = time.time()
            # TODO: Refactor this / clean the logic b/c during first 40 timesteps it replans the plan
            # What should happen: 
            for agent in Rs.agents:
                if agent.path_sequence == []:
                    Rs = router.pathPlan(G, map, Rs, J, path_planning_strategy)
                    break
                
            tok = time.time()
            S.add_total_PF_time(tok-tik)
            
            print(f"Agent path sequences: {[agent.path_sequence for agent in Rs.agents]}")

        print("=============================" +"Taking Step"+ "=============================")
        tik = time.time()
        Rs, J = simulate.simulate(S, G, Rs, J)
        tok = time.time()
        S.add_total_SIM_time(tok-tik)

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
    robots = []
    for robot_id, location in enumerate(G.get_all_occupied()):
        robots.append(agent.Agent(robot_id, location))
    
    al = agent.AgentLoader(robots)
    
    init_locations = [robot.state for robot in robots]
    S.add_paths(init_locations)
    
    
    tik = time.time()
    # Execute online algorithm
    execute(S, map, (al, G), frequency, inbound_outbound_ratio, T, 
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

def arg_main(S : statistics.Stats, G : graph.Graph, num_robots : int, T : int, task_generation_strategy : str, task_assignment_strategy : str, task_sequences : bool, path_planning_strategy : str, map : str, time_limit : int = 99999, to_visualize : bool = False):
    frequency = 1.0
    inbound_outbound_ratio = 1.0 
    
    max_current_tasks = 40
    
    #Initilize state of robots (robot_id, state)
    robots = []
    for robot_id, location in enumerate(G.get_all_occupied()):
        robots.append(agent.Agent(robot_id, location))
    
    Rs = agent.AgentLoader(robots)
    
    init_locations = [agent.state for agent in Rs.agents]
    S.add_paths(init_locations)
    
    # hash_map = utils.construct_distance_hashmap(G)
    
    tik = time.time()
    # Execute online algorithm
    execute(S, map, Rs, G, frequency, inbound_outbound_ratio, T, case_request_strategy=task_generation_strategy, max_task_number=max_current_tasks, task_assignment_strategy=task_assignment_strategy, path_planning_strategy=path_planning_strategy, time_limit=time_limit)    
    tok = time.time()
    S.set_total_runtime(tok-tik)  
    
    # S.print_statistics()
    # S.task_length()
    S.save_data()
    
    folder_name = str(T) + "_" + str(task_generation_strategy) + "_" + str(task_assignment_strategy) + "_" +str(path_planning_strategy) + "_" + str(num_robots)
    S.output_graphs(folder_name)

    if to_visualize:
        print("============================Visualizing Output============================")
        visualize.main((G.width, G.height), G.obstacles, S.return_full_paths(), 'data/videos/' + str(path_planning_strategy) + "_" + str(T) + "_" + str(task_assignment_strategy) + ".mp4", speed=4)

    return 

def entry():
    initial_inventory_amount = 25.
    
    map = "data/maps/symbotic_large"
    
    time_limit = 57600
    
    np.random.seed(2)
    
    T = [1005]*120
    # num_robots = list(range(7, 100))
    num_robots = [40]
    DOF = 4
    task_generation_strategy = "informed_uniform"
    # task_assignment_strategy = "random"
    task_assignment_strategy = "lns_fully_informed"
    # task_assignment_strategy = "lns"
    # task_assignment_strategy = "cost_matrix"
    task_sequences = True
    path_planning_strategy = "ecbs"
    
    visualize = False
    
    for i in range(len(num_robots)):
        output_file = "data/raw_data/" + str(T[i]) + "_" + str(task_generation_strategy) + "_" + str(task_assignment_strategy) + "_" +str(path_planning_strategy) + "_" + str(num_robots[i]) + ".json"      
        S = statistics.Stats(num_robots[i], T[i], output_file)
        G = graph.Graph(num_robots[i], map, DOF, True, "uniform", initial_inventory_amount)
        
        arg_main(S, G, num_robots[i], T[i], task_generation_strategy, task_assignment_strategy, task_sequences, path_planning_strategy, map, time_limit, visualize)

if __name__=="__main__":
    entry()