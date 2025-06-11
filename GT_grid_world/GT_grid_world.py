import time
import numpy as np
import argparse

from src import graph, simulate, task_allocation, case_request_generator, router, agent
from src.analysis import visualize, statistics, buffer

def execute(S : statistics.Stats, B : buffer.Buffer, map : str, Rs : agent.AgentLoader, G : graph.Graph, frequency : float, inbound_to_outbound_ratio: float, 
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
    skip = False
    
    for t in range(T):
        print("============================= T : " + str(t) + "=============================")
        # Check if new tasks need to be generated
        # Update Buffer 
        B.add(Rs.get_agent_states(), t)
        
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
                J_new, last_task_id = case_request_generator.CRG(S, t,J, G, N, inbound_to_outbound_ratio, last_task_id, max_task_number, case_request_strategy)
                tok = time.time()
                S.add_total_CRG_time(tok-tik)
                # Append the new tasks to the list of tasks
                J |= J_new
            
        tik = time.time()

        if not skip:
            print("=============================" + "Task Allocation"+ "=============================")
            # Check if all tasks are allocated, if so, skip
            total = 0
            for agent in Rs.agents:
                total += len(agent.task_sequence)
                
            if total < max_task_number:
                Rs = task_allocation.TaskAllocation(S, G, Rs, J, task_assignment_strategy, task_sequences, map, t)

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
            
            print("=============================" +"Routing"+ "=============================")
            tik = time.time()

            for agent in Rs.agents:
                if agent.path_sequence == []:
                    Rs = router.pathPlan(G, map, Rs, J, path_planning_strategy, S)
                    break
                
            tok = time.time()
            S.add_total_PF_time(tok-tik)
            
            print(f"Agent path sequences: {[agent.path_sequence for agent in Rs.agents]}")
            soc = 0
            for agent in Rs.agents:
                soc += len(agent.path_sequence)
            S.set_soc(soc)

            print("=============================" +"Taking Step"+ "=============================")
            tik = time.time()
            Rs, J = simulate.simulate(S, B, G, Rs, J, map, t)
            tok = time.time()
            S.add_total_SIM_time(tok-tik)
            
            S.compute_unallocated_agents(Rs)

            global_tok = time.time()
            
            if (global_tok - global_tik) >= time_limit:
                return
    return

def main(seed: int, num_robots: int, T: int, max_number_tasks: int, 
         task_generation_strategy: str, task_assignment_strategy: str,
         path_planning_strategy: str, map_name: str, time_limit: int = 86400,
         visualize_output: bool = False, initial_inventory: float = 25.0, 
         frequency: float = 1.0, inbound_outbound_ratio: float = 1.0):
    """
    Run a single instance of the simulation with specified parameters.
    
    Args:
        seed: Random seed for reproducibility
        num_robots: Number of robots in the simulation
        T: Time horizon
        max_number_tasks: Maximum number of tasks
        task_generation_strategy: Strategy for generating tasks
        task_assignment_strategy: Strategy for assigning tasks
        path_planning_strategy: Strategy for path planning
        map_name: Name of the map to use
        time_limit: Maximum runtime in seconds
        visualize_output: Whether to generate visualization
        initial_inventory: Initial inventory amount
        frequency: Frequency of task generation
        inbound_outbound_ratio: Ratio of inbound to outbound tasks
    """
    np.random.seed(seed)
    
    output_file = f"data/raw_data/{T}_{task_generation_strategy}_{task_assignment_strategy}_{path_planning_strategy}_{num_robots}_{max_number_tasks}_{seed}_no_seq_full.json"
    buffer_file = f"data/buffer_data/{T}_{task_generation_strategy}_{task_assignment_strategy}_{path_planning_strategy}_{num_robots}_{max_number_tasks}_{seed}"
    
    B = buffer.Buffer(80, buffer_file)
    S = statistics.Stats(num_robots, T, output_file)
    G = graph.Graph(num_robots, map_name, True, "uniform", initial_inventory)
    
    # Initialize state of robots (robot_id, state)
    robots = []
    for robot_id, location in enumerate(G.get_all_occupied()):
        robots.append(agent.Agent(robot_id, location))
    
    Rs = agent.AgentLoader(robots)
    
    init_locations = [agent.state for agent in Rs.agents]
    S.add_paths(init_locations)
    
    tik = time.time()
    # Execute online algorithm
    execute(S, B, map_name, Rs, G, frequency, inbound_outbound_ratio, T, 
            case_request_strategy=task_generation_strategy, 
            max_task_number=max_number_tasks, 
            task_assignment_strategy=task_assignment_strategy, 
            path_planning_strategy=path_planning_strategy, 
            time_limit=time_limit)
    tok = time.time()
    S.set_total_runtime(tok-tik)
    
    S.save_data()

    folder_name = str(T) + "_" + str(task_generation_strategy) + "_" + str(task_assignment_strategy) + "_" +str(path_planning_strategy) + "_" + str(num_robots) + "_" + str(max_number_tasks) + "_" + str(seed) + "_full"
    # S.output_graphs(folder_name)
    
    if visualize_output:
        print("============================Visualizing Output============================")
        visualize.main((G.width, G.height), G.obstacles, S.return_full_paths(), 
                      f'data/videos/{path_planning_strategy}_{T}_{task_assignment_strategy}.mp4', 
                      speed=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run grid world simulation with specified parameters')
    
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--num-robots', type=int, required=True, help='Number of robots')
    parser.add_argument('--time-horizon', type=int, required=True, help='Time horizon T')
    parser.add_argument('--max-tasks', type=int, required=True, help='Maximum number of tasks')
    parser.add_argument('--task-gen-strategy', type=str, required=True, 
                       choices=['informed_uniform', 'uninformed_uniform'],
                       help='Task generation strategy')
    parser.add_argument('--task-assign-strategy', type=str, required=True,
                       choices=['lns', 'p_lns', 'cost_matrix', 'random'],
                       help='Task assignment strategy')
    parser.add_argument('--path-planning-strategy', type=str, required=True,
                       choices=['ecbs'],
                       help='Path planning strategy')
    parser.add_argument('--map', type=str, default='data/maps/symbotic_small',
                       help='Map file path')
    parser.add_argument('--time-limit', type=int, default=86400,
                       help='Maximum runtime in seconds')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization output')
    parser.add_argument('--initial-inventory', type=float, default=25.0,
                       help='Initial inventory amount')
    parser.add_argument('--frequency', type=float, default=1.0,
                       help='Frequency of task generation')
    parser.add_argument('--inbound-outbound-ratio', type=float, default=1.0,
                       help='Ratio of inbound to outbound tasks')
    
    args = parser.parse_args()
    
    main(
        seed=args.seed,
        num_robots=args.num_robots,
        T=args.time_horizon,
        max_number_tasks=args.max_tasks,
        task_generation_strategy=args.task_gen_strategy,
        task_assignment_strategy=args.task_assign_strategy,
        path_planning_strategy=args.path_planning_strategy,
        map_name=args.map,
        time_limit=args.time_limit,
        visualize_output=args.visualize,
        initial_inventory=args.initial_inventory,
        frequency=args.frequency,
        inbound_outbound_ratio=args.inbound_outbound_ratio
    )