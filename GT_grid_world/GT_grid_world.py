import time
import numpy as np
import argparse

from src import graph, simulate, task_allocation, case_request_generator, router, agent
from src.task_allocation_algorithms.local_repair.branch_and_bound_V2 import BnB
from src.task_allocation_algorithms.local_repair.exact_repair import HA_exact_repair
from src.task_allocation_algorithms.repair_detection.backtracking import detect_backtracking
from src.task_allocation_algorithms.repair_detection.duration_difference import duration_difference
from src.task_allocation_algorithms.repair_detection.sliding_window_progress import sliding_window_progress
from src.analysis import visualize, statistics

def execute(S : statistics.Stats, map : str, Rs : agent.AgentLoader, G : graph.Graph, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            max_task_number : int = 20,
            initial_task_assignment_strategy : str = "lns",
            improvement_task_assignment_strategy : str = "py_lns",
            path_planning_strategy : str = "ecbs", time_limit : int = 999999,
            cost_calculation_method : str = "manhattan",
            removal_operator : str = "worst", repair_operator : str = "greedy",
            acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
            deadline_generation_method: str = "constant",
            deadline_offset: float = 30,
            output_intermediate_data: bool = False,
            intermediate_data_interval: int = 1800,
            base_cost_weight: float = 1.0,
            deadline_weight: float = 0.0,
            sku_distribution_weight: float = 0.0,
            agent_unallocated_penalty: float = 0.0,
            solution_repair_detection_function: str = "none",
            solution_repair_function: str = "none",
            initial_inventory : float = 25.0,
            shuffle_percentage: float = 0.0,
            aisle_dual_cycle: bool = False,
            driveway_dual_cycle: bool = False):
    # Initilize empty dict of tasks, task is defined as (id: (start_loc, goal_loc, deadline, sku_id, inbound))
    J = {}

    last_task_id = 0
    
    global_tik = time.time()
    for t in range(T):
        print("============================= T : " + str(t) + "=============================")
        # Check if new tasks need to be generated
        # Update Buffer 
        # B.add(Rs.get_agent_states(), t)
        
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

                J, last_task_id, __, __ = case_request_generator.CRG(S, t, J, G, Rs, N, 
                                                                                                inbound_to_outbound_ratio, last_task_id, max_task_number,
                                                                                                  G.warehouse, case_request_strategy, 
                                                                                                  deadline_generation_method, deadline_offset, improvement_task_assignment_strategy,
                                                                                                  initial_inventory,
                                                                                                  shuffle_percentage=shuffle_percentage)
                tok = time.time()
                S.add_total_CRG_time(tok-tik)
            
        tik = time.time()

        print("=============================" + "Task Allocation"+ "=============================")
        # Check if all tasks are allocated, if so, skip
        total = 0
        for agent in Rs.agents:
            total += len(agent.task_sequence)
            
        if total < max_task_number:
            print(f"Attempting to allocate tasks")
            Rs, _, _ = task_allocation.TaskAllocation(S, G, Rs, J, initial_task_assignment_strategy, improvement_task_assignment_strategy, map, t, cost_calculation_method, removal_operator, repair_operator, acceptance_function, T_0, alpha, base_cost_weight, deadline_weight, sku_distribution_weight, agent_unallocated_penalty)

        tok = time.time()
        S.add_total_TA_time(tok-tik)

        # Check if any agent is allocated the same tasks
        for agent in Rs.agents:
            task_ids = set()
            for task_id in agent.task_sequence:
                if task_id in task_ids:
                    raise ValueError(f"Task {task_id} is allocated to multiple agents.")
                task_ids.add(task_id)

        # # Count the number of outbound and inbound tasks allocated to each agent
        # outbound_tasks_allocated = 0
        # inbound_tasks_allocated = 0
        # initial_outbound_tasks = 0
        # initial_inbound_tasks = 0
        # for agent in Rs.agents:
        #     for n, task in enumerate(agent.task_sequence):
        #         task_id = task[0]
        #         if task_id in outbound_tasks:
        #             outbound_tasks_allocated += 1
        #         elif task_id in inbound_tasks:
        #             inbound_tasks_allocated += 1
        #         if task_id in outbound_tasks and n == 0:
        #             initial_outbound_tasks += 1
        #         elif task_id in inbound_tasks and n == 0:
        #             initial_inbound_tasks += 1
        # print(f"Outbound tasks allocated: {outbound_tasks_allocated}")
        # print(f"Inbound tasks allocated: {inbound_tasks_allocated}")
        # print(f"Initial outbound tasks: {initial_outbound_tasks}")
        # print(f"Initial inbound tasks: {initial_inbound_tasks}")
        
        print("=============================" +"Routing"+ "=============================")
        tik = time.time()
        
        # if t > 50:
        #     costs = []
        #     for i in range(100):
        #         for agent in Rs.agents:
        #             if agent.path_sequence == []:
        #                 Rs = router.pathPlan(map, Rs, path_planning_strategy, S)
        #                 break
        #         cost = 0
        #         for agent in Rs.agents:
        #             cost += len(agent.path_sequence)
        #         costs.append(cost)
        #     print(f"Routing costs over 100 iterations: {costs}")
        #     print(f"Number of different costs: {len(set(costs))}")
        #     exit()
        # else:
        #     for agent in Rs.agents:
        #         if agent.path_sequence == []:
        #             Rs = router.pathPlan(map, Rs, path_planning_strategy, S)
        #             break            
        
        for agent in Rs.agents:
            if agent.path_sequence == []:
                Rs = router.pathPlan(map, Rs, path_planning_strategy, S)
                break    

        tok = time.time()
        S.add_total_PF_time(tok-tik)
        
        soc = 0
        for agent in Rs.agents:
            soc += len(agent.path_sequence)
        S.set_soc(soc)
        
        print("=============================" +"Task Reallocation"+ "=============================")
        # Identify which agents are backtracking, and need to reallocate their current task 

        if solution_repair_function != "none":
            
            detection_tik = time.time()
            
            if solution_repair_detection_function == "Backtracking":
                aisle_groups = detect_backtracking(Rs)
            elif solution_repair_detection_function == "Duration":
                aisle_groups = duration_difference(Rs, G)
            elif solution_repair_detection_function == "Progress":
                aisle_groups = sliding_window_progress(Rs, G)
            elif solution_repair_detection_function == "Ensamble":
                aisle_groups_bt = detect_backtracking(Rs)
                aisle_groups_du = duration_difference(Rs, G)
                aisle_groups_sw = sliding_window_progress(Rs, G)

                # Combine the aisle groups, remove duplicates
                aisle_groups = {}
                for group in [aisle_groups_bt, aisle_groups_du, aisle_groups_sw]:
                    for aisle, agents in group.items():
                        if aisle not in aisle_groups:
                            aisle_groups[aisle] = agents
                        else:
                            for agent in agents:
                                if agent not in aisle_groups[aisle]:
                                    aisle_groups[aisle].append(agent)
            else:
                raise ValueError(f"Unknown solution repair detection function: {solution_repair_detection_function}: Exiting ...")
                
            detection_tok = time.time()
                
            if aisle_groups:
                print(f"Aisle groups for reallocation: { {aisle: [a.id for a in group] for aisle, group in aisle_groups.items()} }")

                agents = [[a.id for a in group] for aisle, group in aisle_groups.items()]
                t_key = S.create_new_realloc_data(t)
                S.reallocation_data[t_key]["agents"] = []
                S.reallocation_data[t_key]["tasks"] = []
                S.reallocation_data[t_key]["prior_path_cost"] = []
                S.reallocation_data[t_key]["post_path_cost"] = [] 
                S.reallocation_data[t_key]["change_in_path_cost"] = []     
                S.reallocation_data[t_key]["path_planning_compute_time"] = []
                S.reallocation_data[t_key]["lower_bound_and_checks"] = []
                S.reallocation_data[t_key]["rejected_solution"] = []
                S.reallocation_data[t_key]["bnb_init_compute_time"] = []
                S.reallocation_data[t_key]["bnb_solve_time"] = []
                S.reallocation_data[t_key]["bnb_routing_time"] = []
                S.reallocation_data[t_key]["detection_computation_time"] = np.abs(detection_tok - detection_tik)

                S.reallocation_data[t_key]["prior_agent_path_cost"] = []
                S.reallocation_data[t_key]["post_agent_path_cost"] = []
                S.reallocation_data[t_key]["prior_group_path_cost"] = []
                S.reallocation_data[t_key]["post_group_path_cost"] = []
                repair_tik = time.time()
                for agent_group in agents:
                    temp_Rs = Rs.copy()
                    if np.abs(time.time() - repair_tik) >= 1.0:
                        break
                    
                    group_cost = 0
                    for agent_id in agent_group:
                        group_cost += len(temp_Rs.get_agent(agent_id).path_sequence)

                    # Skip if only one agent in group
                    if len(agent_group) <= 1:
                        continue

                    task_ids = []
                    for agent_id in agent_group:
                        task = temp_Rs.get_agent(agent_id).task_sequence[0][0]
                        task_ids.append(task)
                    S.reallocation_data[t_key]["tasks"].append(task_ids)

                    prior_cost = 0
                    for agent in temp_Rs.agents:
                        prior_cost += len(agent.path_sequence)
                    S.reallocation_data[t_key]["prior_path_cost"].append(prior_cost)
                    S.reallocation_data[t_key]["prior_agent_path_cost"].append(temp_Rs.get_agent(agent_group[0]).path_sequence.__len__())

                    prior_group_cost  = 0
                    for agent_id in agent_group:
                        prior_group_cost += len(temp_Rs.get_agent(agent_id).path_sequence)
                    S.reallocation_data[t_key]["prior_group_path_cost"].append(prior_group_cost)

                    if solution_repair_function == "HA":
                        if len(agent_group) >= 5:
                            continue

                        HA_agent_group = agent_group.copy()
                        S.reallocation_data[t_key]["agents"].append(HA_agent_group)
                        # temp_Rs = Rs.copy()
                        temp_Rs = HA_exact_repair(temp_Rs, G, S, J, HA_agent_group)
                    elif solution_repair_function == "BnB":
                        if len(agent_group) >= 4:
                            continue
                            
                        S.reallocation_data[t_key]["agents"].append(agent_group)
                        
                        init_tik = time.time()
                        bnb = BnB(temp_Rs, G, J, S, agent_group, map, t_key, prior_cost)
                        S.reallocation_data[t_key]["bnb_init_compute_time"].append(np.abs(time.time() - init_tik))
                        
                        solve_tik = time.time()
                        temp_Rs = bnb.solve()
                        S.reallocation_data[t_key]["bnb_solve_time"].append(np.abs(time.time() - solve_tik))
                    else:
                        print(f"Unknown solution repair function: {solution_repair_function}: Exiting ...")
                        exit()
                        

                    routing_tik = time.time()
                    temp_Rs = router.pathPlan(map, temp_Rs, path_planning_strategy, S)
                    S.reallocation_data[t_key]["bnb_routing_time"].append(np.abs(time.time() - routing_tik))

                    post_cost = 0

                    for agent in temp_Rs.agents:
                        post_cost += len(agent.path_sequence)
                        
                    S.reallocation_data[t_key]["post_agent_path_cost"].append(temp_Rs.get_agent(agent_group[0]).path_sequence.__len__())
                    
                    post_group_cost = 0
                    for agent_id in agent_group:
                        post_group_cost += len(temp_Rs.get_agent(agent_id).path_sequence)
                    S.reallocation_data[t_key]["post_group_path_cost"].append(post_group_cost)

                    post_goals = []
                    for agent_id in agent_group:
                        if temp_Rs.get_agent(agent_id).status == 1:
                            post_goals.append(temp_Rs.get_agent(agent_id).task_sequence[0][1])
                        elif temp_Rs.get_agent(agent_id).status == 2:
                            post_goals.append(temp_Rs.get_agent(agent_id).task_sequence[0][2])

                    S.reallocation_data[t_key]["post_path_cost"].append(post_cost)
                    S.reallocation_data[t_key]["change_in_path_cost"].append(post_cost - prior_cost)
                    
                    if post_cost < prior_cost:
                        Rs = temp_Rs.copy()
                        S.reallocation_data[t_key]["rejected_solution"].append(False)
                    else:
                        S.reallocation_data[t_key]["rejected_solution"].append(True)

                    # print(f"final assignment : {assignment} with cost: {cost}")
                    
                print(f"Repaired Solution")
                S.reallocation_data[t_key]["computation_time"] = np.abs(time.time() - repair_tik)
                print(S.reallocation_data[t_key]["tasks"])

        print("=============================" +"Taking Step"+ "=============================")
        tik = time.time()
        Rs, J = simulate.simulate(
            S, G, Rs, J, map, t,
            aisle_dual_cycle=aisle_dual_cycle,
            driveway_dual_cycle=driveway_dual_cycle,
        )
        tok = time.time()
        S.add_total_SIM_time(tok-tik)
        
        # Log the number of tasks in the system
        S.append_tasks_in_system(len(J))
        
        # Log warehouse inventory state
        S.append_warehouse_inventory_state(G.warehouse, G.get_aisle_locations())
        
        # Log driveway inventory state
        S.append_driveway_inventory_state(G.driveway)
        
        # Log SKU inventory state
        S.append_sku_inventory_state(G.warehouse, G.driveway, G.warehouse.get_all_skus().__len__())
        
        # Log SKU centroids per timestep
        S.append_sku_centroids(G.warehouse, G.warehouse.get_all_skus().__len__())
        
        # Log SKU locations per timestep
        S.append_sku_locations(G.warehouse, G.driveway, G.warehouse.get_all_skus().__len__())
        
        # Record agent statuses and goal locations for this timestep
        S.add_agent_statuses_and_goals(Rs)
        
        S.compute_unallocated_agents(Rs)

        global_tok = time.time()
        
        # if (global_tok - global_tik) >= time_limit:
        #     return
        
        # exit()
        
        # Output intermediate data if enabled
        if output_intermediate_data and t % intermediate_data_interval == 0:
            intermediate_output_file = S.get_output_file().replace(".json", f"_{t}.json")
            S.save_data(intermediate_output_file)
        
    return

def main(seed: int, num_robots: int, T: int, max_number_tasks: int, 
         task_generation_strategy: str, initial_task_assignment_strategy: str, improvement_task_assignment_strategy: str,
         path_planning_strategy: str, map_name: str, time_limit: int = 999999,
         visualize_output: bool = False, initial_inventory: float = 25.0, 
         frequency: float = 1.0, inbound_outbound_ratio: float = 1.0,
         output_graphs: bool = False, num_skus: int = 10,
         weight_init_method: str = "random",
         cost_calculation_method: str = "manhattan",
         removal_operator: str = "worst", repair_operator: str = "greedy",
         acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
         deadline_generation_method: str = "constant",
         deadline_offset: float = 30,
         output_intermediate_data: bool = False,
         intermediate_data_interval: int = 1800,
         base_cost_weight: float = 1.0,
         deadline_weight: float = 0.0,
         sku_distribution_weight: float = 0.0,
         agent_unallocated_penalty: float = 0.0,
         solution_repair_detection_function: str = "none",
         solution_repair_function: str = "none",
         shuffle_percentage: float = 0.0,
         aisle_dual_cycle: bool = False,
         driveway_dual_cycle: bool = False) -> None:
    """
    Run a single instance of the simulation with specified parameters.
    
    Args:
        seed: Random seed for reproducibility
        num_robots: Number of robots in the simulation
        T: Time horizon
        max_number_tasks: Maximum number of tasks
        task_generation_strategy: Strategy for generating tasks
        initial_task_assignment_strategy: Strategy for assigning tasks
        improvement_task_assignment_strategy: Strategy for improving tasks
        path_planning_strategy: Strategy for path planning
        map_name: Name of the map to use
        time_limit: Maximum runtime in seconds
        visualize_output: Whether to generate visualization
        initial_inventory: Initial inventory fill percentage (0.0 to 100.0)
        frequency: Frequency of task generation
        inbound_outbound_ratio: Ratio of inbound to outbound tasks
        output_graphs: Whether to output analysis graphs
        num_skus: Number of unique SKUs in the warehouse
        weight_init_method: Method for initializing SKU weights ("random" or "uniform")
        cost_calculation_method: Method for calculating cost ("manhattan" or "shortest_path")
        removal_operator: Removal operator for task allocation
        repair_operator: Repair operator for task allocation
        acceptance_function: Acceptance function for LNS (greedy or simulated_annealing)
        T_0: Initial temperature for simulated annealing
        alpha: Temperature decay rate for simulated annealing
        deadline_generation_method: Method for generating task deadlines (e.g., constant, normal, bimodal, etc.)
        deadline_offset: Offset for task deadlines
        output_intermediate_data: Whether to output intermediate data
        intermediate_data_interval: Interval for outputting intermediate data
        base_cost_weight: Weight for base cost
        deadline_weight: Weight for deadline
        sku_distribution_weight: Weight for sku distribution
        agent_unallocated_penalty: Weight for agent unallocated penalty
        solution_repair_detection_function: Function to detect solution repair needs
        solution_repair_function: Function to repair solution post allocation and motion planning
    """
    np.random.seed(seed)
    
    stripped_map_name = map_name.split("/")[-1].replace(".json", "")
    
    output_file = f"data/raw_data/{T}_{task_generation_strategy}_{initial_inventory}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}_{path_planning_strategy}_{stripped_map_name}_{num_robots}_{max_number_tasks}_{base_cost_weight}_{deadline_weight}_{sku_distribution_weight}_{solution_repair_detection_function}_{solution_repair_function}_{seed}.json"
    buffer_file = f"data/buffer_data/{T}_{task_generation_strategy}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}_{path_planning_strategy}_{stripped_map_name}_{num_robots}_{max_number_tasks}_{base_cost_weight}_{deadline_weight}_{sku_distribution_weight}_{seed}"
    
    # B = buffer.Buffer(80, buffer_file)
    S = statistics.Stats(
        num_robots=num_robots,
        simulation_time=T,
        output_file=output_file,
        map_name=map_name,
        cost_calculation_method=cost_calculation_method,
        seed=seed,
        max_tasks=max_number_tasks,
        task_generation_strategy=task_generation_strategy,
        initial_task_assignment_strategy=initial_task_assignment_strategy,
        improvement_task_assignment_strategy=improvement_task_assignment_strategy,
        path_planning_strategy=path_planning_strategy,
        time_limit=time_limit,
        visualize_output=visualize_output,
        initial_inventory=initial_inventory,
        frequency=frequency,
        inbound_outbound_ratio=inbound_outbound_ratio,
        output_graphs=output_graphs,
        num_skus=num_skus,
        weight_init_method=weight_init_method,
        removal_operator=removal_operator,
        repair_operator=repair_operator,
        acceptance_function=acceptance_function,
        T_0=T_0,
        alpha=alpha,
        deadline_generation_method=deadline_generation_method,
        deadline_offset=deadline_offset,
        output_intermediate_data=output_intermediate_data,
        intermediate_data_interval=intermediate_data_interval,
        base_cost_weight=base_cost_weight,
        deadline_weight=deadline_weight,
        sku_distribution_weight=sku_distribution_weight,
        agent_unallocated_penalty=agent_unallocated_penalty,
        solution_repair_detection_function=solution_repair_detection_function,
        solution_repair_function=solution_repair_function
    )
    G = graph.Graph(num_robots, map_name, initial_inventory, num_skus, weight_init_method)

    # print(f"Number of full locations in warehouse: {len(G.warehouse.get_full_locations())}")
    # exit()

    # Initialize state of robots (robot_id, state)
    robots = []
    for robot_id, location in enumerate(G.get_all_occupied()):
        robots.append(agent.Agent(robot_id, location))
    
    Rs = agent.AgentLoader(robots)
    
    init_locations = [agent.state for agent in Rs.agents]
    S.add_paths(init_locations)
    
    tik = time.time()
    # Execute online algorithm
    execute(S, map_name, Rs, G, frequency, inbound_outbound_ratio, T, 
            case_request_strategy=task_generation_strategy, 
            max_task_number=max_number_tasks, 
            initial_task_assignment_strategy=initial_task_assignment_strategy, 
            improvement_task_assignment_strategy=improvement_task_assignment_strategy, 
            path_planning_strategy=path_planning_strategy, 
            time_limit=time_limit,
            cost_calculation_method=cost_calculation_method,
            removal_operator=removal_operator,
            repair_operator=repair_operator,
            acceptance_function=acceptance_function,
            T_0=T_0,
            alpha=alpha,
            deadline_generation_method=deadline_generation_method,
            deadline_offset=deadline_offset,
            output_intermediate_data=output_intermediate_data,
            intermediate_data_interval=intermediate_data_interval,
            base_cost_weight=base_cost_weight,
            deadline_weight=deadline_weight,
            sku_distribution_weight=sku_distribution_weight,
            agent_unallocated_penalty=agent_unallocated_penalty,
            solution_repair_detection_function=solution_repair_detection_function,
            solution_repair_function=solution_repair_function,
            initial_inventory=initial_inventory,
            shuffle_percentage=shuffle_percentage,
            aisle_dual_cycle=aisle_dual_cycle,
            driveway_dual_cycle=driveway_dual_cycle,
    )
    tok = time.time()
    S.set_total_runtime(tok-tik)
    
    S.save_data()
    
    folder_name = f"{T}_{task_generation_strategy}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}_{path_planning_strategy}_{num_robots}_{max_number_tasks}_{deadline_generation_method}_{base_cost_weight}_{deadline_weight}_{sku_distribution_weight}_{seed}"
    if output_graphs:
        S.output_graphs(folder_name)
    
    if visualize_output:
        print("============================Visualizing Output============================")
        visualize.main((G.width, G.height), G.obstacles, S.return_full_paths(), 
                      f'data/videos/{path_planning_strategy}_{T}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}.mp4', 
                      speed=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run grid world simulation with specified parameters')
    
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--num-robots', type=int, required=True, help='Number of robots')
    parser.add_argument('--time-horizon', type=int, required=True, help='Time horizon T')
    parser.add_argument('--max-tasks', type=int, required=True, help='Maximum number of tasks')
    parser.add_argument('--task-gen-strategy', type=str, required=True, 
                       choices=['informed_uniform', 'uninformed_uniform', 'feedback_control'],
                       help='Task generation strategy')
    parser.add_argument('--initial-task-assign-strategy', type=str, required=True,
                       choices=['cost_matrix', 'random', 'greedy', 'randomized_greedy', 'FCF', 'max_regret_FC', 'randomized_max_regret_FC', 'fast_greedy', 'fast_FCF', 'fast_SCF'],
                       help='Task assignment strategy')
    parser.add_argument('--improvement-task-assign-strategy', type=str, required=True,
                       choices=['py_lns', 'c_lns', 'c_p_lns', 'c_rmca', 'none'],
                       help='Task assignment strategy for improvement')
    parser.add_argument('--path-planning-strategy', type=str, required=True,
                       choices=['ecbs', 'pbs'],
                       help='Path planning strategy')
    parser.add_argument('--map', type=str, default='data/maps/symbotic_small',
                       help='Map file path')
    parser.add_argument('--time-limit', type=int, default=999999,
                       help='Maximum runtime in seconds')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization output')
    parser.add_argument('--initial-inventory', type=float, default=25.0,
                       help='Initial inventory fill percentage (0.0 to 100.0)')
    parser.add_argument('--frequency', type=float, default=1.0,
                       help='Frequency of task generation')
    parser.add_argument('--inbound-outbound-ratio', type=float, default=1.0,
                       help='Ratio of inbound to outbound tasks')
    parser.add_argument('--output-graphs', action='store_true',
                       help='Output analysis graphs')
    parser.add_argument('--num-skus', type=int, default=10,
                       help='Number of unique SKUs in the warehouse')
    parser.add_argument('--weight-init-method', type=str, default='random',
                       choices=['random', 'uniform'],
                       help='Method for initializing SKU weights')
    parser.add_argument('--cost-calculation-method', type=str, default='manhattan',
                       choices=['manhattan', 'shortest_path'],
                       help='Method for calculating cost')
    parser.add_argument('--removal-operator', type=str, default='worst',
                       choices=['worst', 'random', 'greedy', 'shaw'],
                       help='Removal operator for task allocation')
    parser.add_argument('--repair-operator', type=str, default='greedy',
                       choices=['greedy', 'random', 'worst', 'fast_SCF'],
                       help='Repair operator for task allocation')
    parser.add_argument('--acceptance-function', type=str, default='greedy', choices=['greedy', 'simulated_annealing'], help='Acceptance function for LNS (greedy or simulated_annealing)')
    parser.add_argument('--T-0', type=float, default=1.0, help='Initial temperature for simulated annealing')
    parser.add_argument('--alpha', type=float, default=0.99, help='Temperature decay rate for simulated annealing')
    parser.add_argument('--deadline-generation-method', type=str, default='constant',
                       help='Method for generating task deadlines (e.g., constant, normal, bimodal, etc.)', choices=['constant', 'normal', 'bimodal', 'none'])
    parser.add_argument('--deadline-offset', type=float, default=30, help='Offset for task deadlines')
    parser.add_argument('--output-intermediate-data', action='store_true',
                       help='Output intermediate data')
    parser.add_argument('--intermediate-data-interval', type=int, default=1800, help='Interval for outputting intermediate data')
    parser.add_argument('--base-cost-weight', type=float, default=1.0, help='Weight for base cost')
    parser.add_argument('--deadline-weight', type=float, default=0.0, help='Weight for deadline')
    parser.add_argument('--sku-distribution-weight', type=float, default=0.0, help='Weight for sku distribution')
    parser.add_argument('--agent-unallocated-penalty', type=float, default=0.0, help='Penalty for unallocated agents')
    parser.add_argument('--solution-repair-detection-function', type=str, default='none', help='Solution repair detection function')
    parser.add_argument('--solution-repair-function', type=str, default='none', help='Solution repair function')
    parser.add_argument('--shuffle-percentage', type=float, default=0.0,
                       help='Per-task probability (0.0-1.0) of generating a shuffle (type=2, '
                            'shelf-to-shelf) task in CRG. 0.0 disables shuffle generation. '
                            'This is the 1.4-skeleton on/off switch; the proper rearrangement '
                            'ratio balancer is roadmap section 3.2.')
    parser.add_argument('--aisle-dual-cycle', action='store_true',
                       help='Enable aisle dual cycling (IB->OB chaining within the same '
                            'warehouse aisle). When set, after an agent completes an inbound '
                            'task it will immediately pick up a same-aisle outbound task if '
                            'one is unallocated, instead of returning to the Free state. '
                            'Default off so it can be ablated independently of rearrangement.')
    parser.add_argument('--driveway-dual-cycle', action='store_true',
                       help='Enable driveway dual cycling (OB->IB chaining at the driveway). '
                            'When set, after an agent completes an outbound delivery at a '
                            'driveway cell it will immediately pick up an unallocated inbound '
                            'task whose pickup is at any driveway cell. Default off.')
    args = parser.parse_args()
    
    main(
        seed=args.seed,
        num_robots=args.num_robots,
        T=args.time_horizon,
        max_number_tasks=args.max_tasks,
        task_generation_strategy=args.task_gen_strategy,
        initial_task_assignment_strategy=args.initial_task_assign_strategy,
        improvement_task_assignment_strategy=args.improvement_task_assign_strategy,
        path_planning_strategy=args.path_planning_strategy,
        map_name=args.map,
        time_limit=args.time_limit,
        visualize_output=args.visualize,
        initial_inventory=args.initial_inventory,
        frequency=args.frequency,
        inbound_outbound_ratio=args.inbound_outbound_ratio,
        output_graphs=args.output_graphs,
        num_skus=args.num_skus,
        weight_init_method=args.weight_init_method,
        cost_calculation_method=args.cost_calculation_method,
        removal_operator=args.removal_operator,
        repair_operator=args.repair_operator,
        acceptance_function=args.acceptance_function,
        T_0=args.T_0,
        alpha=args.alpha,
        deadline_generation_method=args.deadline_generation_method,
        deadline_offset=args.deadline_offset,
        output_intermediate_data=args.output_intermediate_data,
        intermediate_data_interval=args.intermediate_data_interval,
        base_cost_weight=args.base_cost_weight,
        deadline_weight=args.deadline_weight,
        sku_distribution_weight=args.sku_distribution_weight,
        agent_unallocated_penalty=args.agent_unallocated_penalty,
        solution_repair_detection_function=args.solution_repair_detection_function,
        solution_repair_function=args.solution_repair_function,
        shuffle_percentage=args.shuffle_percentage,
        aisle_dual_cycle=args.aisle_dual_cycle,
        driveway_dual_cycle=args.driveway_dual_cycle,
    )