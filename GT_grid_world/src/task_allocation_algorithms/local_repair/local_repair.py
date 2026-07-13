import numpy as np
import time
import math
import random

from ...graph import Graph
from ...agent import AgentLoader
from ...analysis.statistics import Stats

from .destroy_operators import destroy_start_loc, destroy_end_loc, swap_tasks
from .repair_operators import greedy_repair

def local_repair(Rs : AgentLoader, G : Graph, S : Stats, J : dict, reallocation_group : list) -> AgentLoader:
    """Perform local repair for the first task in each agent's task sequence in the reallocation group via
    Large Neighborhood Search (LNS).

    Args:
        Rs (AgentLoader): The agent loader containing all agents.
        G (Graph): The graph representing the environment.
        S (Stats): The statistics object to record performance metrics.
        J (set): Set of tasks
        reallocation_group (list): List of agent IDs to consider for reallocation.
    """
    
    # Collect currently allocated locations for all agents and tasks
    allocated_locations = set()
    for agent in Rs.agents:
        if agent.task_sequence:
            task = agent.task_sequence[0]
            allocated_locations.add(task[1])  # Start location
            allocated_locations.add(task[2])  # End location
            
    best_Rs = copy_solution(Rs)
    best_cost = compute_cost(best_Rs, G)
    current_Rs = copy_solution(Rs)
    current_cost = best_cost
    
    current_allocated_locations = allocated_locations
    
    T_0=1.0
    alpha=0.99
    
    T = T_0
    
    tik = time.time()
    tok = tik
    
    print(f"Reallocation group: {reallocation_group}")
    
    while (np.abs(tok-tik) <= 1.0):
        
        # Apply destroy operator, repair operator, compute cost, update, then repeat until time limit reached
        for agent in current_Rs.agents:
            print(f"Agent {agent.id} task sequences before destroy: {agent.task_sequence}")
        
        # Randomly select a destroy operator
        destroy_operator = np.random.choice([destroy_start_loc, destroy_end_loc, swap_tasks])
        new_Rs, modified_agents, new_allocated_locations = destroy_operator(current_Rs, reallocation_group, current_allocated_locations)

        # Print changes to Rs after destroy operator
        print(f"After destroy operator {destroy_operator.__name__}, modified agents: {modified_agents}")
        for agent_id in modified_agents:
            agent = new_Rs.agents[agent_id]
            print(f"Agent {agent.id} task sequence after destroy: {[task for task in agent.task_sequence]}")

        if destroy_operator != swap_tasks:
            # Apply repair operator
            new_Rs, new_allocated_locations = greedy_repair(new_Rs, G, J, reallocation_group, new_allocated_locations)
        
        for agent in new_Rs.agents:
            print(f"Agent {agent.id} task sequences after repair: {agent.task_sequence}")

        new_cost = compute_cost(new_Rs, G)
        if new_cost > best_cost:
            best_Rs = copy_solution(new_Rs)
            current_Rs = copy_solution(new_Rs)
            current_allocated_locations = new_allocated_locations.copy()
        else:
            prob = math.exp(-1*(current_cost - new_cost) / T) if T > 0 else 0
            if random.random() < prob:
                current_Rs = copy_solution(new_Rs)
                current_cost = new_cost
                current_allocated_locations = new_allocated_locations.copy()
            
        T = alpha*T

        tok = time.time()
    
    
    print(f"Reallocation Group: {reallocation_group}")
            
    print(f"Original Task Sequence:")
    for agent in Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
    print(f"New Task Sequences:")
    for agent in best_Rs.agents:
        print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
    # exit()
    
    return Rs


def compute_cost(Rs : AgentLoader, G : Graph) -> float:
    """Compute the total cost for the agent's first allocations in their task sequences.
    This cost is from shortest distance paths between start and end locations. 

    Args:
        Rs (AgentLoader): The agent loader containing all agents.
        G (Graph): The graph representing the environment.

    Returns:
        float: The total cost of the initial task allocations.
    """
    
    total_cost = 0.0
    for agent in Rs.agents:
        if agent.task_sequence:
            task = agent.task_sequence[0]
            start_location = task[1]
            end_location = task[2]
            total_cost += G.get_distance(start_location, end_location)
    return total_cost


def copy_solution(solution: AgentLoader) -> AgentLoader:
    """Create a deep copy of the current solution."""
    new_solution = AgentLoader([])
    for agent in solution.agents:
        new_agent = agent.__class__(agent.id, agent.state, task_sequence=agent.task_sequence.copy(), home=agent.home)
        new_agent.status = agent.status
        new_agent.path_sequence = agent.path_sequence.copy()
        new_agent.sku_id_carrying = agent.sku_id_carrying
        new_agent.pick_place_counter = agent.pick_place_counter
        new_solution.agents.append(new_agent)
    return new_solution
