import numpy as np

from typing import List, Tuple

from ...agent import AgentLoader
from ...graph import Graph

def destroy_start_loc(Rs : AgentLoader, reallocation_group : list, allocated_locations : set) -> Tuple[AgentLoader, List]:
    """randomly select a subset of agents from the reallocation group and remove the first task's start location from their task sequence

    Args:
        Rs (AgentLoader): The agent loader containing all agents.
        reallocation_group (list): List of agent IDs to consider for reallocation.
        allocated_locations (set): Set of currently allocated locations.

    Returns:
        AgentLoader: A new AgentLoader instance with the first task removed from each agent's task sequence.
        modified_agents (list): List of agent IDs whose task sequences were modified.
    """
    
    # Modify temporary reallocation group to remove agents whose status is 2 (to_delivery)
    temp_reallocation_group = []
    for agent_id in reallocation_group:
        agent = Rs.agents[agent_id]
        if agent.status != 2:
            temp_reallocation_group.append(agent_id)
            
    if len(temp_reallocation_group) == 0:
        return Rs, [], allocated_locations

    # Select half of the agents in the reallocation group randomly rounded up
    num_to_remove = int(np.ceil((len(temp_reallocation_group) + 1) / 2))
    # num_to_remove = 2
    print(num_to_remove)
    if num_to_remove == 0:
        return Rs, [], allocated_locations
    agents_to_modify = np.random.choice(temp_reallocation_group, size=num_to_remove, replace=False)

    for agent_id in agents_to_modify:
        agent = Rs.get_agent(agent_id)
        if agent.task_sequence:
            allocated_locations.discard(agent.task_sequence[0][1])  # Remove start location from allocated locations
            agent.task_sequence[0] = (agent.task_sequence[0][0], None, agent.task_sequence[0][2], agent.task_sequence[0][3])  # Remove start location
    
    return Rs, agents_to_modify, allocated_locations

def destroy_end_loc(Rs : AgentLoader, reallocation_group : list, allocated_locations : set) -> Tuple[AgentLoader, List]:
    """randomly select a subset of agents from the reallocation group and remove the first task's end location from their task sequence

    Args:
        Rs (AgentLoader): The agent loader containing all agents.
        reallocation_group (list): List of agent IDs to consider for reallocation.
        allocated_locations (set): Set of currently allocated locations.
    Returns:
        AgentLoader: A new AgentLoader instance with the first task removed from each agent's task sequence.
        modified_agents (list): List of agent IDs whose task sequences were modified.
    """
    
    # Select half of the agents in the reallocation group randomly rounded up
    num_to_remove = int(np.ceil((len(reallocation_group) + 1) / 2))
    # num_to_remove = 2
    if num_to_remove == 0:
        return Rs, [], allocated_locations
    agents_to_modify = np.random.choice(reallocation_group, size=num_to_remove, replace=False)

    for agent_id in agents_to_modify:
        agent = Rs.agents[agent_id]
        if agent.task_sequence:
            allocated_locations.discard(agent.task_sequence[0][2])  # Remove end location from allocated locations
            agent.task_sequence[0] = (agent.task_sequence[0][0], agent.task_sequence[0][1], None, agent.task_sequence[0][3])  # Remove end location

    return Rs, agents_to_modify, allocated_locations

def swap_tasks(Rs : AgentLoader, reallocation_group : list, allocated_locations : list) -> Tuple[AgentLoader, List]:
    """randomly select two agents from the reallocation group and swap their first tasks

    Args:
        Rs (AgentLoader): The agent loader containing all agents.
        reallocation_group (list): List of agent IDs to consider for reallocation.

    Returns:
        AgentLoader: A new AgentLoader instance with the first tasks swapped between two agents.
        modified_agents (list): List of agent IDs whose task sequences were modified.
    """
    
    if len(reallocation_group) < 2:
        return Rs, [], allocated_locations

    # Modify temporary reallocation group to remove agents whose status is 2 (to_delivery)
    temp_reallocation_group = []
    for agent_id in reallocation_group:
        agent = Rs.get_agent(agent_id)
        if agent.status != 2:
            temp_reallocation_group.append(agent_id)
    
    if len(temp_reallocation_group) < 2:
        return Rs, [], allocated_locations

    # Randomly select two distinct agents from the reallocation group
    agent_ids = np.random.choice(temp_reallocation_group, size=2, replace=False)
    agent1 = Rs.agents[agent_ids[0]]
    agent2 = Rs.agents[agent_ids[1]]

    # Swap their first tasks if both have at least one task
    if agent1.task_sequence and agent2.task_sequence:
        agent1.task_sequence[0], agent2.task_sequence[0] = agent2.task_sequence[0], agent1.task_sequence[0]
        modified_agents = [agent_ids[0], agent_ids[1]]
    else:
        modified_agents = []

    return Rs, modified_agents, allocated_locations
    