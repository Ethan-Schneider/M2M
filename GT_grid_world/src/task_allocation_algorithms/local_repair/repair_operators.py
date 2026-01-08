import numpy as np

from sortedcontainers import SortedList

from ...agent import AgentLoader
from ...graph import Graph

def greedy_repair(Rs : AgentLoader, G : Graph, J : dict, reallocation_group : list, allocated_locations : set) -> AgentLoader:
    """Greedily repair the first task in each agent's task sequence in the reallocation group by assigning the closest available
    start and end locations.

    Args:
        Rs (AgentLoader): The agent loader containing all agents.
        G (Graph): The graph representing the environment.
        reallocation_group (list): List of agent IDs to consider for reallocation.
        allocated_locations (set): Set of currently allocated locations.

    Returns:
        AgentLoader: The updated AgentLoader with repaired task sequences.
    """

    # Construct a sorted list with keys (agent_id, possible_location) and value as distance from Graph G
    # Here the possible location is the possible pickup or dropoff location for the task 
    task_heap = SortedList()
    
    candidate_locations = set()
    
    print(f"Allocated Locations: {allocated_locations}")

    for agent_id in reallocation_group:
        agent = Rs.get_agent(agent_id)
        
        print(f"Agent {agent.id} task sequence before repair: {agent.task_sequence}")
        print(f"Task Details: {J[agent.task_sequence[0][0]]}")
            
        if agent.task_sequence[0][1] is None:
            # Check if task is inbound or outbound
            print(f"HERE: {J[agent.task_sequence[0][0]]}")
            for loc in J[agent.task_sequence[0][0]][0]:  # Inbound task, check all possible start locations
                if loc not in allocated_locations:
                    print(f"loc: {loc}")
                    distance = G.get_distance(agent.state, loc)
                    task_heap.add( (distance, (agent_id, loc)) )
                    candidate_locations.add(loc)
        elif agent.task_sequence[0][2] is None:
            for loc in J[agent.task_sequence[0][0]][1]:  # Outbound task, check all possible end locations
                if loc not in allocated_locations:
                    print(f"loc: {loc}")
                    distance = G.get_distance(agent.state, loc)
                    task_heap.add( (distance, (agent_id, loc)) )
                    candidate_locations.add(loc)
        else:
            continue  # Both locations are already assigned

    print(f"Candidate locations for repair: {candidate_locations}")
    print(f"Task heap for repair: {task_heap}")
    
    while True: 
        print(f"Task heap: {task_heap}")
        if not task_heap:
            return Rs, allocated_locations
        
        task = task_heap.pop(0)
        
        print(f"Task: {task}")
        print(f"Agent id: {task[1][0]}")
        
        print(f"Agent {Rs.get_agent(task[1][0]).id} task sequence {Rs.get_agent(task[1][0]).task_sequence}")
        
        # Check if discard possible task ... 
        
        if task[1][1] in allocated_locations:
            continue
        
        if Rs.get_agent(task[1][0]).task_sequence[0][1] is not None and Rs.get_agent(task[1][0]).task_sequence[0][2] is not None:
            continue
    
        print(f"Task Heap after pop: {task_heap}")
        
        agent_id = task[1][0]
        allocated_locations.add(task[1][1])
        
        print(f"Agent {agent_id} with first task {Rs.get_agent(agent_id).task_sequence[0]}")
        if Rs.get_agent(agent_id).task_sequence[0][1] is None:
            Rs.get_agent(agent_id).task_sequence[0] = (Rs.get_agent(agent_id).task_sequence[0][0], task[1][1], Rs.get_agent(agent_id).task_sequence[0][2], Rs.get_agent(agent_id).task_sequence[0][3])
        elif Rs.get_agent(agent_id).task_sequence[0][2] is None:
            Rs.get_agent(agent_id).task_sequence[0] = (Rs.get_agent(agent_id).task_sequence[0][0], Rs.get_agent(agent_id).task_sequence[0][1], task[1][1], Rs.get_agent(agent_id).task_sequence[0][3])
        else:
            print(f"Error: Agent has no unallocated start/goal location ... ")
            exit()
        
        done = True
        # Check if every agent has been allocated to
        for agent in Rs.agents:
            print(f"Agent {agent.id} task sequence: {agent.task_sequence}")
            if agent.task_sequence:
                if agent.task_sequence[0][1] is None or agent.task_sequence[0][2] is None:
                    done = False
                    break
            
        if done:
            # exit()
            return Rs, allocated_locations