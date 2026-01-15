from ...agent import AgentLoader
from ...graph import Graph

def sliding_window_progress(Rs : AgentLoader, G : Graph) -> dict:
    agents_to_reallocate = []
    W = 5
    
    for agent in Rs.agents:
        if len(agent.path_sequence) < W:
            continue
        
        if agent_.status == 1:
            current_target_location = agent_.task_sequence[0][1]
        elif agent_.status == 2:
            current_target_location = agent_.task_sequence[0][2]
        else:
            continue
            
        d_t = G.get_distance(agent.state, current_target_location)
        
        w_state = agent.path_sequence[W]
        
        d_tW = G.get_distance(w_state, current_target_location)
        
        value = (d_t - d_tW) / W
        
        if value <= 0.4:
            agents_to_reallocate.append(agent)
    
    aisle_groups = {}
    
    for agent_ in agents_to_reallocate:
        # get agent_id's current goal location column
        if Rs.get_agent(agent_.id).status == 1:
            goal_aisle = Rs.get_agent(agent_.id).task_sequence[0][1][1]
        elif Rs.get_agent(agent_.id).status == 2:
            goal_aisle = Rs.get_agent(agent_.id).task_sequence[0][2][1]
        else:
            print(f"Agent {agent_.id}")
        
        if goal_aisle not in aisle_groups.keys():
            aisle_groups[goal_aisle] = [agent_]
        
        for comparison_agent in Rs.agents:
            if comparison_agent.id != agent_.id and comparison_agent.task_sequence:
                if comparison_agent.status == 1:
                    comp_aisle = comparison_agent.task_sequence[0][1][1]
                elif comparison_agent.status == 2:
                    comp_aisle = comparison_agent.task_sequence[0][2][1]
                else:
                    continue
                if goal_aisle == comp_aisle and comparison_agent not in aisle_groups[goal_aisle]:
                    aisle_groups[goal_aisle].append(comparison_agent)
                    
    return aisle_groups