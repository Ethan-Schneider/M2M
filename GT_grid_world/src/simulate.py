from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *


def simulate(S : Stats, G : Graph, Rs : AgentLoader, J : set):
    # Update state of robots
    for agent in Rs.agents:
        # If robot sequence is stationary, leave the robot in place (wait action)
        # BUG: crashes when router fails to find a solution 
        if len(agent.path_sequence) == 0:
            continue
        # If robot does have a sequence of actions, pop next state and update
        else:
            # Update Agent State and Graph Occupied States
            old_state = agent.state
            G.set_occupied(agent.state, False)
            agent.state = agent.path_sequence.pop(0)
            G.set_occupied(agent.state, True)
            
            if agent.status == 1:
                S.update_actual_pickup_distance(agent.task_sequence[0], euclidian_distance(old_state, agent.state))
                S.update_actual_pickup_duration(agent.task_sequence[0], euclidian_distance(old_state, agent.state))
            elif agent.status == 2:
                S.update_actual_distance(agent.task_sequence[0], euclidian_distance(old_state, agent.state))
                S.update_actual_duration(agent.task_sequence[0], euclidian_distance(old_state, agent.state))
            else:
                pass
    
    S.append_number_of_collisions(Rs.detect_collisions())
    
    # Update Statistics for the total paths taken
    S.add_paths(Rs.get_agent_states())
    
    # Update Statistics for Asile and Driveway Occupancy
    S.add_aisle_occupancy(G.get_aisle_occupancy())
    S.add_driveway_occupancy(G.get_driveway_occupancy())
    # S.add_aisle_occupancy(G.get_aisle_occupied())
    # S.add_driveway_occupancy(G.get_driveway_occupied())
    
    # Update free_agents, to_pickup, and to_delivery
    # TODO: Update warehouse and driveway inventory when a robot has reached goal location (i.e. for both for loops below) 
          
    for agent in Rs.agents:
        if agent.status == 1:
            if agent.state == get_task_start_location(J, agent.task_sequence[0]):
                S.add_completed_to_pickup_task_id(agent.task_sequence[0])
                agent.status == 2
        elif agent.status == 2:
            if agent.state == get_task_goal_location(J, agent.task_sequence[0]):
                S.add_completed_task_id(agent.task_sequence[0])
                agent.status = 0
                J.remove(agent.task_sequence[0])
                agent.task_sequence.pop(0)
            
    return Rs, J