from .statistics import Stats
from ..utils import *
from ..agent import AgentLoader

from ..task_allocation_algorithms.external_algorithms.lns import dc

import json

class Buffer(object):
    def __init__(self, buffer_size : int, filename : str):
        self.buffer_size = buffer_size
        self.state_buffer = []
        self.time = []
        
        self.filename = filename

    def add(self, state : list, t : int) -> None:
        if len(self.state_buffer) >= self.buffer_size:
            self.state_buffer.pop(0)
            self.time.pop(0)
        self.time.append(t)
        self.state_buffer.append(state)
        
    def dump(self, agent_id : int, task_id : int, J : set, S : Stats):
        data = {
            "timesteps" : self.time,
            "agent" : agent_id,
            "task" : task_id,
            "task start location" : get_task_start_location(J, task_id),
            "task goal location" : get_task_goal_location(J, task_id),
            "deadline" : next((task[3] for task in J if task[0] == task_id), None),
            "estimated task duration" : S.get_estimated_duration(task_id),
            "estimated to-pickup duration" : S.get_estimated_pickup_duration(task_id),
            "estimated total duration" : S.get_estimated_duration(task_id) + S.get_estimated_pickup_duration(task_id),
            "actual task duration" : S.get_actual_duration(task_id),
            "actual to-pickup duration" : S.get_actual_pickup_duration(task_id),
            "actual total duration" : S.get_actual_duration(task_id) + S.get_actual_pickup_duration(task_id),
            "states" : self.state_buffer
        }
        
        filename = self.filename + f"_{agent_id}_{task_id}.json"
        
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    def compute_allocation_dif(self, Rs : AgentLoader, J : set):
        map_name = "GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns/maps/symbotic_small.map"
        
        tau_extras = []
        tau_extra_assigned_cost = []
        tau_unassigned_agent_costs = []
        agent_task_sequences = []
        agent_task_sequence_cost = []
        
        unassigned_agents = Rs.get_free_agents()
        unassigned_agent_ids = [agent.id for agent in unassigned_agents]
        
        for agent in Rs.agents:
            temp_agent_task_sequence = []
            temp_agent_task_sequence_cost = []
            for seq_num, task in enumerate(agent.task_sequence):
                task_id = task[0]
                if seq_num == 0:
                    temp_agent_task_sequence.append(task_id)
                    cost = dc.distance(map_name, agent.state, get_task_start_location(J, task_id)) + dc.distance(map_name, get_task_start_location(J, task_id), get_task_goal_location(J, task_id))
                    temp_agent_task_sequence_cost.append(cost)
                    continue
                else:
                    temp_agent_task_sequence.append(task_id)
                    tau_extras.append(task_id)
                    cost = dc.distance(map_name, get_task_goal_location(J, agent.task_sequence[seq_num - 1]), get_task_start_location(J, task_id)) + dc.distance(map_name, get_task_start_location(J, task_id), get_task_goal_location(J, task_id))
                    tau_extra_assigned_cost.append(cost)
                    temp_agent_task_sequence_cost.append(cost)
                    
                    temp_costs = []
                    for unassigned_agent in unassigned_agents:
                        cost = dc.distance(map_name, unassigned_agent.state, get_task_start_location(J, task_id)) + dc.distance(map_name, get_task_start_location(J, task_id), get_task_goal_location(J, task_id))
                        temp_costs.append(cost)
                    tau_unassigned_agent_costs.append(temp_costs)
            agent_task_sequences.append(temp_agent_task_sequence)
            agent_task_sequence_cost.append(temp_agent_task_sequence_cost)
                    
        return tau_extras, tau_extra_assigned_cost, tau_unassigned_agent_costs, unassigned_agent_ids, agent_task_sequences, agent_task_sequence_cost
            
    def dump_lns_allocation(self, t : int, Rs : AgentLoader, J : set):
        tau_extras, tau_extra_assigned_cost, tau_unassigned_agent_costs, unassigned_agent_ids, agent_task_sequences, agent_task_sequence_cost = self.compute_allocation_dif(Rs, J)
        
        
        
        data = {
            "timestep" : t,
            "extra task ids" : tau_extras,
            "extra task assigned costs" : tau_extra_assigned_cost,
            "extra tasks possible costs" : tau_unassigned_agent_costs,
            "unassigned agent ids" : unassigned_agent_ids,
            "agent task sequences" : agent_task_sequences,
            "agent task sequence costs" : agent_task_sequence_cost,
            "tasks" : list(J),
            "states" : self.state_buffer
        }
        
        filename = filename = self.filename + f"_{t}_unallocated.json"
        
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
