from .statistics import Stats
from ..utils import *

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
