class Agent:
    def __init__(self, agent_id : int, state : tuple, task_sequence : list = None, home : tuple = None):
        self.id = agent_id
        self.task_sequence = task_sequence if task_sequence is not None else []
        self.path_sequence = []
        self.state = state
        if home is None:
            self.home = (state[0], state[1])
        else:
            self.home = home
        # 0 == Free Agent
        # 1 == To_Pickup
        # 2 == To_Delivery
        # 3 == Picking (waiting at pickup location)
        # 4 == Placing (waiting at delivery location)
        self.status = 0
        self.sku_id_carrying = None
        self.pick_place_counter = 0
        self.blocked_ticks = 0
        # True while the agent is waiting at the buffer after being rejected
        # from placing an item there; cleared once it begins placing.
        self.waiting_at_buffer = False
    
    def set_sku_id_carrying(self, sku_id : int):
        self.sku_id_carrying = sku_id
    
    def get_sku_id_carrying(self) -> int:
        return self.sku_id_carrying

    def set_active_on_task(self, status : int):
        """Sets Status of the Agent
        0 == Free Agent
        1 == To_Pickup
        2 == To_Delivery
        3 == Picking
        4 == Placing

        Args:
            status (int): Current status of agent
        """
        self.active_on_task = status
    
    def set_agent_id(self, agent_id):
        self.id = agent_id

    def get_agent_id(self):
        return self.id

    def set_task_sequence(self, task_sequences : list):
        self.task_sequences = task_sequences
    
    def get_assigned_task_ids(self) -> list:
        return [task_id for task_id in self.task_sequence]
    
    def set_path_sequences(self, path_sequences : list):
        self.path_sequence = path_sequences
        
    def add_to_path_sequences(self, path_sequence):
        self.path_sequence.append(path_sequence)
    
    def __str__(self):
        return f"Agent ID: {self.id}, State: {self.state}"
    
    
class AgentLoader:
    def __init__(self, agents : list):
        self.agents = agents
        
    def get_agent(self, agent_id) -> Agent:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        
    def get_agent_states(self) -> list:
        return [agent.state for agent in self.agents]

    def get_free_agents(self):
        return [agent for agent in self.agents if agent.status == 0]
    
    def get_to_pickup_agents(self):
        return [agent for agent in self.agents if agent.status == 1] 
    
    def get_to_delivery_agents(self):
        return [agent for agent in self.agents if agent.status == 2]
    
    def get_all_assigned_tasks_per_robot(self):
        return [agent.get_assigned_task_ids() for agent in self.agents]   
    
    def get_all_agent_carrying_skus(self) -> list:
        skus = []
        for agent in self.agents:
            skus.append(agent.get_sku_id_carrying())
        
        return skus
    
    def get_all_assigned_tasks(self): 
        tasks = []
        for agent in self.agents:
            for task in agent.task_sequence:
                tasks.append(task)
        return tasks
    
    def get_assigned_agent(self, task_id) -> int:
        for agent in self.agents:
            if task_id in agent.task_sequence:
                return agent.id
    
    def detect_collisions(self) -> int:
        return len(self.agents) - len(set(self.get_agent_states()))

    def __str__(self):
        return f"AgentLoader with {len(self.agents)} agents out of {len(self.agents)} allowed"
    
    def copy(self):
        """Create a deep copy of the current solution."""
        new_solution = AgentLoader([])
        for agent in self.agents:
            new_agent = agent.__class__(agent.id, agent.state, task_sequence=agent.task_sequence.copy(), home=agent.home)
            new_agent.status = agent.status
            new_agent.path_sequence = agent.path_sequence.copy()
            new_agent.sku_id_carrying = agent.sku_id_carrying
            new_agent.pick_place_counter = agent.pick_place_counter
            new_agent.blocked_ticks = agent.blocked_ticks
            new_agent.waiting_at_buffer = agent.waiting_at_buffer
            new_solution.agents.append(new_agent)
        return new_solution
