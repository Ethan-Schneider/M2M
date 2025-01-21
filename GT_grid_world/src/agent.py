class Agent:
    def __init__(self, agent_id : int, state : tuple, task_sequence : list = None):
        self.id = agent_id
        self.task_sequence = task_sequence if task_sequence is not None else []
        self.path_sequence = []
        self.state = state
        self.free_agent = True
        self.active_on_task = False

    def set_active_on_task(self, status : bool):
        self.active_on_task = status
    
    def set_agent_id(self, agent_id):
        self.id = agent_id

    def get_agent_id(self):
        return self.id

    def set_task_sequence(self, task_sequences : list):
        self.task_sequences = task_sequences
    
    def get_assigned_tasks(self) -> list:
        return [task_sequence[0] for task_sequence in self.task_sequence]
    
    def set_state(self, state : tuple):
        self.state = state
    
    def get_state(self):
        return self.state
    
    def set_free_agent(self, free_agent : bool):
        self.free_agent = free_agent
        
    def is_free_agent(self):
        return self.free_agent
    
    def set_path_sequences(self, path_sequences : list):
        self.path_sequence = path_sequences
        
    def add_to_path_sequences(self, path_sequence):
        self.path_sequence.append(path_sequence)
    
    def __str__(self):
        return f"Agent ID: {self.id}, State: {self.state}"
    
    
class AgentLoader:
    def __init__(self, agents : list):
        self.num_agents = len(agents)
        self.agents = agents
        
    def get_agent(self, agent_id : int):
        for agent in self.agents:
            if agent.get_agent_id() == agent_id:
                return agent
        return None

    def get_free_agents(self):
        return [agent for agent in self.agents if agent.is_free_agent()]
    
    def get_busy_agents(self):
        return [agent for agent in self.agents if not agent.is_free_agent()] 
    
    def get_all_assigned_tasks(self):
        return [agent.get_assigned_tasks() for agent in self.agents]   

    def __str__(self):
        return f"AgentLoader with {len(self.agents)} agents out of {self.num_agents} allowed"
