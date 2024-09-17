import numpy as np
    
def CRG(J: set, G : Graph, N: int, inbound_to_outbound: float, last_task_id: int, strategy: str = "uniform", item : list = []) -> set:
    """_summary_

    Args:
        J (set): _description_
        G (Graph)
        N (int): _description_
        strategy (str, optional): _description_. Defaults to uniform.
        item (list, optional): _description_. Defaults to [].

    Returns:
        set: _description_
    """
    inbound_probability = inbound_to_outbound/(inbound_to_outbound + 1)
    outbound_probability = 1 - inbound_probability
    tasks_to_generate = np.random.choice([0, 1], size=N, p=[outbound_probability, inbound_probability])
    J_new = set([])
    
    #Generate tasks uniformly throughout the warehouse without inventory information
    if strategy == "uniform":
        for task in tasks_to_generate:
            #Generate inbound task
            if task == 1:
                J_new = J_new | set([(last_task_id + 1, np.random.choice(np.arange(DW.max_pos)), np.random.choice(np.arange(W.max_pos)))])
                last_task_id += 1
            #Generate outbound task
            elif task == 0:
                J_new = J_new | set([(last_task_id + 1, np.random.choice(np.arange(W.max_pos)), np.random.choice(np.arange(DW.max_pos)))])
                last_task_id += 1
    return J_new, last_task_id
    