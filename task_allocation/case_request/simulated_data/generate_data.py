#!/usr/bin/env python3

import numpy as np

def generate_data(num_data: int, outbound_to_inbound_ratio: float, id_type: str):
    """Generate synthetic inbound and outbound tasks.

    Args:
        num_data (int): number of tasks to generate
        outbound_to_inbound_ratio (float): ratio of the number of outbound tasks to inbound tasks generated
        id_type (str): "random": random id assignment
        file_name (str): json output file name
        
    """
    #Initilization: dataset and tasks (ID, I/O)
    dataset = {}
    tasks = generate_tasks(num_data, outbound_to_inbound_ratio)
    
    for taskID, outbound_inbound in tasks:
        dataset[int(taskID)] = {'outbound_inbound': outbound_inbound}
    
    return dataset
    
    
def generate_tasks(num_data: int, outbound_to_inbound_ratio: float):
    N = np.arange(num_data)
    I = set(np.random.choice(N, int(num_data//(1+outbound_to_inbound_ratio)), replace=False))
    O = set(N) - I
    
    I = [(i, 1) for i in I]
    O = [(o, 0) for o in O]
    
    return (I + O)


if __name__ == '__main__':
    generate_data(20, 1, 'random', 'test_set')
    print(generate_data(20, 1, 'random'))