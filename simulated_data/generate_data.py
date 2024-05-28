#!/usr/bin/env python3

import json
import numpy as np
import datetime

def generate_data(num_data: int, outbound_to_inbound_ratio: float, id_type: str, file_name: str = None):
    """Generate synthetic inbound and outbound tasks.

    Args:
        num_data (int): number of tasks to generate
        outbound_to_inbound_ratio (float): ratio of the number of outbound tasks to inbound tasks generated
        id_type (str): "random": random id assignment
        file_name (str): json output file name
        
    """
    #Initilization
    dataset = {}
    tasks = generate_tasks(num_data, outbound_to_inbound_ratio)
    
    for taskID, outbound_inbound in tasks:
        dataset[int(taskID)] = {'outbound_inbound': outbound_inbound}
    
    __save_to_json(dataset, file_name)
    
    
def generate_tasks(num_data: int, outbound_to_inbound_ratio: float):
    N = np.arange(num_data)
    I = set(np.random.choice(N, int(num_data//(1+outbound_to_inbound_ratio)), replace=False))
    O = set(N) - I
    
    I = [(i, 1) for i in I]
    O = [(o, 0) for o in O]
    
    return (I + O)


def __save_to_json(data: list, file_name):
    if file_name == None:
        file_name = datetime.now()
    file_name =  str(file_name) + ".json"
    
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    generate_data(20, 1, 'random', 'test_set')