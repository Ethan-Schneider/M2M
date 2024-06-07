from case_request import generate_case_requets
from generate_robots import generate_robots

from random_assignment import random_assignment

def task_allocation(num_robots : int, num_tasks : int, inbound_outbound_ratio : float, id_type : str, task_allocator : str = 'random'):
    """Runs task allocation on generated robots and tasks with the listed task allocation methods: 
    
    - random
    - TA-Hybrid

    Args:
        num_robots (int): _description_
        num_tasks (int): _description_
        inbound_outbound_ratio (float): _description_
        id_type (str): _description_
        task_allocator (str, optional): _description_. Defaults to 'random'.

    Raises:
        ValueError: _description_
    """
    
    case_requests = generate_case_requets.generate_case_requests(num_tasks, inbound_outbound_ratio, id_type, "test_data", 0)
    # print(case_requests[1])
    robots = generate_robots.generate_robots(num_robots)
    # print(robots)
    
    match task_allocator:
      
        case 'random':
            A = random_assignment.random_assignment(case_requests, robots)  
        case _:
            raise ValueError('{task_allocator} not a valid task allocator. Please use listed methods in docstring.')


if __name__ == "__main__":
    task_allocation(10, 10, 1.0, 'random', 'random')