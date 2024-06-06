import numpy as np

def random_assignment(caserequests: dict, robots: list):
    """_summary_

    Args:
        caserequests (dict): _description_
        robots (list): _description_

    Returns:
        _type_: _description_
    """
    
    A = np.zeros(shape=(len(robots), len(caserequests)))
    
    # Loop over every task
    for task in range(A.shape[1]):
        agent = np.random.randint(0, A.shape[0])
        A[agent][task] = 1
    
    return A