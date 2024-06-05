from case_request import generate_case_requets
from generate_robots import generate_robots

def task_allocation(num_robots : int, num_tasks : int, inbound_outbound_ratio : float, id_type : str, task_allocator : str = 'random'):
    data = generate_case_requets.generate_case_requests(num_tasks, inbound_outbound_ratio, id_type, "test_data", 0)
    print(data[1])
    robots = generate_robots.generate_robots(num_robots)
    print(robots)

if __name__ == "__main__":
    task_allocation(40, 1000, 1.0, 'random', 'random')