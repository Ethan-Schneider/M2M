import os

from src import graph, router, simulate, task_allocation, case_request_generator

def main():
    #Initilize state of robots
    Rs_init = [(0, (80, 100), True, -1)]
    
    G = graph.Graph(8, "GT_grid_world/src/maps/symbotic_small", 4, True, "uniform", 20.)
    
    # G.warehouse.printInventory()
    print(len(G.warehouse.findFull())/G.warehouse.max_pos)
    
    # Init values for task frequency and ratio, total number of timesteps, etc. 
    frequency = 1
    inbound_outbound_ratio = 1.0
    T = 100
    task_generation_strategy = "informed_uniform"
    
    
    # Execute online algorithm
    execute((Rs_init, G), frequency, inbound_outbound_ratio, T)
    
    
def execute(I: tuple, frequency, inbound_to_outbound_ratio: float, T: int, case_request_strategy: str = "uninformed_uniform"):
    Rs, G = I
    J = set()
    Ra = []
    
    last_task_id = 0
    
    for t in range(T):
        # Check if new tasks need to be generated
        if t%frequency == 0:
            if frequency > 0: 
                N = 1
            elif frequency < 0: 
                N = frequency**-1
            else:
                N = 0
            # Generate new tasks
            J_new, last_task_id = case_request_generator.CRG(J, G, N, inbound_to_outbound_ratio, last_task_id, case_request_strategy)
            J = J | J_new
    print(J)
    print(len(J))

if __name__=="__main__":
    main()