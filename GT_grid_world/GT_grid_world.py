from src import case_request_generator, router, simulate, task_allocation

def main():
    #Initilize state of robots
    Rs_init = [(0, (80, 100), True, -1)]
    
    # Initilize state of the warehouse
    warehouse = case_request_generator.Warehouse(2660)
    warehouse.readData()

    # Initilize state of the driveway    
    driveway = case_request_generator.Warehouse(75)
    
    # Init values for task frequency and ratio, total number of timesteps, etc. 
    frequency = 1
    inbound_outbound_ratio = 1.0
    T = 100
    
    
    # Execute online algorithm
    execute((Rs_init, warehouse, driveway), frequency, inbound_outbound_ratio, T)
    
    
def execute(I: tuple, frequency, inbound_to_outbound_ratio: float, T: int, case_request_strategy: str = "uniform"):
    Rs, W, DW = I
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
            J_new, last_task_id = case_request_generator.CRG(J, W, DW, N, inbound_to_outbound_ratio, last_task_id, case_request_strategy)
            J = J | J_new
    print(J)
    print(len(J))

if __name__=="__main__":
    main()