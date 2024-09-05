from src import case_request_generator, router, simulate, task_allocation

def main():
    pass
    
def execute(I, frequency, T, case_request_strategy):
    Rs, W = I
    J = set()
    Ra = []
    for t in T:
        
        # Check if new tasks need to be generated
        if t%frequency == 0:
            if frequency > 0: 
                N = 1
            elif frequency < 0: 
                N = frequency**-1
            else:
                N = 0
            # Generate new tasks
            J = J | case_request_generator.CRG(W, N, case_request_strategy)

if __name__=="__main__":
    main()