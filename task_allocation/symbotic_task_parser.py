import numpy as np
import json

def main():
    data = []
    tasks = []
    file = open('/home/Ethan/Documents/Symbotic/symbotic_tamp/task_allocation/taskList2.txt', "r")
    current_task_id = 1
    while True:
        content = file.readline()
        if not content:
            tasks.append(((data[-2]["Location"]["X"], data[-2]["Location"]["Y"]), (data[-1]["Location"]["X"], data[-1]["Location"]["Y"])))
            break
        content = json.loads(content)
        if current_task_id != content["SRETaskID"]:
            tasks.append(((data[-2]["Location"]["X"], data[-2]["Location"]["Y"]), (data[-1]["Location"]["X"], data[-1]["Location"]["Y"])))
            current_task_id = content["SRETaskID"]
        data.append(content)
    file.close()
    
    print(tasks)
        

if __name__=="__main__":
    main()