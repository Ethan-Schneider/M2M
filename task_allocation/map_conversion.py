import json
import numpy as np
import xml.etree.ElementTree as ET
import os

def main():
    #Open XML File
    tree = ET.parse('/home/Ethan/Documents/Symbotic/symbotic_tamp/task_allocation/network.xml')

    root = tree.getroot()
    
    #Initialize list of aisle nodes and driveway nodes
    nodes = []
    
    id_to_coor = {}

    # Iterate over xml file and append (X, Y) for asile   
    max_col = 0
    max_row = 0  
    
    ep = 0
    for elem in root.iter('Node'):
        if ((elem.attrib['SurfaceType'] == 'Aisle' or elem.attrib['SurfaceType'] == 'Deck') and elem.attrib['Orientation'] == 'North') or elem.attrib['SurfaceType'] == 'Transfer' or elem.attrib['SurfaceType'] == 'TransferCharger':
            nodes.append((int(elem.attrib['Row']), int(elem.attrib['Column'])))
            if int(elem.attrib['Row']) > max_row:
                max_row = int(elem.attrib['Row'])
            if int(elem.attrib['Column']) > max_col:
                max_col = int(elem.attrib['Column'])
            if elem.attrib['SurfaceType'] != 'Deck':
                id_to_coor[ep] = (elem.attrib['X'], elem.attrib['Y'])
                ep += 1
                
        else:
            if elem.attrib['Orientation'] == 'North':
                print(elem.attrib['SurfaceType'])
    print(nodes)
    
    map = ""
    for row in range(max_row, 0, -1):
        for col in range(max_col + 1):
            if (row, col) in nodes:
                map = map + "e"
            else:
                map = map + "@"
        map = map +"\n"
    print(map)
    
    print(id_to_coor)
    check_data(id_to_coor)
    print("Number of Endpoints: ", ep)
    print("max_row: ", max_row)
    print("max_col: ", max_col)
    
    convert_tasks(id_to_coor)
    

def check_data(conversion):
    data = []
    tasks = []
    print(os.getcwd())
    file = open("/home/Ethan/Documents/Symbotic/symbotic_tamp/task_allocation/taskList2.txt", "r")
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

    coordinate_points = [(int(point[0]), int(point[1]))for point in list(conversion.values())]
    
    for task in tasks:
        for point in task:
            if point not in coordinate_points:
                
                print("Does Not Exist: ", point)
    print("=================DONE================")
    
    
def convert_tasks(conversion: dict):
    data = []
    output = ""
    file = open("/home/Ethan/Documents/Symbotic/symbotic_tamp/task_allocation/taskList2.txt", "r")
    current_task_id = 1
    print("==============Building Kiva Tasks==============")
    key_list = list(conversion.keys())
    value_list = [(int(point[0]), int(point[1]))for point in list(conversion.values())]
    time = 0 
    while True:
        content = file.readline()
        if not content:
            start_position = value_list.index((data[-2]["Location"]["X"], data[-2]["Location"]["Y"]))
            goal_position = value_list.index((data[-1]["Location"]["X"], data[-1]["Location"]["Y"]))
            output = output + str(time) + "\t"  + str(key_list[start_position]) + "\t" + str(key_list[goal_position]) + "\t0\t0\n"
            break
        content = json.loads(content)
        if current_task_id != content["SRETaskID"]:
            start_position = value_list.index((data[-2]["Location"]["X"], data[-2]["Location"]["Y"]))
            goal_position = value_list.index((data[-1]["Location"]["X"], data[-1]["Location"]["Y"]))
            output = output + str(time) + "\t"  + str(key_list[start_position]) + "\t" + str(key_list[goal_position]) + "\t0\t0\n"
            current_task_id = content["SRETaskID"]
            time += 1
        data.append(content)
    file.close()
    
    print(output)
    print("==============Finished Building Kiva Tasks==============")
    

if __name__=="__main__":
    main()