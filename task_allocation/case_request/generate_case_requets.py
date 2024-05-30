#!/usr/bin/env -m python3

import json
import xml.etree.ElementTree as ET
import numpy as np
import datetime

from simulated_data import generate_data

def generate_case_requests(num_data: int, outbound_to_inbound_ratio: float, id_type: str, filename: str = "test_data", save : bool = 0):
    # Generate data
    data_points = generate_data.generate_data(num_data, outbound_to_inbound_ratio, 'random')
    
    # Import network coordinates
    aisle_nodes, drive_way_nodes = __import_network()
    
    # Add start and goal locations for tasks
    for data_point in data_points:
        if data_points[data_point]['outbound_inbound'] == 1:
            data_points[data_point]['startLoc'] = drive_way_nodes[np.random.choice(len(drive_way_nodes))]
            data_points[data_point]['goalLoc'] = aisle_nodes[np.random.choice(len(aisle_nodes))]
        else:
            data_points[data_point]['startLoc'] = aisle_nodes[np.random.choice(len(aisle_nodes))]
            data_points[data_point]['goalLoc'] = drive_way_nodes[np.random.choice(len(drive_way_nodes))]

    # Save tasks to Json
    if save:
        __save_to_json(data_points, filename)
    else:
        return data_points
    
def __import_network():
    """Returns coordinates foar aisle and driveway from map file

    Returns:
        list: X and Y coordinates for aisle (shelf) nodes
        list: X and Y coordinates for driveway (pickup/dropoff) nodes
    """
    
    #Open XML File
    tree = ET.parse('./task_allocation/network.xml')

    root = tree.getroot()
    
    #Initialize list of aisle nodes and driveway nodes
    aisle_nodes = []
    drive_way_nodes = []

    # Iterate over xml file and append (X, Y) for asile     
    for elem in root.iter('Node'):
        if elem.attrib['SurfaceType'] == 'Aisle':
            aisle_nodes.append((elem.attrib['X'], elem.attrib['Y']))
        elif elem.attrib['SurfaceType'] == 'Transfer' or elem.attrib['SurfaceType'] == 'TransferCharger':
            drive_way_nodes.append((elem.attrib['X'], elem.attrib['Y']))
            
    return aisle_nodes, drive_way_nodes

def __save_to_json(data: list, file_name: str):
    """Saves inbound/outbound task list to json file

    Args:
        data (list): I/O task data
        file_name (str): filename to be saved to 
    """
    if file_name == None:
        file_name = datetime.now()
    file_name =  str(file_name) + ".json"
    
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    generate_case_requests(20, 1, "random")
    print(generate_case_requests(50, 1, "random", save=0))
