import xml.etree.ElementTree as ET
import numpy as np

def generate_robots(num_robots : int):
    """Generate robots within a map's driveway

    Args:
        num_robots (int): Number of robots to generate.

    Returns:
        list[tuple[int, tuple[int, int]]]: list of robotID and start location tuples.
    """
    _, drive_way_nodes = __import_network()
    
    startLocs = np.random.choice(len(drive_way_nodes), size=num_robots, replace=False)
    robots = [(i, drive_way_nodes[startLocs[i]]) for i in range(num_robots)]
    
    return robots

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
        elif elem.attrib['SurfaceType'] == 'TransferCharger':
            drive_way_nodes.append((elem.attrib['X'], elem.attrib['Y']))
            
    return aisle_nodes, drive_way_nodes

if __name__ == "__main__":
    generate_robots(72)