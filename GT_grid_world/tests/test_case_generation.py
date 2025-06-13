import unittest
import numpy as np
import matplotlib.pyplot as plt
from GT_grid_world.src.inventory_manager.inventory import Inventory
from GT_grid_world.src.case_request_generator import CRG
from GT_grid_world.src.graph import Graph
from GT_grid_world.src.analysis.statistics import Stats

class TestCaseRequest(unittest.TestCase):
    def setUp(self):
        # Create a small test warehouse with a 5x5 grid
        G = Graph(3, "data/maps/symbotic_small", 25.0, 5, "uniform")