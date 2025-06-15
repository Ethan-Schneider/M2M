import unittest
import numpy as np
import matplotlib.pyplot as plt
from GT_grid_world.src.inventory_manager.inventory import Inventory
from GT_grid_world.src.inventory_manager.item import WeightInitialization
from GT_grid_world.src.case_request_generator import CRG
from GT_grid_world.src.graph import Graph
from GT_grid_world.src.analysis.statistics import Stats

class TestCaseRequest(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.G = Graph(3, "data/maps/small_test", 25.0, 5)
        self.S = Stats(3, 100, "test_output.json")
        self.J = set()  # Set of unassigned tasks
        self.J_assigned = set()  # Set of assigned tasks (s, d)
        self.last_task_id = 0
        self.max_task_number = 20
        self.inbound_to_outbound = 1.0
        
    def test_uninformed_uniform_task_generation(self):
        """Test basic task generation with uninformed uniform strategy"""
        # Generate tasks
        N = 5
        J_new, new_last_task_id = CRG(
            self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
            self.last_task_id, self.max_task_number, self.G.warehouse,
            strategy="uninformed_uniform"
        )
        
        # Check that we got the expected number of tasks
        self.assertEqual(len(J_new), N)
        
        # Check task structure and validity
        for task in J_new:
            task_id, start_locations, goal_locations = task
            # Check task ID
            self.assertGreater(task_id, self.last_task_id)
            self.assertLessEqual(task_id, new_last_task_id)
            
            # Check that start and goal locations are frozensets
            self.assertIsInstance(start_locations, frozenset)
            self.assertIsInstance(goal_locations, frozenset)
            
            # Check that locations are valid
            for loc in start_locations:
                self.assertLessEqual(loc[0], self.G.height)
                self.assertLessEqual(loc[1], self.G.width)
            for loc in goal_locations:
                self.assertLessEqual(loc[0], self.G.height)
                self.assertLessEqual(loc[1], self.G.width)
    
    def test_informed_uniform_task_generation(self):
        """Test task generation with informed uniform strategy"""
        # Create custom weights for SKUs
        custom_weights = {
            1: (0.8, 0.8),  # High appearance and tasking weight
            2: (0.2, 0.2),  # Low appearance and tasking weight
            3: (0.5, 0.5),  # Medium weights
            4: (0.3, 0.7),  # Low appearance, high tasking
            5: (0.7, 0.3)   # High appearance, low tasking
        }
        
        # Create inventory with custom weights
        inventory = Inventory(
            num_skus=5,
            warehouse_locations=self.G.get_aisle_locations(),
            fill_percentage=50.0,
            weight_init=WeightInitialization.CUSTOM,
            custom_weights=custom_weights
        )
        
        # Generate tasks
        N = 5
        J_new, new_last_task_id = CRG(
            self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
            self.last_task_id, self.max_task_number, inventory,
            strategy="informed_uniform"
        )
        # Check task structure and validity
        for task in J_new:
            task_id, start_locations, goal_locations = task
            # Check task ID
            self.assertGreater(task_id, self.last_task_id)
            self.assertLessEqual(task_id, new_last_task_id)
            
            # Check that start and goal locations are frozensets
            self.assertIsInstance(start_locations, frozenset)
            self.assertIsInstance(goal_locations, frozenset)
            
            # Check that locations are valid
            for loc in start_locations:
                self.assertLessEqual(loc[0], self.G.height)
                self.assertLessEqual(loc[1], self.G.width)
            for loc in goal_locations:
                self.assertLessEqual(loc[0], self.G.height)
                self.assertLessEqual(loc[1], self.G.width)
    
    # def test_task_generation_with_full_warehouse(self):
    #     """Test task generation when warehouse is full"""
    #     # Create inventory with 100% fill using only valid aisle locations
    #     inventory = Inventory(
    #         num_skus=5,
    #         warehouse_locations=self.G.get_aisle_locations(),
    #         fill_percentage=100.0
    #     )
        
    #     # Generate tasks
    #     N = 5
    #     J_new, new_last_task_id = CRG(
    #         self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
    #         self.last_task_id, self.max_task_number, inventory,
    #         strategy="uninformed_uniform"
    #     )
        
    #     # Should only generate outbound tasks since warehouse is full
    #     for task in J_new:
    #         _, start_locations, goal_locations = task
    #         # Check that start locations are in the aisle locations and have SKUs
    #         self.assertTrue(all(loc in self.G.get_aisle_locations() for loc in start_locations))
    #         self.assertTrue(all(loc in inventory.get_full_locations() for loc in start_locations))
    #         # Check that goal locations are station locations
    #         self.assertTrue(all(loc in self.G.get_station_locations() for loc in goal_locations))
    
    def test_task_generation_with_empty_warehouse(self):
        """Test task generation when warehouse is empty"""
        # Create inventory with 0% fill
        inventory = Inventory(
            num_skus=5,
            warehouse_locations=self.G.get_aisle_locations(),
            fill_percentage=0.0
        )
        
        # Generate tasks
        N = 5
        J_new, new_last_task_id = CRG(
            self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
            self.last_task_id, self.max_task_number, inventory,
            strategy="uninformed_uniform"
        )
        
        # Should only generate inbound tasks since warehouse is empty
        for task in J_new:
            _, start_locations, goal_locations = task
            # Check that start locations are station locations
            self.assertTrue(all(loc in self.G.get_station_locations() for loc in start_locations))
            # Check that goal locations are empty aisle locations
            self.assertTrue(all(loc in self.G.get_aisle_locations() for loc in goal_locations))
            self.assertTrue(all(loc in inventory.get_empty_locations() for loc in goal_locations))
    
    def test_task_generation_with_max_tasks(self):
        """Test task generation when max tasks is reached"""
        # Fill J with max tasks
        for i in range(self.max_task_number):
            self.J.add((i, frozenset(self.G.get_station_locations()), 
                       frozenset(self.G.get_aisle_locations())))
        
        # Try to generate more tasks
        N = 5
        J_new, new_last_task_id = CRG(
            self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
            self.last_task_id, self.max_task_number, self.G.warehouse,
            strategy="uninformed_uniform"
        )
        
        # Should not generate any new tasks
        self.assertEqual(len(J_new), 0)
    
    def test_invalid_strategy(self):
        """Test task generation with invalid strategy"""
        with self.assertRaises(ValueError):
            CRG(
                self.S, 0, self.J, self.G, 5, self.inbound_to_outbound,
                self.last_task_id, self.max_task_number, self.G.warehouse,
                strategy="invalid_strategy"
            )