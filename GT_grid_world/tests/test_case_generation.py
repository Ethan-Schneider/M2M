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
        # Create a small test warehouse with a 5x5 grid
        self.G = Graph(3, "data/maps/small_test", 25.0, 5, "uniform")
        self.S = Stats(3, 100, "test_stats.json")
        self.J = set()  # Empty set of tasks
        self.last_task_id = 0
        self.max_task_number = 20
        self.inbound_to_outbound = 1.0  # Equal probability for inbound/outbound
        
    def test_uninformed_uniform_task_generation(self):
        """Test basic task generation with uninformed uniform strategy"""
        # Generate 5 tasks
        N = 5
        J_new, new_last_task_id = CRG(
            self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
            self.last_task_id, self.max_task_number, self.G.warehouse,
            strategy="uninformed_uniform"
        )
        
        # Check that we got the expected number of tasks
        self.assertLessEqual(len(J_new), N)
        
        # Check task structure
        for task in J_new:
            task_id, start_loc, goal_loc = task
            # Check task ID is sequential
            self.assertGreater(task_id, self.last_task_id)
            # Check locations are valid
            self.assertIn(start_loc, self.G.get_all_non_obstacles())
            self.assertIn(goal_loc, self.G.get_all_non_obstacles())
            # Check start and goal are different
            self.assertNotEqual(start_loc, goal_loc)
            
    def test_informed_uniform_task_generation(self):
        """Test task generation with informed uniform strategy"""
        # Create custom weights to ensure clear distribution
        custom_weights = {
            1: (0.7, 0.8),  # High tasking weight
            2: (0.5, 0.3),  # Medium tasking weight
            3: (0.3, 0.1),  # Low tasking weight
            4: (0.4, 0.2),  # Low tasking weight
            5: (0.6, 0.4)   # Medium tasking weight
        }
        
        # Create new inventory with custom weights
        inventory = Inventory(
            num_skus=5,
            warehouse_locations=self.G.get_aisle_locations(),
            fill_percentage=50.0,
            weight_init=WeightInitialization.CUSTOM,
            custom_weights=custom_weights
        )
        
        # Generate 10 tasks
        N = 10
        J_new, new_last_task_id = CRG(
            self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
            self.last_task_id, self.max_task_number, inventory,
            strategy="informed_uniform"
        )
        
        # Check that we got the expected number of tasks
        self.assertLessEqual(len(J_new), N)
        
        # Check task structure
        for task in J_new:
            task_id, start_loc, goal_loc = task
            # Check task ID is sequential
            self.assertGreater(task_id, self.last_task_id)
            # Check locations are valid
            self.assertIn(start_loc, self.G.get_all_non_obstacles())
            self.assertIn(goal_loc, self.G.get_all_non_obstacles())
            # Check start and goal are different
            self.assertNotEqual(start_loc, goal_loc)
            
    def test_task_generation_with_full_warehouse(self):
        """Test task generation when warehouse is full"""
        # Create inventory with 100% fill using only valid aisle locations
        inventory = Inventory(
            num_skus=5,
            warehouse_locations=self.G.get_aisle_locations(),
            fill_percentage=100.0
        )
        
        # Generate tasks
        N = 5
        J_new, new_last_task_id = CRG(
            self.S, 0, self.J, self.G, N, self.inbound_to_outbound,
            self.last_task_id, self.max_task_number, inventory,
            strategy="uninformed_uniform"
        )
        
        # Should only generate outbound tasks since warehouse is full
        # TODO: This test is not workign as expected. Need to invetigate further. Otherwise the system runs as expected.
        for task in J_new:
            _, start_loc, _ = task
            # Check that start location is in the aisle locations and has an SKU
            self.assertIn(start_loc, self.G.get_aisle_locations())
            self.assertIn(start_loc, inventory.get_full_locations())
            # Verify it's a valid location in the warehouse
            self.assertLess(start_loc[0], 15)  # Row should be less than 15
            self.assertLess(start_loc[1], 13)  # Column should be less than 13
            
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
            _, _, goal_loc = task
            self.assertIn(goal_loc, self.G.get_aisle_locations())
            
    def test_task_generation_with_max_tasks(self):
        """Test task generation when max tasks is reached"""
        # Create a set of existing tasks that reaches max_task_number
        self.J = {(i, (0,0), (1,1)) for i in range(self.max_task_number)}
        
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