import unittest
import numpy as np
import matplotlib.pyplot as plt
from GT_grid_world.src.inventory_manager.inventory import Inventory
from GT_grid_world.src.inventory_manager.item import SKU, WeightInitialization

class TestInventory(unittest.TestCase):
    def setUp(self):
        # Create a small test warehouse with a 5x5 grid
        self.warehouse_locations = [(x, y) for x in range(5) for y in range(5)]
        self.num_skus = 3
        
    def test_sku_creation(self):
        """Test basic SKU creation and properties"""
        sku = SKU(1, appearance_weight=0.5, tasking_weight=0.7)
        self.assertEqual(sku.sku_id, 1)
        self.assertEqual(sku.appearance_weight, 0.5)
        self.assertEqual(sku.tasking_weight, 0.7)
        
    def test_random_weight_init(self):
        """Test random weight initialization"""
        sku = SKU(1, weight_init=WeightInitialization.RANDOM)
        self.assertGreaterEqual(sku.appearance_weight, 0.1)
        self.assertLessEqual(sku.appearance_weight, 1.0)
        self.assertGreaterEqual(sku.tasking_weight, 0.1)
        self.assertLessEqual(sku.tasking_weight, 1.0)
        
    def test_uniform_weight_init(self):
        """Test uniform weight initialization"""
        sku = SKU(1, weight_init=WeightInitialization.UNIFORM)
        self.assertEqual(sku.appearance_weight, 1.0)
        self.assertEqual(sku.tasking_weight, 1.0)
        
    def test_inventory_creation(self):
        """Test basic inventory creation"""
        inventory = Inventory(self.num_skus, self.warehouse_locations)
        self.assertEqual(len(inventory.get_all_skus()), self.num_skus)
        
    def test_inventory_fill(self):
        """Test inventory filling with different percentages"""
        # Test 50% fill
        inventory = Inventory(self.num_skus, self.warehouse_locations, fill_percentage=50.0)
        filled_locations = inventory.get_full_locations()
        self.assertAlmostEqual(len(filled_locations) / len(self.warehouse_locations), 0.5, delta=0.1)
        
        # Test 100% fill
        inventory = Inventory(self.num_skus, self.warehouse_locations, fill_percentage=100.0)
        filled_locations = inventory.get_full_locations()
        self.assertEqual(len(filled_locations), len(self.warehouse_locations))
        
    def test_sku_operations(self):
        """Test adding and removing SKUs"""
        inventory = Inventory(self.num_skus, self.warehouse_locations)
        
        # Test adding SKU
        location = (0, 0)
        inventory.add_sku_instance(1, location)
        self.assertEqual(inventory.get_sku_at_location(location).sku_id, 1)
        
        # Test removing SKU
        inventory.remove_sku_instance(location)
        self.assertIsNone(inventory.get_sku_at_location(location))
        
    def test_sku_distribution(self):
        """Test SKU distribution based on appearance weights"""
        # Create custom weights to ensure clear distribution
        custom_weights = {
            1: (1.6, 0.3),  # High appearance weight
            2: (0.8, 0.5),  # Medium appearance weight
            3: (0.1, 0.2)   # Low appearance weight
        }
        
        inventory = Inventory(
            self.num_skus, 
            self.warehouse_locations,
            fill_percentage=100.0,
            weight_init=WeightInitialization.CUSTOM,
            custom_weights=custom_weights
        )
        
        plt1 = visualize_inventory(inventory, "Custom Weight Initialization")
        plt1.savefig('custom_weight_inventory.png')
        plt1.close()

        # Count instances of each SKU
        sku_counts = {
            sku_id: len(inventory.get_sku_instances(sku_id))
            for sku_id in range(1, self.num_skus + 1)
        }
        
        # Verify that SKU 1 has more instances than SKU 2, which has more than SKU 3
        self.assertGreater(sku_counts[1], sku_counts[2])
        self.assertGreater(sku_counts[2], sku_counts[3])

def visualize_inventory(inventory: Inventory, title: str = "Warehouse SKU Distribution"):
    """Visualize the warehouse SKU distribution using matplotlib"""
    # Get the maximum x and y coordinates to determine grid size
    max_x = max(loc[0] for loc in inventory.get_full_locations())
    max_y = max(loc[1] for loc in inventory.get_full_locations())
    grid = np.zeros((max_x + 1, max_y + 1))
    
    # Map SKU IDs to colors
    sku_colors = {}
    for sku_id in inventory.get_all_skus():
        # Generate a unique color for each SKU
        hue = sku_id / len(inventory.get_all_skus())
        sku_colors[sku_id] = plt.cm.hsv(hue)
    
    # Fill the grid with SKU colors
    for location, sku_id in inventory._Inventory__location_to_sku.items():
        x, y = location
        grid[x, y] = sku_id
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='tab20')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('SKU ID')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Add grid lines
    plt.grid(True, which='both', color='black', linewidth=0.5)
    
    return plt

if __name__ == '__main__':
    # Run unit tests
    unittest.main() 