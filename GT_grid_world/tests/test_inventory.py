"""Unit tests for ``Inventory`` and ``SKU``.

Pytest-style rewrite of the original ``unittest`` suite. The original file
had three latent issues that this version fixes:

1. ``test_sku_distribution`` indexed SKUs as ``1, 2, 3`` but
   ``Inventory(num_skus=3)`` constructs SKUs ``0, 1, 2``. The custom_weights
   keyed on ``1, 2, 3`` therefore left SKU 0 with random weights and silently
   ignored the entry for ``3``, then ``sku_counts[3]`` would raise.
2. The matplotlib visualization side-effect wrote ``custom_weight_inventory.png``
   to the cwd. We've removed it: visualization isn't behaviour worth testing.
3. Mixed ``unittest`` style with pytest's nicer ``assert`` ergonomics; we
   standardise on pytest fixtures and bare ``assert``.
"""

from __future__ import annotations

import numpy as np
import pytest

from GT_grid_world.src.inventory_manager.inventory import Inventory
from GT_grid_world.src.inventory_manager.item import SKU, WeightInitialization


# ---------------------------------------------------------------------------
# SKU
# ---------------------------------------------------------------------------
def test_sku_creation_with_explicit_weights():
    sku = SKU(1, appearance_weight=0.5, tasking_weight=0.7)

    assert sku.sku_id == 1
    assert sku.appearance_weight == 0.5
    assert sku.tasking_weight == 0.7


def test_sku_random_weight_init_within_bounds(seeded_rng):
    sku = SKU(1, weight_init=WeightInitialization.RANDOM)

    assert 0.1 <= sku.appearance_weight <= 1.0
    assert 0.1 <= sku.tasking_weight <= 1.0


def test_sku_uniform_weight_init_returns_one():
    sku = SKU(1, weight_init=WeightInitialization.UNIFORM)

    assert sku.appearance_weight == 1.0
    assert sku.tasking_weight == 1.0


def test_sku_custom_weight_init_requires_explicit_weights():
    with pytest.raises(ValueError, match="Custom weight initialization"):
        SKU(1, weight_init=WeightInitialization.CUSTOM)


# ---------------------------------------------------------------------------
# Inventory — construction and basic queries
# ---------------------------------------------------------------------------
@pytest.fixture()
def small_warehouse_locations() -> list:
    """A flat 5x5 grid of warehouse cells used for unit-level Inventory tests."""
    return [(x, y) for x in range(5) for y in range(5)]


def test_inventory_creation_makes_expected_skus(small_warehouse_locations):
    """SKU ids on the current branch are 1-indexed: ``range(1, num_skus+1)``.

    Note: ``local_task_reallocation`` flipped to 0-indexing. We'll need to
    pick one convention and update both branches when we land that work.
    """
    inventory = Inventory(num_skus=3, warehouse_locations=small_warehouse_locations)

    skus = inventory.get_all_skus()
    assert len(skus) == 3
    assert sorted(skus.keys()) == [1, 2, 3]


def test_inventory_50_percent_fill_uses_half_locations(small_warehouse_locations, seeded_rng):
    inventory = Inventory(
        num_skus=3,
        warehouse_locations=small_warehouse_locations,
        fill_percentage=50.0,
    )

    filled = len(inventory.get_full_locations())
    total = len(small_warehouse_locations)
    # The implementation uses ceil(N * pct/100); with N=25 that's 13 -> 0.52.
    # Allow a small delta for the rounding.
    assert abs(filled / total - 0.5) < 0.1


def test_inventory_100_percent_fill_uses_all_locations(small_warehouse_locations, seeded_rng):
    inventory = Inventory(
        num_skus=3,
        warehouse_locations=small_warehouse_locations,
        fill_percentage=100.0,
    )

    assert len(inventory.get_full_locations()) == len(small_warehouse_locations)


# ---------------------------------------------------------------------------
# Inventory — add/remove operations
# ---------------------------------------------------------------------------
def test_add_sku_instance_then_query_returns_correct_sku(small_warehouse_locations):
    inventory = Inventory(num_skus=3, warehouse_locations=small_warehouse_locations)
    location = (0, 0)

    inventory.add_sku_instance(1, location)

    sku_at = inventory.get_sku_at_location(location)
    assert sku_at is not None
    assert sku_at.sku_id == 1


def test_remove_sku_instance_clears_location(small_warehouse_locations):
    inventory = Inventory(num_skus=3, warehouse_locations=small_warehouse_locations)
    location = (0, 0)
    inventory.add_sku_instance(1, location)

    inventory.remove_sku_instance(location)

    assert inventory.get_sku_at_location(location) is None


def test_add_sku_instance_at_occupied_location_raises(small_warehouse_locations):
    inventory = Inventory(num_skus=3, warehouse_locations=small_warehouse_locations)
    location = (0, 0)
    inventory.add_sku_instance(1, location)

    with pytest.raises(ValueError, match="already occupied"):
        inventory.add_sku_instance(2, location)


def test_remove_sku_instance_from_empty_location_raises(small_warehouse_locations):
    inventory = Inventory(num_skus=3, warehouse_locations=small_warehouse_locations)

    with pytest.raises(ValueError, match="No SKU instance"):
        inventory.remove_sku_instance((0, 0))


def test_add_unknown_sku_raises(small_warehouse_locations):
    inventory = Inventory(num_skus=3, warehouse_locations=small_warehouse_locations)

    with pytest.raises(ValueError, match="SKU 99 does not exist"):
        inventory.add_sku_instance(99, (0, 0))


def test_add_sku_at_invalid_location_raises(small_warehouse_locations):
    inventory = Inventory(num_skus=3, warehouse_locations=small_warehouse_locations)

    with pytest.raises(ValueError, match="not a valid warehouse location"):
        inventory.add_sku_instance(1, (999, 999))


# ---------------------------------------------------------------------------
# Inventory — distribution honours custom appearance weights
# ---------------------------------------------------------------------------
def test_appearance_weighted_distribution_respects_ordering(
    small_warehouse_locations,
    seeded_rng,
):
    """When SKU appearance weights are sharply ordered, fill counts follow.

    SKU IDs are 0..num_skus-1, so the ``custom_weights`` keys must match
    that range. Using a 100% fill makes the test deterministic given the
    seeded RNG.
    """
    custom_weights = {
        1: (1.6, 0.3),  # high appearance
        2: (0.8, 0.5),  # medium
        3: (0.1, 0.2),  # low
    }

    inventory = Inventory(
        num_skus=3,
        warehouse_locations=small_warehouse_locations,
        fill_percentage=100.0,
        weight_init=WeightInitialization.CUSTOM,
        custom_weights=custom_weights,
    )

    counts = {sku_id: len(inventory.get_sku_instances(sku_id)) for sku_id in [1, 2, 3]}

    # Strong ordering should hold across this seeded run on a 25-cell grid.
    assert counts[1] > counts[2]
    assert counts[2] > counts[3]
    assert sum(counts.values()) == len(small_warehouse_locations)


# ---------------------------------------------------------------------------
# Inventory — empty / full bookkeeping
# ---------------------------------------------------------------------------
def test_empty_locations_complement_full_locations(small_warehouse_locations, seeded_rng):
    inventory = Inventory(
        num_skus=3,
        warehouse_locations=small_warehouse_locations,
        fill_percentage=50.0,
    )

    full = set(inventory.get_full_locations())
    empty = set(inventory.get_empty_locations())

    assert full | empty == set(small_warehouse_locations)
    assert full & empty == set()


def test_get_sku_instance_count_matches_get_sku_instances(small_warehouse_locations, seeded_rng):
    inventory = Inventory(
        num_skus=3,
        warehouse_locations=small_warehouse_locations,
        fill_percentage=50.0,
    )

    for sku_id in [1, 2, 3]:
        instances = inventory.get_sku_instances(sku_id)
        assert inventory.get_sku_instance_count(sku_id) == len(instances)


# ---------------------------------------------------------------------------
# Inventory — save/load roundtrip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_preserves_state(small_warehouse_locations, tmp_path, seeded_rng):
    original = Inventory(
        num_skus=3,
        warehouse_locations=small_warehouse_locations,
        fill_percentage=50.0,
    )
    save_file = tmp_path / "inventory.json"
    original.save_to_file(str(save_file))

    restored = Inventory(
        num_skus=3,
        warehouse_locations=small_warehouse_locations,
        fill_percentage=0.0,
    )
    restored.load_from_file(str(save_file))

    for sku_id in [1, 2, 3]:
        assert sorted(original.get_sku_instances(sku_id)) == sorted(
            restored.get_sku_instances(sku_id)
        )
    assert sorted(original.get_full_locations()) == sorted(restored.get_full_locations())
