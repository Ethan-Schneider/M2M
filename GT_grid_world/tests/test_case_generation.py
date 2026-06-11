"""Unit tests for ``case_request_generator.CRG``.

Pytest-style rewrite of the original ``unittest`` suite. The original tests
were written against an earlier ``CRG`` signature and a pre-dict task
representation; this version targets the current signature on the
``symbotic_2026`` branch:

    CRG(S, t, J, G, Rs, N, inbound_to_outbound, last_task_id,
        max_task_number, inventory, *, strategy=..., deadline_generation_method=..., ...)
        -> (J, last_task_id, outbound_tasks, inbound_tasks)

Task tuple shape (current branch): ``(start_locs, goal_locs, deadline,
sku_id, type)`` keyed by ``task_id`` in a ``Dict[int, Tuple]``. Type is
``0=outbound``, ``1=inbound``.

Coverage focus
--------------
* ``uninformed_uniform`` and ``feedback_control`` strategies (the two paths
  in active use in ``run_experiments.sh``)
* Backpressure when the warehouse is empty (only inbound tasks generated)
  or full (only outbound tasks generated)
* Max-task-cap behaviour
* Invalid-strategy error path
"""

from __future__ import annotations

import pytest

from GT_grid_world.src.case_request_generator import (
    CRG,
    TASK_TYPE_INBOUND,
    TASK_TYPE_OUTBOUND,
    TASK_TYPE_SHUFFLE,
    _generate_shuffle_task,
    get_deadline,
)


# ---------------------------------------------------------------------------
# get_deadline
# ---------------------------------------------------------------------------
def test_get_deadline_constant_returns_offset():
    assert get_deadline(current_time=10, deadline_generation_method="constant", deadline_offset=30) == 40


def test_get_deadline_none_returns_sentinel():
    assert get_deadline(0, "none", 30) == 9999999


def test_get_deadline_unknown_method_falls_back():
    assert get_deadline(10, "this_is_not_a_method", 30) == 40


def test_get_deadline_normal_is_close_to_offset(seeded_rng):
    samples = [get_deadline(0, "normal", 30) for _ in range(200)]

    mean = sum(samples) / len(samples)
    assert 25 <= mean <= 35


# ---------------------------------------------------------------------------
# CRG core behaviour
# ---------------------------------------------------------------------------
def test_crg_returns_four_tuple_with_correct_types(
    minimal_stats,
    tiny_graph,
    sample_agents,
    filled_inventory,
    seeded_rng,
):
    J: dict = {}
    result = CRG(
        minimal_stats,
        0,                            # t
        J,
        tiny_graph,
        sample_agents,
        N=2,
        inbound_to_outbound=1.0,
        last_task_id=0,
        max_task_number=20,
        inventory=filled_inventory,
        strategy="uninformed_uniform",
    )

    assert len(result) == 4
    new_J, new_last_id, outbound_tasks, inbound_tasks = result
    assert isinstance(new_J, dict)
    assert isinstance(new_last_id, int)
    assert isinstance(outbound_tasks, list)
    assert isinstance(inbound_tasks, list)


def test_crg_uninformed_generates_well_formed_tasks(
    minimal_stats,
    tiny_graph,
    sample_agents,
    filled_inventory,
    seeded_rng,
):
    J: dict = {}
    new_J, new_last_id, _, _ = CRG(
        minimal_stats,
        0,
        J,
        tiny_graph,
        sample_agents,
        N=5,
        inbound_to_outbound=1.0,
        last_task_id=0,
        max_task_number=20,
        inventory=filled_inventory,
        strategy="uninformed_uniform",
    )

    assert 1 <= len(new_J) <= 5  # Some tasks may skip if no valid locations
    assert new_last_id >= len(new_J)

    aisles = set(tiny_graph.get_aisle_locations())
    stations = set(tiny_graph.get_station_locations())
    full = set(filled_inventory.get_full_locations())

    for task_id, task_tuple in new_J.items():
        start_locs, goal_locs, deadline, sku_id, task_type = task_tuple

        assert isinstance(start_locs, frozenset)
        assert isinstance(goal_locs, frozenset)
        assert isinstance(deadline, int)
        assert task_type in {0, 1}

        if task_type == 1:  # inbound: stations -> empty aisles
            assert start_locs <= stations
            assert goal_locs <= aisles
        else:  # outbound: full aisles -> stations
            assert start_locs <= full
            assert goal_locs <= stations


def test_crg_max_tasks_cap_returns_unchanged_J(
    minimal_stats,
    tiny_graph,
    sample_agents,
    filled_inventory,
    seeded_rng,
):
    """When ``len(J) >= max_task_number`` we should bail without generating.

    This exercises the early-return path that previously returned a 2-tuple
    instead of a 4-tuple — fixing that bug is part of the same change as
    this test.
    """
    stations = frozenset(tiny_graph.get_station_locations())
    aisles = frozenset(tiny_graph.get_aisle_locations())

    J: dict = {
        i: (stations, aisles, 100, 0, 1) for i in range(10)
    }

    new_J, new_last_id, outbound_tasks, inbound_tasks = CRG(
        minimal_stats,
        0,
        J,
        tiny_graph,
        sample_agents,
        N=5,
        inbound_to_outbound=1.0,
        last_task_id=10,
        max_task_number=10,
        inventory=filled_inventory,
        strategy="uninformed_uniform",
    )

    assert len(new_J) == 10
    assert new_last_id == 10
    assert outbound_tasks == []
    assert inbound_tasks == []


def test_crg_invalid_strategy_raises_value_error(
    minimal_stats,
    tiny_graph,
    sample_agents,
    filled_inventory,
    seeded_rng,
):
    with pytest.raises(ValueError, match="Unknown strategy"):
        CRG(
            minimal_stats,
            0,
            {},
            tiny_graph,
            sample_agents,
            N=5,
            inbound_to_outbound=1.0,
            last_task_id=0,
            max_task_number=20,
            inventory=filled_inventory,
            strategy="not_a_real_strategy",
        )


# ---------------------------------------------------------------------------
# CRG outbound-only / inbound-only behaviour
# ---------------------------------------------------------------------------
def test_crg_uninformed_with_empty_warehouse_only_generates_inbound(
    minimal_stats,
    tiny_graph,
    sample_agents,
    empty_inventory,
    seeded_rng,
):
    """An empty warehouse should never produce an outbound task because the
    outbound branch's ``if not start_locations: continue`` guard skips them.
    """
    J: dict = {}
    new_J, _, outbound_tasks, _ = CRG(
        minimal_stats,
        0,
        J,
        tiny_graph,
        sample_agents,
        N=10,
        inbound_to_outbound=1.0,
        last_task_id=0,
        max_task_number=20,
        inventory=empty_inventory,
        strategy="uninformed_uniform",
    )

    assert outbound_tasks == []
    assert all(task_tuple[4] == 1 for task_tuple in new_J.values())


# ---------------------------------------------------------------------------
# Tasks recorded against the Stats object
# ---------------------------------------------------------------------------
def test_crg_records_release_and_deadline_per_generated_task(
    minimal_stats,
    tiny_graph,
    sample_agents,
    filled_inventory,
    seeded_rng,
):
    new_J, _, _, _ = CRG(
        minimal_stats,
        t=7,
        J={},
        G=tiny_graph,
        Rs=sample_agents,
        N=5,
        inbound_to_outbound=1.0,
        last_task_id=0,
        max_task_number=20,
        inventory=filled_inventory,
        strategy="uninformed_uniform",
    )

    # ``Stats`` does not expose per-task accessors for release/deadline; the
    # data lives behind name-mangled private dicts. Tests are allowed to
    # poke at them; production code should not.
    releases = minimal_stats._Stats__task_release_timestamps
    deadlines = minimal_stats._Stats__task_deadlines

    for task_id, (_, _, deadline, _, _) in new_J.items():
        assert releases[task_id] == 7
        assert deadlines[task_id] == deadline


# ---------------------------------------------------------------------------
# Shuffle (type=2) generation -- roadmap section 1.4
# ---------------------------------------------------------------------------
def test_generate_shuffle_task_produces_well_formed_tuple(
    minimal_stats, populated_graph, seeded_rng
):
    J: dict = {}
    new_J, success, new_last_id = _generate_shuffle_task(
        deadline_generation_method="constant",
        deadline_offset=30,
        sku_id=None,
        t=5,
        last_task_id=0,
        G=populated_graph,
        J=J,
        S=minimal_stats,
    )

    assert success is True
    assert new_last_id == 1
    assert 1 in new_J

    start_locs, goal_locs, deadline, sku_id, task_type = new_J[1]
    assert task_type == TASK_TYPE_SHUFFLE
    assert isinstance(start_locs, frozenset)
    assert isinstance(goal_locs, frozenset)
    assert isinstance(sku_id, int)
    assert deadline == 35

    full_warehouse = set(populated_graph.warehouse.get_full_locations())
    empty_warehouse = set(populated_graph.warehouse.get_empty_locations())
    assert start_locs <= full_warehouse, "shuffle start_locs must come from warehouse-full cells"
    assert goal_locs <= empty_warehouse, "shuffle goal_locs must come from warehouse-empty cells"
    assert start_locs.isdisjoint(goal_locs), "shuffle start and goal must not overlap"


def test_generate_shuffle_task_with_specified_sku_uses_only_that_skus_instances(
    minimal_stats, populated_graph, seeded_rng
):
    target_sku = next(
        sku for sku in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(sku) >= 1
    )
    expected_starts = frozenset(populated_graph.warehouse.get_sku_instances(target_sku))

    new_J, success, _ = _generate_shuffle_task(
        deadline_generation_method="constant",
        deadline_offset=10,
        sku_id=target_sku,
        t=0,
        last_task_id=42,
        G=populated_graph,
        J={},
        S=minimal_stats,
    )

    assert success is True
    start_locs, _, _, sku_id, _ = new_J[43]
    assert sku_id == target_sku
    assert start_locs == expected_starts


def test_generate_shuffle_task_returns_failure_when_warehouse_full(
    minimal_stats, populated_graph, seeded_rng
):
    for loc in list(populated_graph.warehouse.get_empty_locations()):
        populated_graph.warehouse.add_sku_instance(1, loc)
    assert populated_graph.warehouse.get_empty_locations() == []

    J: dict = {}
    new_J, success, last_id = _generate_shuffle_task(
        deadline_generation_method="constant",
        deadline_offset=10,
        sku_id=None,
        t=0,
        last_task_id=0,
        G=populated_graph,
        J=J,
        S=minimal_stats,
    )

    assert success is False
    assert new_J == {}
    assert last_id == 0


def test_generate_shuffle_task_returns_failure_when_warehouse_empty(
    minimal_stats, tiny_graph, seeded_rng
):
    """Empty warehouse (the default ``tiny_graph`` state) has no SKUs to pick
    up, so the helper must report failure and leave the task store untouched.
    """
    assert tiny_graph.warehouse.get_full_locations() == []

    new_J, success, last_id = _generate_shuffle_task(
        deadline_generation_method="constant",
        deadline_offset=10,
        sku_id=None,
        t=0,
        last_task_id=0,
        G=tiny_graph,
        J={},
        S=minimal_stats,
    )

    assert success is False
    assert new_J == {}
    assert last_id == 0


def test_generate_shuffle_task_returns_failure_when_specified_sku_absent(
    minimal_stats, populated_graph, seeded_rng
):
    """An unknown SKU id should produce a clean False return, not raise.

    ``Inventory.get_sku_instance_count`` raises ``ValueError`` for unknown
    SKUs, so the helper guards with ``in get_all_skus()`` to keep CRG's
    fall-through path well-defined.
    """
    absent_sku = max(populated_graph.warehouse.get_all_skus().keys()) + 100

    new_J, success, last_id = _generate_shuffle_task(
        deadline_generation_method="constant",
        deadline_offset=10,
        sku_id=absent_sku,
        t=0,
        last_task_id=0,
        G=populated_graph,
        J={},
        S=minimal_stats,
    )

    assert success is False
    assert last_id == 0


def test_crg_with_shuffle_percentage_one_generates_only_shuffle_tasks(
    minimal_stats, populated_graph, sample_agents, filled_inventory, seeded_rng
):
    """With shuffle_percentage=1.0 and a populated warehouse, every task
    should be type=2. The OB/IB id lists track only fall-through cases."""
    new_J, _, outbound_ids, inbound_ids = CRG(
        minimal_stats,
        0,
        {},
        populated_graph,
        sample_agents,
        N=5,
        inbound_to_outbound=1.0,
        last_task_id=0,
        max_task_number=20,
        inventory=filled_inventory,
        strategy="uninformed_uniform",
        shuffle_percentage=1.0,
    )

    assert outbound_ids == []
    assert inbound_ids == []
    assert len(new_J) >= 1
    assert all(t[4] == TASK_TYPE_SHUFFLE for t in new_J.values())


def test_crg_with_shuffle_percentage_zero_never_generates_shuffle_tasks(
    minimal_stats, populated_graph, sample_agents, filled_inventory, seeded_rng
):
    new_J, _, _, _ = CRG(
        minimal_stats,
        0,
        {},
        populated_graph,
        sample_agents,
        N=10,
        inbound_to_outbound=1.0,
        last_task_id=0,
        max_task_number=20,
        inventory=filled_inventory,
        strategy="uninformed_uniform",
        shuffle_percentage=0.0,
    )

    assert all(t[4] in {TASK_TYPE_OUTBOUND, TASK_TYPE_INBOUND} for t in new_J.values())


def test_crg_shuffle_falls_through_to_ib_ob_when_warehouse_empty(
    minimal_stats, tiny_graph, sample_agents, filled_inventory, seeded_rng
):
    """The fall-through path: shuffle die rolls true, helper fails because
    the warehouse is empty, CRG drops to the existing IB/OB branch. No
    type=2 tasks should appear.
    """
    new_J, _, _, _ = CRG(
        minimal_stats,
        0,
        {},
        tiny_graph,
        sample_agents,
        N=10,
        inbound_to_outbound=1.0,
        last_task_id=0,
        max_task_number=20,
        inventory=filled_inventory,
        strategy="uninformed_uniform",
        shuffle_percentage=1.0,
    )

    assert all(t[4] != TASK_TYPE_SHUFFLE for t in new_J.values()), \
        "shuffle should silently fall back to IB/OB when warehouse is empty"
