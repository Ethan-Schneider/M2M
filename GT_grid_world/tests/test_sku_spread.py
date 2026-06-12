"""Unit tests for the SKU Spread metric (roadmap section 1.7 / plan 3.6).

Two layers:

1. Pure math tests on ``compute_sku_spread`` -- empty / degenerate cases,
   the all-clustered = 0 invariant, the uniform-distribution closed-form
   value, and a Hypothesis property test that EZC is always non-negative.

2. Integration tests on ``Stats.append_sku_spread`` -- verify it clusters
   by aisle column and feeds the right ``eta`` matrix into the math.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from GT_grid_world.src.analysis.statistics import Stats, compute_sku_spread


# ---------------------------------------------------------------------------
# compute_sku_spread -- pure math
# ---------------------------------------------------------------------------
def test_compute_sku_spread_empty_matrix_returns_zero():
    """Empty input -> 0 (no SKUs to spread)."""
    assert compute_sku_spread(np.zeros((0, 0))) == 0.0
    assert compute_sku_spread(np.zeros((0, 5))) == 0.0
    assert compute_sku_spread(np.zeros((5, 0))) == 0.0


def test_compute_sku_spread_all_zero_inventory_returns_zero():
    """Every SKU has zero instances -> EZC == 0."""
    eta = np.zeros((4, 6), dtype=np.int64)
    assert compute_sku_spread(eta) == 0.0


def test_compute_sku_spread_each_sku_in_one_cluster_returns_zero():
    """If every SKU's instances all sit in a single cluster, every per-SKU
    entropy ``H_i`` is 0, so the weighted sum is 0. This is the core
    "clustered placement is bad" baseline.
    """
    eta = np.array([
        [10, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 7, 0],
    ], dtype=np.int64)
    assert compute_sku_spread(eta) == pytest.approx(0.0)


def test_compute_sku_spread_uniform_one_sku_matches_closed_form():
    """One SKU with N instances split evenly across C clusters has
    ``H_i = ln(C)`` (max entropy) and ``EZC = N * ln(C)``.
    """
    N, C = 12, 4
    per_cluster = N // C
    eta = np.full((1, C), per_cluster, dtype=np.int64)
    expected = N * math.log(C)
    assert compute_sku_spread(eta) == pytest.approx(expected)


def test_compute_sku_spread_two_skus_independent_contributions():
    """EZC is additive across SKUs. Two independent uniform SKUs should add."""
    sku1 = np.array([[3, 3, 3, 3]], dtype=np.int64)   # N=12, H=ln(4)
    sku2 = np.array([[5, 5]], dtype=np.int64)         # N=10, H=ln(2)
    expected = 12 * math.log(4) + 10 * math.log(2)
    eta = np.array([
        [3, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 5, 5],
    ], dtype=np.int64)
    # The all-zero columns for each SKU contribute 0*log(0) := 0, so EZC
    # is the same as if they were absent: each SKU sees the right entropy.
    assert compute_sku_spread(eta) == pytest.approx(expected)
    # Sanity-check by computing the two slices separately.
    assert compute_sku_spread(sku1) + compute_sku_spread(sku2) == pytest.approx(expected)


def test_compute_sku_spread_uniform_higher_than_clustered():
    """Same total instances; the more-uniform layout has strictly higher EZC."""
    clustered = np.array([[8, 0, 0, 0]], dtype=np.int64)
    half_split = np.array([[4, 4, 0, 0]], dtype=np.int64)
    uniform = np.array([[2, 2, 2, 2]], dtype=np.int64)
    s_cl = compute_sku_spread(clustered)
    s_half = compute_sku_spread(half_split)
    s_uni = compute_sku_spread(uniform)
    assert s_cl < s_half < s_uni


def test_compute_sku_spread_zero_count_columns_do_not_warn():
    """0 instances in a cluster contribute 0 (the 0*log(0) := 0 convention).
    The implementation must not produce numpy warnings about log-of-zero.
    """
    eta = np.array([[5, 0, 0, 5]], dtype=np.int64)
    with np.errstate(divide="raise", invalid="raise"):
        spread = compute_sku_spread(eta)
    # Two equal halves -> H = ln(2); EZC = 10 * ln(2).
    assert spread == pytest.approx(10 * math.log(2))


def test_compute_sku_spread_skus_with_zero_inventory_skipped():
    """A SKU with zero instances must contribute 0 and must not corrupt the
    sum (e.g. via 0/0 NaNs)."""
    eta = np.array([
        [0, 0, 0, 0],   # SKU 1 absent
        [3, 3, 0, 0],   # SKU 2 split half-half across two clusters
    ], dtype=np.int64)
    expected = 6 * math.log(2)
    assert compute_sku_spread(eta) == pytest.approx(expected)


def test_compute_sku_spread_accepts_float_counts():
    """Counts are typically int but the math must also accept floats (e.g.
    fractional counts from a smoothing or weighting layer)."""
    eta = np.array([[2.0, 2.0, 2.0, 2.0]])
    assert compute_sku_spread(eta) == pytest.approx(8.0 * math.log(4))


def test_compute_sku_spread_rejects_non_2d_gracefully():
    """1D / 3D inputs should not raise -- they return 0.0, treating
    malformed shapes as "no clusters defined" rather than blowing up the
    simulation snapshot path."""
    assert compute_sku_spread(np.array([1, 2, 3])) == 0.0
    assert compute_sku_spread(np.zeros((2, 3, 4))) == 0.0


# ---------------------------------------------------------------------------
# Hypothesis property: EZC is always non-negative for any valid eta
# ---------------------------------------------------------------------------
@given(arrays(dtype=np.int64, shape=st.tuples(
    st.integers(min_value=0, max_value=8),
    st.integers(min_value=0, max_value=8),
), elements=st.integers(min_value=0, max_value=20)))
@settings(max_examples=50, deadline=None)
def test_compute_sku_spread_property_nonnegative(eta):
    """EZC is a sum of N_i * H_i where H_i is Shannon entropy >= 0 and
    N_i >= 0, so EZC must always be non-negative for any valid eta.
    This is a direct invariant of the math.
    """
    spread = compute_sku_spread(eta)
    assert spread >= 0.0
    assert math.isfinite(spread)


# ---------------------------------------------------------------------------
# Stats.append_sku_spread -- integration with warehouse / aisle clustering
# ---------------------------------------------------------------------------
# These tests read the per-timestep series via name-mangled access to
# ``_Stats__sku_spread_per_timestep`` rather than ``save_data``-then-JSON,
# because ``save_data`` calls ``compute_velocity_timesteps`` which requires
# populated ``self.__paths``. A bare ``Stats`` fixture doesn't have agent
# paths recorded, and that downstream brittleness is unrelated to the
# metric. ``Stats`` doesn't expose a getter for the series, so direct
# name-mangled access is the cleanest route until / unless one is added.
def _spread_series(stats: Stats) -> list:
    return stats._Stats__sku_spread_per_timestep


def test_append_sku_spread_zero_for_empty_warehouse(tiny_graph, minimal_stats):
    """An empty warehouse (0% fill) yields EZC = 0 because no SKU has any
    instances. The metric must be recorded as a 0.0 entry, not silently
    skipped.
    """
    minimal_stats.append_sku_spread(
        tiny_graph.warehouse,
        tiny_graph.warehouse.get_all_skus().__len__(),
        tiny_graph.get_aisle_locations(),
    )
    assert _spread_series(minimal_stats) == [0.0]


def test_append_sku_spread_records_one_entry_per_call(populated_graph, minimal_stats):
    """Each call appends exactly one timestep entry. After 3 calls we expect
    3 entries, all >= 0 (warehouse is partially populated).
    """
    n_skus = populated_graph.warehouse.get_all_skus().__len__()
    aisles = populated_graph.get_aisle_locations()
    for _ in range(3):
        minimal_stats.append_sku_spread(populated_graph.warehouse, n_skus, aisles)

    series = _spread_series(minimal_stats)
    assert len(series) == 3
    assert all(s >= 0.0 for s in series)


def test_append_sku_spread_handles_no_aisles_gracefully(tiny_graph, minimal_stats):
    """If the caller passes an empty aisle list (degenerate map / config),
    the metric must record 0.0 rather than raise.
    """
    minimal_stats.append_sku_spread(
        tiny_graph.warehouse,
        tiny_graph.warehouse.get_all_skus().__len__(),
        [],
    )
    assert _spread_series(minimal_stats) == [0.0]


def test_append_sku_spread_handles_zero_num_skus_gracefully(tiny_graph, minimal_stats):
    """If ``num_skus == 0`` (empty SKU registry) the metric records 0.0."""
    minimal_stats.append_sku_spread(
        tiny_graph.warehouse,
        0,
        tiny_graph.get_aisle_locations(),
    )
    assert _spread_series(minimal_stats) == [0.0]


def test_append_sku_spread_clusters_by_column_not_row(populated_graph, minimal_stats):
    """Same warehouse, two different aisle-location lists that disagree on
    column membership produce different EZC values. This locks in that the
    Stats method actually uses the aisle_locations argument to define
    clusters (rather than e.g. always using the warehouse's full extent).
    """
    n_skus = populated_graph.warehouse.get_all_skus().__len__()
    full_aisles = populated_graph.get_aisle_locations()

    one_column = sorted({loc[1] for loc in full_aisles})[0]
    single_col_aisles = [loc for loc in full_aisles if loc[1] == one_column]

    minimal_stats.append_sku_spread(populated_graph.warehouse, n_skus, full_aisles)
    minimal_stats.append_sku_spread(populated_graph.warehouse, n_skus, single_col_aisles)

    full_spread, single_col_spread = _spread_series(minimal_stats)
    assert single_col_spread == 0.0, \
        "single-column clustering must collapse to EZC = 0 (every SKU in one cluster)"
    assert full_spread >= single_col_spread
