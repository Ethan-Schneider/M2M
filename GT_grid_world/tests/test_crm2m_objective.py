"""Unit tests for the crM2M (concatenated rearrangement) objective and the
J_r consumer/lifecycle layer (roadmap section 4.2).

Two layers are covered:

1. The rearrangement utility itself -- ``rearrangement_cost_cube`` (cost = -U,
   gated) and ``compute_crm2m_terms`` (the static detour/benefit reference
   distances + same-aisle coupling mask). These are tested with hand-computed
   values so the three gates (U<=0, weighted-detour cutoff, cross-aisle) and the
   ``U = b - lambda*Delta`` arithmetic are pinned exactly.

2. The J_r consumer lifecycle -- ``merge_reallocation_tasks_into_J`` (4-tuple ->
   5-tuple, per-SKU dedup, high id range), ``drop_expired_rearrangements`` (a
   shuffle past its window must not be re-attempted), and
   ``prune_uncommitted_rearrangements`` (candidates no agent committed to are not
   left lingering in J). These use lightweight fakes so the lifecycle logic is
   tested in isolation from the full warehouse/graph stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from GT_grid_world.src.agent import Agent, AgentLoader
from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
    compute_ambient_slack,
    compute_crm2m_terms,
    rearrangement_cost_cube,
)
from GT_grid_world.src.task_allocation_algorithms.initial_solutions.fast_greedy import (
    fast_greedy_call,
)
from GT_grid_world.src.reallocation_tasks.jr_consumer import (
    add_reallocation_tasks_to_J_a,
    drop_expired_rearrangements,
    prune_uncommitted_rearrangements,
    REARRANGEMENT_TASK_ID_BASE,
    TASK_TYPE_SHUFFLE,
)


# ---------------------------------------------------------------------------
# rearrangement_cost_cube: arithmetic + gates
# ---------------------------------------------------------------------------
def test_cost_cube_returns_negative_utility_when_beneficial():
    """cost = -U = -(b - lambda*max(0, Delta)) with slack=c_pp=0.

    The penalty term is ``lambda * max(0, effective_delay - slack)``, so a
    *negative* detour is floored to zero penalty (it never yields a bonus).
    Note that in the real pipeline Delta = dist(anchor,s_p) + dist(s_p,h0) -
    dist(anchor,h0) >= 0 by the triangle inequality (both terms share the anchor),
    so this flooring is only exercised by the synthetic p=1 column here."""
    # P=2 starts, Q=2 goals, M=1 agent.
    start_home = np.array([10.0, 5.0])      # dist(s_p, h0)
    goal_home = np.array([1.0, 8.0])        # dist(d_q, h0)
    agent_start = np.array([[0.0, 0.0]])    # dist(g^m_{i-1}, s_p)
    agent_home = np.array([10.0])           # dist(a_m, h0)
    coupling = np.ones((2, 2), dtype=bool)
    valid_p = np.array([0, 1])
    valid_q = np.array([0, 1])

    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        valid_p, valid_q, lambda_=1.0, detour_cutoff=100.0,
    )

    # Delta[0]=0+10-10=0 -> max(0,Delta)=0; Delta[1]=0+5-10=-5 -> max(0,Delta)=0.
    assert cost.shape == (1, 2, 2)
    assert cost[0, 0, 0] == pytest.approx(-9.0)   # b=10-1=9,  U=9
    assert cost[0, 0, 1] == pytest.approx(-2.0)   # b=10-8=2,  U=2
    assert cost[0, 1, 0] == pytest.approx(-4.0)   # b=5-1=4,   U=4 (detour floored)
    assert np.isinf(cost[0, 1, 1])                # b=5-8=-3,  U=-3 <= 0 -> gated


def test_cost_cube_gates_non_positive_utility():
    """U <= 0 must be gated to +inf (no net benefit -> never chosen)."""
    start_home = np.array([5.0])
    goal_home = np.array([5.0])             # b = 0
    agent_start = np.array([[1.0]])
    agent_home = np.array([5.0])            # Delta = 1+5-5 = 1 > 0 -> U = 0 - 1 < 0
    coupling = np.ones((1, 1), dtype=bool)
    vp = np.array([0]); vq = np.array([0])

    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=1.0, detour_cutoff=100.0,
    )
    assert np.isinf(cost[0, 0, 0]) and cost[0, 0, 0] > 0


def test_cost_cube_gates_weighted_detour_cutoff():
    """A candidate with positive U but lambda*Delta >= cutoff is still gated."""
    start_home = np.array([10.0])
    goal_home = np.array([1.0])             # b = 9
    agent_start = np.array([[3.0]])
    agent_home = np.array([10.0])           # Delta = 3+10-10 = 3
    coupling = np.ones((1, 1), dtype=bool)
    vp = np.array([0]); vq = np.array([0])

    # lambda*Delta = 2*3 = 6 >= cutoff 5, even though U = 9 - 6 = 3 > 0.
    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=2.0, detour_cutoff=5.0,
    )
    assert np.isinf(cost[0, 0, 0])

    # Loosen the cutoff and the same candidate becomes choosable (cost = -3).
    cost2 = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=2.0, detour_cutoff=100.0,
    )
    assert cost2[0, 0, 0] == pytest.approx(-3.0)


def test_cost_cube_gates_cross_aisle_pairs():
    """coupling_mask False (start/goal not same aisle) must be gated to +inf."""
    start_home = np.array([10.0, 10.0])
    goal_home = np.array([1.0, 1.0])
    agent_start = np.array([[0.0, 0.0]])
    agent_home = np.array([10.0])
    # Only the diagonal pairs share an aisle.
    coupling = np.array([[True, False], [False, True]], dtype=bool)
    vp = np.array([0, 1]); vq = np.array([0, 1])

    cost = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling,
        vp, vq, lambda_=1.0, detour_cutoff=100.0,
    )
    assert np.isfinite(cost[0, 0, 0]) and np.isfinite(cost[0, 1, 1])
    assert np.isinf(cost[0, 0, 1]) and np.isinf(cost[0, 1, 0])


# ---------------------------------------------------------------------------
# rearrangement_cost_cube: buffer-aware slack + pick/place (c_pp) behaviour
# ---------------------------------------------------------------------------
def _single_candidate():
    """A single beneficial candidate: b=9, Delta=2 (M=P=Q=1)."""
    start_home = np.array([10.0])
    goal_home = np.array([1.0])          # b = 10 - 1 = 9
    agent_start = np.array([[2.0]])
    agent_home = np.array([10.0])        # Delta = 2 + 10 - 10 = 2
    coupling = np.ones((1, 1), dtype=bool)
    return start_home, goal_home, agent_start, agent_home, coupling


def test_cost_cube_slack_zero_cpp_zero_reduces_to_baseline():
    """slack=0 and c_pp=0 (the defaults) must reproduce the original b-lambda*Delta."""
    sh, gh, a_s, a_h, cp = _single_candidate()
    vp = np.array([0]); vq = np.array([0])
    baseline = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                       lambda_=1.5, detour_cutoff=100.0)
    explicit = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                       lambda_=1.5, detour_cutoff=100.0,
                                       slack=0.0, c_pp=0.0)
    # U = 9 - 1.5*2 = 6 -> cost = -6
    assert baseline[0, 0, 0] == pytest.approx(-6.0)
    assert explicit[0, 0, 0] == pytest.approx(baseline[0, 0, 0])


def test_cost_cube_slack_fully_hides_detour():
    """slack >= effective_delay -> penalty is zero -> U = b (judged on benefit)."""
    sh, gh, a_s, a_h, cp = _single_candidate()
    vp = np.array([0]); vq = np.array([0])
    # effective_delay = Delta + c_pp = 2; slack = 5 >= 2 -> discounted = 0 -> U = b = 9.
    cost = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                   lambda_=1.5, detour_cutoff=100.0,
                                   slack=5.0, c_pp=0.0)
    assert cost[0, 0, 0] == pytest.approx(-9.0)


def test_cost_cube_slack_partially_hides_detour():
    """0 < slack < effective_delay -> only the overflow is penalized."""
    sh, gh, a_s, a_h, cp = _single_candidate()
    vp = np.array([0]); vq = np.array([0])
    # effective_delay = 2, slack = 1 -> discounted = 1 -> U = 9 - 1.5*1 = 7.5.
    cost = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                   lambda_=1.5, detour_cutoff=100.0,
                                   slack=1.0, c_pp=0.0)
    assert cost[0, 0, 0] == pytest.approx(-7.5)


def test_cost_cube_cpp_adds_to_effective_delay():
    """c_pp increases the effective delay and thus the penalty (lowers U)."""
    sh, gh, a_s, a_h, cp = _single_candidate()
    vp = np.array([0]); vq = np.array([0])
    # effective_delay = Delta + c_pp = 2 + 3 = 5, slack = 0 -> U = 9 - 1.5*5 = 1.5.
    cost = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                   lambda_=1.5, detour_cutoff=100.0,
                                   slack=0.0, c_pp=3.0)
    assert cost[0, 0, 0] == pytest.approx(-1.5)


def test_cost_cube_gate_acts_on_raw_detour_not_cpp_or_slack():
    """The hard cutoff gates on lambda*Delta (raw travel detour) and IGNORES both
    c_pp and slack: c_pp is a fixed pick+place service cost, not a physical move
    length, so it must never push an otherwise-short shuffle over the ceiling."""
    sh, gh, a_s, a_h, cp = _single_candidate()  # Delta = 2, b = 9
    vp = np.array([0]); vq = np.array([0])
    # lambda*Delta = 2*2 = 4 < cutoff 5 -> NOT gated, even though a large c_pp=1
    # would have made lambda*(Delta+c_pp) = 6 >= 5 exceed the OLD effective-delay
    # gate. slack=100 fully hides the (Delta+c_pp) penalty, so U = b = 9.
    cost = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                   lambda_=2.0, detour_cutoff=5.0,
                                   slack=100.0, c_pp=1.0)
    assert cost[0, 0, 0] == pytest.approx(-9.0)

    # Raise the raw travel detour past the cutoff (lambda*Delta = 2*4 = 8 >= 5)
    # and the candidate is gated regardless of how much slack hides the penalty.
    long_start = np.array([[4.0]])   # Delta = 4 + 10 - 10 = 4
    cost2 = rearrangement_cost_cube(long_start, a_h, sh, gh, cp, vp, vq,
                                    lambda_=2.0, detour_cutoff=5.0,
                                    slack=100.0, c_pp=0.0)
    assert np.isinf(cost2[0, 0, 0])


def test_cost_cube_return_margin_default_zero_is_noop():
    """return_margin defaults to 0.0 -> byte-for-byte identical to omitting it."""
    sh, gh, a_s, a_h, cp = _single_candidate()
    vp = np.array([0]); vq = np.array([0])
    without = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                      lambda_=1.5, detour_cutoff=100.0,
                                      slack=1.0, c_pp=3.0)
    with_zero = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                        lambda_=1.5, detour_cutoff=100.0,
                                        slack=1.0, c_pp=3.0, return_margin=0.0)
    assert with_zero[0, 0, 0] == pytest.approx(without[0, 0, 0])


def test_cost_cube_return_margin_raises_penalty():
    """return_margin adds to the effective delay in the penalty (lowers U)."""
    sh, gh, a_s, a_h, cp = _single_candidate()  # b = 9, Delta = 2
    vp = np.array([0]); vq = np.array([0])
    # effective_delay = Delta + c_pp + margin = 2 + 0 + 3 = 5, slack = 0
    # -> U = 9 - 1.5*5 = 1.5.
    cost = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                   lambda_=1.5, detour_cutoff=100.0,
                                   slack=0.0, c_pp=0.0, return_margin=3.0)
    assert cost[0, 0, 0] == pytest.approx(-1.5)


def test_cost_cube_return_margin_shrinks_free_window():
    """A shuffle that was free (slack fully hid the delay) is no longer free once
    the margin pushes the required slack above what is available: the margin bar is
    slack >= Delta + c_pp + margin."""
    sh, gh, a_s, a_h, cp = _single_candidate()  # b = 9, Delta = 2
    vp = np.array([0]); vq = np.array([0])
    # slack = 2 exactly hides Delta+c_pp = 2 (free, U = b = 9) with no margin...
    free = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                   lambda_=1.5, detour_cutoff=100.0,
                                   slack=2.0, c_pp=0.0, return_margin=0.0)
    assert free[0, 0, 0] == pytest.approx(-9.0)
    # ...but with margin=2 the bar becomes slack >= 4, so 2 ticks overflow ->
    # U = 9 - 1.5*(2+0+2-2) = 9 - 1.5*2 = 6.
    margined = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                       lambda_=1.5, detour_cutoff=100.0,
                                       slack=2.0, c_pp=0.0, return_margin=2.0)
    assert margined[0, 0, 0] == pytest.approx(-6.0)


def test_cost_cube_return_margin_does_not_affect_gate():
    """The reject-gate is on raw lambda*Delta only; return_margin must NOT gate a
    physically-short shuffle even when margin+c_pp is large."""
    sh, gh, a_s, a_h, cp = _single_candidate()  # Delta = 2, b = 9
    vp = np.array([0]); vq = np.array([0])
    # lambda*Delta = 1.5*2 = 3 < cutoff 5 -> NOT gated. Huge margin only affects the
    # penalty; slack=100 hides all of it so U = b = 9 and the entry stays finite.
    cost = rearrangement_cost_cube(a_s, a_h, sh, gh, cp, vp, vq,
                                   lambda_=1.5, detour_cutoff=5.0,
                                   slack=100.0, c_pp=0.0, return_margin=50.0)
    assert cost[0, 0, 0] == pytest.approx(-9.0)


# ---------------------------------------------------------------------------
# compute_ambient_slack: the system-wide slack scalar
# ---------------------------------------------------------------------------
def test_ambient_slack_zero_when_buffer_disabled():
    assert compute_ambient_slack(0.0, 0, 1.0, 0.0, 1.5, 10.0) == 0.0


def test_ambient_slack_zero_when_buffer_healthy():
    """B <= K-1 (a free slot already exists) -> no wait -> slack 0."""
    # B=5, K=10 -> overshoot = max(0, 5 - 9) = 0.
    assert compute_ambient_slack(5.0, 10, 1.0, 0.5, 1.5, 10.0) == 0.0


def test_ambient_slack_overshoot_over_net_rate():
    """Full buffer, uncapped: slack = overshoot / (mu - alpha)."""
    # B=10, K=10 -> overshoot = 1; mu-alpha = 1.0 - 0.5 = 0.5 -> slack = 2.0.
    # slack_cap = cutoff/lambda = 10/1.5 ~ 6.667, does not bind.
    assert compute_ambient_slack(10.0, 10, 1.0, 0.5, 1.5, 10.0) == pytest.approx(2.0)


def test_ambient_slack_capped_by_slack_cap():
    """Near-saturation makes the raw value huge; slack_cap bounds it.

    With the default c_pp=0, slack_cap = c_pp + cutoff/lambda = 10/2 = 5.
    """
    # overshoot=1, mu-alpha=0.01 -> raw slack = 100; cap = 0 + 10/2 = 5.
    assert compute_ambient_slack(10.0, 10, 1.0, 0.99, 2.0, 10.0) == pytest.approx(5.0)


def test_ambient_slack_cap_includes_cpp():
    """slack_cap = c_pp + cutoff/lambda so slack can fully offset the pick+place
    service cost plus a max-detour's worth (decoupled from the raw-detour gate)."""
    # overshoot=1, mu-alpha=0.01 -> raw slack = 100; cap = 8 + 10/1.5 = 14.6667.
    expected = 8.0 + 10.0 / 1.5
    assert compute_ambient_slack(
        10.0, 10, 1.0, 0.99, 1.5, 10.0, c_pp=8.0
    ) == pytest.approx(expected)


def test_ambient_slack_cap_cpp_cancels_gate_passing_penalty():
    """At max slack, a gate-passing shuffle (Delta < cutoff/lambda) incurs ZERO
    penalty: discounted = max(0, (Delta+c_pp) - (c_pp + cutoff/lambda)) = 0."""
    # slack pinned to cap = 8 + 10/1.5; a Delta=4 (< 6.667) shuffle with c_pp=8:
    slack = compute_ambient_slack(10.0, 10, 1.0, 0.99, 1.5, 10.0, c_pp=8.0)
    effective = 4.0 + 8.0
    discounted = max(0.0, effective - slack)
    assert discounted == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# committed_shuffle_terms: per-completed-shuffle benefit/detour/utility logging
# ---------------------------------------------------------------------------
def test_committed_shuffle_terms_baseline_arithmetic():
    """benefit/detour/utility match the hand-computed objective (slack=c_pp=0).

    home=(0,0), prev=(3,0), s_p=(4,0), d_q=(1,0), manhattan:
      Delta   = |3-4| + 4 - 3 = 2
      benefit = 4 - 1 = 3
      U       = 3 - 1.5 * max(0, (2+0) - 0) = 3 - 3 = 0
    """
    from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
        committed_shuffle_terms,
    )
    benefit, detour, utility = committed_shuffle_terms(
        G=None, prev_cell=(3, 0), s_p=(4, 0), d_q=(1, 0), home=(0, 0),
        slack=0.0, c_pp=0.0, lambda_=1.5, method="manhattan",
    )
    assert benefit == pytest.approx(3.0)
    assert detour == pytest.approx(2.0)
    assert utility == pytest.approx(0.0)


def test_committed_shuffle_terms_slack_and_cpp():
    """c_pp raises the penalty; slack hides part of the effective delay."""
    from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
        committed_shuffle_terms,
    )
    # Delta=2, benefit=3, c_pp=2 -> effective=4.
    # slack=0 -> U = 3 - 1.5*4 = -3.
    _, _, u_no_slack = committed_shuffle_terms(
        G=None, prev_cell=(3, 0), s_p=(4, 0), d_q=(1, 0), home=(0, 0),
        slack=0.0, c_pp=2.0, lambda_=1.5, method="manhattan",
    )
    assert u_no_slack == pytest.approx(-3.0)
    # slack=4 fully hides the effective delay -> U = benefit = 3.
    _, _, u_full_slack = committed_shuffle_terms(
        G=None, prev_cell=(3, 0), s_p=(4, 0), d_q=(1, 0), home=(0, 0),
        slack=4.0, c_pp=2.0, lambda_=1.5, method="manhattan",
    )
    assert u_full_slack == pytest.approx(3.0)


def test_committed_shuffle_terms_return_margin():
    """return_margin adds to the effective delay in the logged utility, matching the
    cube. Delta=2, benefit=3, c_pp=2, margin=2, slack=4:
      U = 3 - 1.5 * max(0, (2 + 2 + 2) - 4) = 3 - 1.5*2 = 0."""
    from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
        committed_shuffle_terms,
    )
    _, _, u = committed_shuffle_terms(
        G=None, prev_cell=(3, 0), s_p=(4, 0), d_q=(1, 0), home=(0, 0),
        slack=4.0, c_pp=2.0, lambda_=1.5, method="manhattan", return_margin=2.0,
    )
    assert u == pytest.approx(0.0)


def test_committed_shuffle_terms_matches_cost_cube():
    """committed_shuffle_terms utility must equal -rearrangement_cost_cube for the
    same geometry (the logger and the allocator score identically)."""
    from GT_grid_world.src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
        committed_shuffle_terms,
    )
    # Encode the same triple into the cube's per-cell inputs:
    #   agent_start_cost_tensor[m,p] = dist(prev, s_p) = 1
    #   start_home[p]  = dist(s_p, h0) = 4
    #   goal_home[q]   = dist(d_q, h0) = 1
    #   agent_home[m]  = dist(prev, h0) = 3
    agent_start = np.array([[1.0]])
    agent_home = np.array([3.0])
    start_home = np.array([4.0])
    goal_home = np.array([1.0])
    coupling = np.ones((1, 1), dtype=bool)
    vp = np.array([0]); vq = np.array([0])
    # slack=4 fully hides the effective delay so U = benefit = 3 > 0 (a committed
    # shuffle always cleared the U>0 gate, so this is the representative regime).
    cube = rearrangement_cost_cube(
        agent_start, agent_home, start_home, goal_home, coupling, vp, vq,
        lambda_=1.5, detour_cutoff=100.0, slack=4.0, c_pp=2.0,
    )
    _, _, utility = committed_shuffle_terms(
        G=None, prev_cell=(3, 0), s_p=(4, 0), d_q=(1, 0), home=(0, 0),
        slack=4.0, c_pp=2.0, lambda_=1.5, method="manhattan",
    )
    assert utility == pytest.approx(-cube[0, 0, 0])


def test_ambient_slack_eps_floor_when_arrivals_exceed_drain():
    """alpha >= mu (backlog not clearing): the eps floor prevents div-by-zero /
    negatives, and the result saturates at slack_cap."""
    # mu-alpha = 1 - 1 = 0 -> floored at eps -> raw slack enormous -> capped at 5.
    val = compute_ambient_slack(10.0, 10, 1.0, 1.0, 2.0, 10.0)
    assert np.isfinite(val)
    assert val == pytest.approx(5.0)
    # alpha strictly greater than mu -> same clamp, still finite.
    val2 = compute_ambient_slack(10.0, 10, 1.0, 2.0, 2.0, 10.0)
    assert np.isfinite(val2) and val2 == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# OutputBuffer.estimate_outbound_fraction: alpha proxy for the timeless queue
# ---------------------------------------------------------------------------
def test_estimate_outbound_fraction_counts_outbound_in_window():
    from GT_grid_world.src.output_buffer import OutputBuffer
    # queue rows are [sku_id, task_type]; task_type 0 = outbound, 1 = inbound.
    queue = np.array([[4, 1], [14, 0], [25, 1], [7, 0]])
    # Full window: 2 outbound of 4 -> 0.5.
    assert OutputBuffer.estimate_outbound_fraction(queue, 4) == pytest.approx(0.5)
    # Window truncates to first 2 rows (1 outbound of 2) -> 0.5 here too.
    assert OutputBuffer.estimate_outbound_fraction(queue, 2) == pytest.approx(0.5)
    # First 3 rows: 1 outbound of 3.
    assert OutputBuffer.estimate_outbound_fraction(queue, 3) == pytest.approx(1.0 / 3.0)


def test_estimate_outbound_fraction_empty_queue_is_zero():
    from GT_grid_world.src.output_buffer import OutputBuffer
    assert OutputBuffer.estimate_outbound_fraction(np.empty((0, 2)), 10) == 0.0
    assert OutputBuffer.estimate_outbound_fraction(None, 10) == 0.0


# ---------------------------------------------------------------------------
# OutputBuffer.estimate_arrival_rate: alpha from the timestamped queue column
# ---------------------------------------------------------------------------
def test_estimate_arrival_rate_reads_timestamp_span():
    from GT_grid_world.src.output_buffer import OutputBuffer
    # rows [sku_id, task_type, arrival_second]; type 0 = outbound. Timestamps at
    # R=60 (1 task/sec) -> seconds 1..4, span = 3 s. 2 outbound / (3/60 min) = 40/min.
    queue = np.array([[4, 1, 1], [14, 0, 2], [25, 1, 3], [7, 0, 4]])
    assert OutputBuffer.estimate_arrival_rate(queue, 4) == pytest.approx(40.0)


def test_estimate_arrival_rate_zero_without_outbound():
    from GT_grid_world.src.output_buffer import OutputBuffer
    queue = np.array([[4, 1, 1], [25, 1, 2], [9, 1, 3]])
    assert OutputBuffer.estimate_arrival_rate(queue, 3) == 0.0


def test_estimate_arrival_rate_requires_timestamp_column():
    from GT_grid_world.src.output_buffer import OutputBuffer
    # 2-column queue has no timestamp column -> estimator must reject it so the
    # caller falls back to the outbound-fraction heuristic instead.
    queue = np.array([[4, 1], [14, 0]])
    with pytest.raises(ValueError):
        OutputBuffer.estimate_arrival_rate(queue, 2)


def test_estimate_arrival_rate_oversubscribed_regime_alpha_exceeds_drain():
    """R=60 with ~50% outbound is intentionally oversubscribed vs mu=25/min.

    Outbound offered ~= 0.5*60 = 30/min EXCEEDS the 25/min drain, so the net
    clearing rate mu-alpha is negative and slack pins to its cap. We assert
    alpha > mu to lock that regime in.
    """
    from GT_grid_world.src.output_buffer import (
        OutputBuffer,
        tasks_per_min_to_per_tick,
    )
    # 50/50 in/out, timestamps 1..4 at R=60 -> 2 outbound over 3 s span = 40/min.
    queue = np.array([[4, 1, 1], [14, 0, 2], [25, 1, 3], [7, 0, 4]])
    alpha_per_tick = tasks_per_min_to_per_tick(
        OutputBuffer.estimate_arrival_rate(queue, 4)
    )
    mu_per_tick = tasks_per_min_to_per_tick(25.0)
    assert alpha_per_tick > mu_per_tick


# ---------------------------------------------------------------------------
# compute_crm2m_terms: reference distances + coupling + anchor
# ---------------------------------------------------------------------------
def test_compute_crm2m_terms_distances_coupling_and_anchor(sample_agents, tiny_graph):
    """start_home/goal_home are Manhattan distances to h_0 = agents[0].home;
    coupling is same-column; agent_home uses the agent's anchor (last task goal,
    else current state)."""
    from GT_grid_world.src.utils import manhattan_distance

    Rs = sample_agents
    h0 = Rs.agents[0].home

    aisles = tiny_graph.get_aisle_locations()
    # Pick two starts and two goals spanning at least two columns when possible.
    cols = sorted({c for _, c in aisles})
    start_locs = [next(a for a in aisles if a[1] == cols[0])]
    goal_locs = [next(a for a in aisles if a[1] == cols[0])]
    if len(cols) > 1:
        start_locs.append(next(a for a in aisles if a[1] == cols[1]))
        goal_locs.append(next(a for a in aisles if a[1] == cols[1]))

    # Give agent 1 an anchor via a task sequence (goal = some aisle cell).
    anchor_goal = aisles[-1]
    Rs.agents[1].task_sequence.append((42, aisles[0], anchor_goal, 100))

    home, start_home, goal_home, agent_home, coupling = compute_crm2m_terms(
        Rs, tiny_graph, start_locs, goal_locs, method="manhattan"
    )

    assert home == h0
    for i, s in enumerate(start_locs):
        assert start_home[i] == pytest.approx(manhattan_distance(s, h0))
    for j, g in enumerate(goal_locs):
        assert goal_home[j] == pytest.approx(manhattan_distance(g, h0))

    # Coupling reflects same-column membership.
    for i, s in enumerate(start_locs):
        for j, g in enumerate(goal_locs):
            assert bool(coupling[i, j]) == (s[1] == g[1])

    # Idle agent 0 anchors at its state; agent 1 anchors at its last task goal.
    assert agent_home[0] == pytest.approx(manhattan_distance(Rs.agents[0].state, h0))
    assert agent_home[1] == pytest.approx(manhattan_distance(anchor_goal, h0))


# ---------------------------------------------------------------------------
# Integration: a beneficial shuffle is actually allocated by fast_greedy_call
# ---------------------------------------------------------------------------
def _beneficial_shuffle_scenario(G):
    """Find (column, start, anchor, goal) for a clearly-beneficial same-aisle
    shuffle on graph ``G``: a SKU instance ``s``, an empty anchor cell one row
    toward home, and an empty goal >= 4 rows below ``s`` in the same column.
    Returns None if the warehouse layout doesn't admit one."""
    aisles = G.get_aisle_locations()
    full = set(G.warehouse.get_full_locations())
    empty = set(G.warehouse.get_empty_locations())

    by_col = {}
    for loc in aisles:
        by_col.setdefault(loc[1], []).append(loc)

    for c, locs in by_col.items():
        rows = sorted(r for r, _ in locs)
        for s_row in rows:
            s = (s_row, c)
            a = (s_row + 1, c)
            if s not in full or a not in empty:
                continue
            g_candidates = [(gr, c) for gr in rows if gr >= s_row + 4 and (gr, c) in empty]
            if not g_candidates:
                continue
            g = max(g_candidates, key=lambda cell: cell[0])
            return (c, s, a, g)
    return None



def test_fast_greedy_allocates_beneficial_shuffle(populated_graph, minimal_stats, seeded_rng):
    """End-to-end through the live allocator: a same-aisle shuffle that moves a
    SKU much closer to the home reference, with the agent anchored right next to
    the pickup (small detour), has U > 0 and is allocated with negative cost.

    Geometry (Manhattan, single aisle column ``c``, home at the driveway row 17):
        b     = dist(s, h0) - dist(g, h0) = (g_row - s_row)         (same column)
        Delta = dist(anchor, s) + dist(s, h0) - dist(anchor, h0) = 2 (anchor one
                row below s, i.e. one step toward home)
        U     = b - lambda*Delta = (g_row - s_row) - 3   (lambda=1.5)
    so g at least ~4 rows below s gives U > 0.
    """
    G = populated_graph
    scenario = _beneficial_shuffle_scenario(G)
    if scenario is None:
        pytest.skip("populated_graph has no aisle column with the needed full/empty layout")

    c, s, a, g = scenario
    sku = G.warehouse.get_sku_at_location(s).sku_id
    h0 = (17, c)

    # Idle agent anchored at ``a`` (its state, since it has no task sequence),
    # with home = h0 so dist(a, h0) = dist(s, h0) - 1 and the detour is just 2.
    agent = Agent(agent_id=0, state=a, home=h0)
    Rs = AgentLoader([agent])

    shuffle_id = REARRANGEMENT_TASK_ID_BASE
    J = {shuffle_id: (frozenset({s}), frozenset({g}), 100, sku, TASK_TYPE_SHUFFLE)}

    _, allocations, cost = fast_greedy_call(
        minimal_stats, G, Rs, J, current_time=0,
        method="manhattan", base_cost_weight=1.0, deadline_weight=0.0,
        sku_distribution_weight=0.0, crm2m_lambda=1.5, crm2m_detour_cutoff=100.0,
    )

    assert allocations, "a clearly beneficial same-aisle shuffle should be allocated"
    assert allocations[0][1] == shuffle_id
    assert cost < 0, "a beneficial shuffle has cost -U < 0"


def test_py_lns_handles_shuffle_via_greedy_repair(populated_graph, minimal_stats, seeded_rng):
    """The LNS improver (py_lns_call -> greedy_repair) must thread the crM2M
    type=2 path without raising and keep a clearly-beneficial shuffle assigned."""
    from GT_grid_world.src.task_allocation_algorithms.py_lns_V2 import py_lns_call

    G = populated_graph
    scenario = _beneficial_shuffle_scenario(G)
    if scenario is None:
        pytest.skip("populated_graph has no aisle column with the needed full/empty layout")
    c, s, a, g = scenario
    sku = G.warehouse.get_sku_at_location(s).sku_id

    agent = Agent(agent_id=0, state=a, home=(17, c))
    Rs = AgentLoader([agent])
    shuffle_id = REARRANGEMENT_TASK_ID_BASE
    J = {shuffle_id: (frozenset({s}), frozenset({g}), 100, sku, TASK_TYPE_SHUFFLE)}

    Rs_out, allocations, _ = py_lns_call(
        minimal_stats, G, Rs, J, initial_task_assignment_strategy="fast_greedy",
        time_limit=0.05, cost_calculation_method="manhattan", repair_operator="greedy",
        t=0, crm2m_lambda=1.5, crm2m_detour_cutoff=100.0,
    )

    assigned_ids = {task[0] for ag in Rs_out.agents for task in ag.task_sequence}
    assert shuffle_id in assigned_ids, "LNS should keep the beneficial shuffle assigned"


# ---------------------------------------------------------------------------
# simulate: completed shuffle is reported on the rearrangement track
# ---------------------------------------------------------------------------
def test_simulate_routes_completed_shuffle_to_rearrangement_stats(
    populated_graph, minimal_stats, seeded_rng
):
    """A completed type=2 task must be recorded via
    ``add_completed_rearrangement_task_id`` (separate throughput track) and NOT
    via the regular ``add_completed_task_id`` / service-time path."""
    from GT_grid_world.src.simulate import simulate, TASK_TYPE_SHUFFLE as SIM_SHUFFLE

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]

    task_id = REARRANGEMENT_TASK_ID_BASE
    deadline = 100
    J = {task_id: (frozenset({pickup_loc}), frozenset({goal_loc}), deadline, sku, SIM_SHUFFLE)}
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(pickup_loc, True)

    # Tick 0: pickup. Tick 1: dropoff (completes the shuffle).
    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, J, {}, "small_test", t=0)
    agent.state = goal_loc
    agent.path_sequence = []
    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, J, {}, "small_test", t=1)

    assert task_id in minimal_stats.get_completed_rearrangement_task_ids()
    assert task_id not in minimal_stats.get_completed_task_ids()
    assert minimal_stats.get_total_completed_rearrangement_tasks() == 1


def test_simulate_picks_up_shuffle_living_in_J_a(
    populated_graph, minimal_stats, seeded_rng
):
    """A shuffle stored in ``J_a`` (not ``J``) must be pickable.

    Regression: the status=1 pickup gate previously only admitted ``task_id in
    J``; shuffles live in ``J_a``, so every shuffle pickup was silently skipped
    and the move could never execute. Here the agent sits on the pickup cell with
    the task in ``J_a`` only -- after one tick it must be carrying the SKU
    (status 1 -> 2)."""
    from GT_grid_world.src.simulate import simulate, TASK_TYPE_SHUFFLE as SIM_SHUFFLE

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]

    task_id = REARRANGEMENT_TASK_ID_BASE
    deadline = 100
    J_a = {task_id: (frozenset({pickup_loc}), frozenset({goal_loc}), deadline, sku, SIM_SHUFFLE)}
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(pickup_loc, True)

    # Task is only in J_a; J is empty. The agent is already on the pickup cell.
    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, {}, J_a, "small_test", t=0)

    assert agent.status == 2, "agent should have picked up the J_a shuffle and moved to delivery"
    assert agent.get_sku_id_carrying() == sku


def test_simulate_aborts_stale_shuffle_with_wrong_sku_at_cell(
    populated_graph, minimal_stats, seeded_rng
):
    """A shuffle whose committed source cell no longer holds its SKU must be
    aborted, not executed.

    Regression: under warehouse churn an outbound can empty a shuffle's source
    cell and an inbound can refill it with a *different* SKU before the agent
    arrives. Picking up the wrong item used to crash at the delivery-side
    SKU-match check. The agent must instead drop the stale shuffle (pop from
    ``J_a``), carry nothing, and be freed for re-tasking -- no exception."""
    from GT_grid_world.src.simulate import simulate, TASK_TYPE_SHUFFLE as SIM_SHUFFLE

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]
    actual_sku = populated_graph.warehouse.get_sku_at_location(pickup_loc).sku_id

    # The task expects a SKU that is NOT the one physically at the cell.
    wrong_sku = actual_sku + 1
    task_id = REARRANGEMENT_TASK_ID_BASE
    deadline = 100
    J_a = {task_id: (frozenset({pickup_loc}), frozenset({goal_loc}), deadline, wrong_sku, SIM_SHUFFLE)}

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(pickup_loc, True)

    Rs, J, J_a = simulate(minimal_stats, populated_graph, Rs, {}, J_a, "small_test", t=0)

    assert task_id not in J_a, "stale shuffle should have been dropped from J_a"
    assert agent.get_sku_id_carrying() is None, "agent must not pick up the wrong SKU"
    assert agent.status == 0 and agent.task_sequence == [], "agent should be freed"
    # The SKU that was at the cell must remain in the warehouse (not removed).
    assert pickup_loc in populated_graph.warehouse.get_full_locations()


def test_simulate_pick_place_delay_defers_pickup(
    populated_graph, minimal_stats, seeded_rng
):
    """With ``pick_place_time`` on, an agent arriving at its pickup cell must
    enter PICKING (status 3) and hold the SKU in place for ``pick_place_duration``
    ticks before the SKU is removed (status -> 2)."""
    from GT_grid_world.src.simulate import (
        simulate, STATUS_PICKING, STATUS_TO_DELIVERY, TASK_TYPE_SHUFFLE as SIM_SHUFFLE,
    )

    sku = next(
        s for s in populated_graph.warehouse.get_all_skus()
        if populated_graph.warehouse.get_sku_instance_count(s) >= 1
    )
    pickup_loc = list(populated_graph.warehouse.get_sku_instances(sku))[0]
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]

    task_id = REARRANGEMENT_TASK_ID_BASE
    deadline = 100
    J_a = {task_id: (frozenset({pickup_loc}), frozenset({goal_loc}), deadline, sku, SIM_SHUFFLE)}
    minimal_stats.add_actual_pickup_duration(task_id)
    minimal_stats.add_actual_duration(task_id)

    agent = Agent(agent_id=0, state=pickup_loc)
    agent.status = 1
    agent.task_sequence = [(task_id, pickup_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])
    populated_graph.set_occupied(pickup_loc, True)

    duration = 4
    # Tick 0: agent reaches pickup -> enters PICKING; SKU must NOT be removed yet.
    Rs, J, J_a = simulate(
        minimal_stats, populated_graph, Rs, {}, J_a, "small_test", t=0,
        pick_place_time=True, pick_place_duration=duration,
    )
    assert agent.status == STATUS_PICKING
    assert agent.get_sku_id_carrying() is None
    assert pickup_loc in populated_graph.warehouse.get_full_locations(), "SKU removed too early"

    # Wait out the pick service; on the duration-th decrement the pickup executes.
    for tick in range(1, duration + 1):
        Rs, J, J_a = simulate(
            minimal_stats, populated_graph, Rs, {}, J_a, "small_test", t=tick,
            pick_place_time=True, pick_place_duration=duration,
        )
    assert agent.status == STATUS_TO_DELIVERY, "pickup should have executed after the pick delay"
    assert agent.get_sku_id_carrying() == sku
    assert pickup_loc not in populated_graph.warehouse.get_full_locations()


def test_simulate_pick_place_delay_defers_delivery(
    populated_graph, minimal_stats, seeded_rng
):
    """With ``pick_place_time`` on, an agent arriving at its delivery cell must
    enter PLACING (status 4) and wait ``pick_place_duration`` ticks before the
    task is completed."""
    from GT_grid_world.src.simulate import (
        simulate, STATUS_PLACING, STATUS_FREE, TASK_TYPE_INBOUND,
    )

    sku = next(iter(populated_graph.warehouse.get_all_skus()))
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]
    start_loc = list(populated_graph.driveway.get_full_locations() or [(0, 0)])[0]

    task_id = 7
    deadline = 100
    J = {task_id: (frozenset({start_loc}), frozenset({goal_loc}), deadline, sku, TASK_TYPE_INBOUND)}
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=goal_loc)
    agent.status = 2
    agent.set_sku_id_carrying(sku)
    agent.task_sequence = [(task_id, start_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])

    duration = 4
    # Tick 0: agent reaches delivery -> enters PLACING; task NOT completed yet.
    Rs, J, J_a = simulate(
        minimal_stats, populated_graph, Rs, J, {}, "small_test", t=0,
        pick_place_time=True, pick_place_duration=duration,
    )
    assert agent.status == STATUS_PLACING
    assert task_id in J, "task completed before the place delay elapsed"
    assert agent.get_sku_id_carrying() == sku

    for tick in range(1, duration + 1):
        Rs, J, J_a = simulate(
            minimal_stats, populated_graph, Rs, J, {}, "small_test", t=tick,
            pick_place_time=True, pick_place_duration=duration,
        )
    assert task_id not in J, "task should complete after the place delay"
    assert agent.status == STATUS_FREE
    assert agent.get_sku_id_carrying() is None


def test_simulate_outbound_blocked_when_buffer_full(
    populated_graph, minimal_stats, seeded_rng
):
    """A full output buffer must block an OUTBOUND delivery (agent waits, task
    stays open), then allow it once there is room."""
    from GT_grid_world.src.simulate import simulate, STATUS_TO_DELIVERY, STATUS_FREE, TASK_TYPE_OUTBOUND
    from GT_grid_world.src.output_buffer import OutputBuffer

    sku = next(iter(populated_graph.warehouse.get_all_skus()))
    start_loc = list(populated_graph.warehouse.get_sku_instances(sku) or [(0, 0)])[0]
    goal_loc = list(populated_graph.driveway.get_empty_locations())[0]

    task_id = 11
    deadline = 100
    J = {task_id: (frozenset({start_loc}), frozenset({goal_loc}), deadline, sku, TASK_TYPE_OUTBOUND)}
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=goal_loc)
    agent.status = 2
    agent.set_sku_id_carrying(sku)
    agent.task_sequence = [(task_id, start_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])

    # Full buffer that never drains (rate 0) -> outbound stays blocked.
    full_buffer = OutputBuffer(capacity=1, consumption_rate_per_min=0.0, level=1.0)
    Rs, J, J_a = simulate(
        minimal_stats, populated_graph, Rs, J, {}, "small_test", t=0,
        output_buffer=full_buffer,
    )
    assert task_id in J, "outbound delivery should be blocked by the full buffer"
    assert agent.status == STATUS_TO_DELIVERY
    assert agent.get_sku_id_carrying() == sku

    # Buffer with room -> the same delivery now completes and is recorded.
    open_buffer = OutputBuffer(capacity=5, consumption_rate_per_min=0.0, level=0.0)
    Rs, J, J_a = simulate(
        minimal_stats, populated_graph, Rs, J, {}, "small_test", t=1,
        output_buffer=open_buffer,
    )
    assert task_id not in J, "outbound delivery should complete once the buffer has room"
    assert agent.status == STATUS_FREE
    assert open_buffer.level == 1.0, "completed outbound must be recorded into the buffer"


def test_simulate_inbound_unaffected_by_full_buffer(
    populated_graph, minimal_stats, seeded_rng
):
    """A full output buffer must NOT block inbound deliveries -- the buffer is
    outbound-only."""
    from GT_grid_world.src.simulate import simulate, STATUS_FREE, TASK_TYPE_INBOUND
    from GT_grid_world.src.output_buffer import OutputBuffer

    sku = next(iter(populated_graph.warehouse.get_all_skus()))
    goal_loc = list(populated_graph.warehouse.get_empty_locations())[0]
    start_loc = list(populated_graph.driveway.get_full_locations() or [(0, 0)])[0]

    task_id = 13
    deadline = 100
    J = {task_id: (frozenset({start_loc}), frozenset({goal_loc}), deadline, sku, TASK_TYPE_INBOUND)}
    minimal_stats.add_task_release(task_id, 0)
    minimal_stats.add_task_deadline(task_id, deadline)
    minimal_stats.add_actual_distance(task_id)
    minimal_stats.add_actual_pickup_distance(task_id)
    minimal_stats.add_actual_duration(task_id)
    minimal_stats.add_actual_pickup_duration(task_id)

    agent = Agent(agent_id=0, state=goal_loc)
    agent.status = 2
    agent.set_sku_id_carrying(sku)
    agent.task_sequence = [(task_id, start_loc, goal_loc, deadline)]
    Rs = AgentLoader([agent])

    full_buffer = OutputBuffer(capacity=1, consumption_rate_per_min=0.0, level=1.0)
    Rs, J, J_a = simulate(
        minimal_stats, populated_graph, Rs, J, {}, "small_test", t=0,
        output_buffer=full_buffer,
    )
    assert task_id not in J, "inbound delivery must not be blocked by a full output buffer"
    assert agent.status == STATUS_FREE
    assert full_buffer.level == 1.0, "inbound completion must not touch the buffer"


# ---------------------------------------------------------------------------
# J_r consumer: fakes
# ---------------------------------------------------------------------------
class _FakeSku:
    def __init__(self, sku_id):
        self.sku_id = sku_id


class _FakeWarehouse:
    def __init__(self, loc_to_sku):
        self._loc_to_sku = loc_to_sku

    def get_sku_at_location(self, loc):
        sku = self._loc_to_sku.get(loc)
        return _FakeSku(sku) if sku is not None else None


class _FakeGraph:
    def __init__(self, loc_to_sku):
        self.warehouse = _FakeWarehouse(loc_to_sku)


class _FakeAgent:
    def __init__(self, task_sequence=None, status=0):
        self.task_sequence = list(task_sequence or [])
        self.status = status
        self.path_sequence = []


class _FakeRs:
    def __init__(self, agents):
        self.agents = agents


def test_merge_reallocation_tasks_dedups_by_sku_and_uses_high_ids():
    """Two candidates for the SAME sku produce a single J entry; ids start at the
    rearrangement base; the 4-tuple becomes a 5-tuple type=2 J entry."""
    s1, g1 = (5, 2), (3, 2)
    s2, g2 = (6, 4), (2, 4)
    G = _FakeGraph({s1: 7, s2: 7})  # both starts hold sku 7
    Ta = {
        0: ({(s1, g1)}, 0, 50, TASK_TYPE_SHUFFLE),
        1: ({(s2, g2)}, 0, 60, TASK_TYPE_SHUFFLE),  # same sku 7 -> deduped
    }
    J = {}
    next_id = add_reallocation_tasks_to_J_a(Ta, J, G, REARRANGEMENT_TASK_ID_BASE)

    assert len(J) == 1
    task_id, entry = next(iter(J.items()))
    assert task_id == REARRANGEMENT_TASK_ID_BASE
    assert next_id == REARRANGEMENT_TASK_ID_BASE + 1
    starts, goals, deadline, sku, ttype = entry
    assert ttype == TASK_TYPE_SHUFFLE and sku == 7
    assert starts == frozenset({s1}) and goals == frozenset({g1})
    assert deadline == 50


def test_merge_skips_sku_already_active_in_J():
    """A candidate whose sku already has a live rearrangement task in J is skipped."""
    s, g = (5, 2), (3, 2)
    G = _FakeGraph({s: 7})
    J = {REARRANGEMENT_TASK_ID_BASE: (frozenset({(9, 2)}), frozenset({(4, 2)}), 40, 7, TASK_TYPE_SHUFFLE)}
    Ta = {0: ({(s, g)}, 0, 50, TASK_TYPE_SHUFFLE)}

    next_id = add_reallocation_tasks_to_J_a(Ta, J, G, REARRANGEMENT_TASK_ID_BASE + 1)
    assert len(J) == 1  # nothing added
    assert next_id == REARRANGEMENT_TASK_ID_BASE + 1


def test_drop_expired_rearrangements_removes_past_window_and_resets_agent():
    """A committed (status=1, heading to pickup) shuffle whose deadline has passed
    is dropped from J and from the agent's sequence, and the agent is reset."""
    rid = REARRANGEMENT_TASK_ID_BASE
    shuffle_task = (rid, (5, 2), (3, 2), 40)
    agent = _FakeAgent(task_sequence=[shuffle_task], status=1)
    Rs = _FakeRs([agent])
    J = {rid: (frozenset({(5, 2)}), frozenset({(3, 2)}), 40, 7, TASK_TYPE_SHUFFLE)}

    drop_expired_rearrangements(J, Rs, t=40)  # t >= deadline -> expired

    assert rid not in J
    assert agent.task_sequence == []
    assert agent.status == 0


def test_drop_expired_keeps_carried_shuffle():
    """An agent already carrying the shuffled SKU (status=2) finishes the move
    even if the window passed -- it must not be stranded mid-carry."""
    rid = REARRANGEMENT_TASK_ID_BASE
    shuffle_task = (rid, (5, 2), (3, 2), 40)
    agent = _FakeAgent(task_sequence=[shuffle_task], status=2)
    Rs = _FakeRs([agent])
    J = {rid: (frozenset({(5, 2)}), frozenset({(3, 2)}), 40, 7, TASK_TYPE_SHUFFLE)}

    drop_expired_rearrangements(J, Rs, t=50)

    assert rid in J
    assert agent.task_sequence == [shuffle_task]


def test_prune_uncommitted_rearrangements_drops_only_unallocated():
    """Type=2 tasks not in any agent sequence are pruned; committed ones remain.
    Real (non-type-2) tasks are never touched."""
    committed_id = REARRANGEMENT_TASK_ID_BASE
    orphan_id = REARRANGEMENT_TASK_ID_BASE + 1
    agent = _FakeAgent(task_sequence=[(committed_id, (5, 2), (3, 2), 40)], status=1)
    Rs = _FakeRs([agent])
    J = {
        committed_id: (frozenset({(5, 2)}), frozenset({(3, 2)}), 40, 7, TASK_TYPE_SHUFFLE),
        orphan_id: (frozenset({(6, 4)}), frozenset({(2, 4)}), 50, 8, TASK_TYPE_SHUFFLE),
        1: (frozenset({(9, 0)}), frozenset({(0, 0)}), 100, 3, 0),  # real OB, untouched
    }

    prune_uncommitted_rearrangements(J, Rs)

    assert committed_id in J
    assert orphan_id not in J
    assert 1 in J
