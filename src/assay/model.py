"""Training-run economics.

Every function here is pure and takes seconds and dollars. The CLI does the
unit wrangling; this module does the arithmetic, so the arithmetic can be
tested without going near argument parsing or terminal rendering.

The one non-obvious result is `spot_wall_time`. Preemptions cost time, and the
time they cost is itself proportional to the wall time you are solving for, so
the naive formulation is implicit. It has a closed form — see below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SEC_PER_HOUR = 3600.0


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def checkpoint_overhead_ratio(write_sec: float, interval_sec: float) -> float:
    """Fraction of extra wall time spent writing checkpoints.

    A run that computes for `interval_sec` then writes for `write_sec` spends
    `write_sec / interval_sec` extra time per unit of useful work.
    """
    if interval_sec <= 0:
        raise ValueError("checkpoint interval must be positive")
    return write_sec / interval_sec


def optimal_checkpoint_interval(write_sec: float, preempt_per_hour: float) -> float | None:
    """Young/Daly optimal checkpoint interval, in seconds.

        T* = sqrt(2 · c · MTBF)

    Returns None when the failure rate is zero: with nothing to recover from,
    the optimal interval is unbounded and the question is meaningless. Callers
    should say so rather than print a number.

    The approximation assumes T* is small relative to MTBF. We flag the case
    where it is not, because there the formula stops being trustworthy.
    """
    if preempt_per_hour <= 0:
        return None
    mtbf = SEC_PER_HOUR / preempt_per_hour
    return math.sqrt(2.0 * write_sec * mtbf)


# --------------------------------------------------------------------------
# wall time
# --------------------------------------------------------------------------

def on_demand_wall_time(compute_sec: float, write_sec: float, interval_sec: float) -> float:
    """Wall time with no preemptions: useful compute plus checkpoint writes."""
    return compute_sec * (1.0 + checkpoint_overhead_ratio(write_sec, interval_sec))


def spot_wall_time(
    compute_sec: float,
    write_sec: float,
    interval_sec: float,
    restart_sec: float,
    preempt_per_hour: float,
) -> float | None:
    """Expected wall time on a preemptible instance, in seconds.

    Each preemption costs, in expectation, half a checkpoint interval of lost
    work plus a restart. Preemptions arrive at rate λ over the wall time W, so

        W = base + λ·W·(T/2 + r)

    where base is the on-demand wall time. W appears on both sides; solving:

        W = base / (1 − λ·(T/2 + r))

    Returns None when λ·(T/2 + r) >= 1. That is not an error — it means
    recovery consumes time at least as fast as the run produces it, so the run
    never finishes. "Spot is not viable at this preemption rate" is the answer,
    and the caller should print it as a verdict.
    """
    base = on_demand_wall_time(compute_sec, write_sec, interval_sec)
    lam = preempt_per_hour / SEC_PER_HOUR
    drag = lam * (interval_sec / 2.0 + restart_sec)
    if drag >= 1.0:
        return None
    return base / (1.0 - drag)


def break_even_preempt_rate(
    on_demand_price: float,
    spot_price: float,
    interval_sec: float,
    restart_sec: float,
) -> float | None:
    """Preemption rate (per hour) at which spot stops being cheaper.

    Spot wins while  spot_price · W_spot  <  on_demand_price · W_on_demand.
    Both share the same base, so the bases cancel and

        λ* = (1 − spot_price/on_demand_price) / (T/2 + r)

    Returns None when spot is not cheaper per hour in the first place — then
    there is no rate at which it wins, and the answer is "don't".
    """
    if on_demand_price <= 0:
        raise ValueError("on-demand price must be positive")
    if spot_price >= on_demand_price:
        return None
    ratio = 1.0 - spot_price / on_demand_price
    return ratio / (interval_sec / 2.0 + restart_sec) * SEC_PER_HOUR


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunCost:
    """Cost decomposition for one placement."""
    wall_sec: float
    total: float
    compute: float
    checkpoint: float
    recovery: float          # restart time after preemptions
    lost_work: float         # work redone since the last checkpoint
    preemptions: float

    @property
    def wall_hours(self) -> float:
        return self.wall_sec / SEC_PER_HOUR


def price_run(
    compute_sec: float,
    write_sec: float,
    interval_sec: float,
    restart_sec: float,
    preempt_per_hour: float,
    price_per_gpu_hour: float,
    gpu_count: int,
) -> RunCost | None:
    """Full cost decomposition. None when the run cannot finish (see spot_wall_time)."""
    wall = spot_wall_time(compute_sec, write_sec, interval_sec, restart_sec, preempt_per_hour)
    if wall is None:
        return None

    node_rate = price_per_gpu_hour * gpu_count / SEC_PER_HOUR   # $/second
    lam = preempt_per_hour / SEC_PER_HOUR
    preemptions = lam * wall

    checkpoint_sec = compute_sec * checkpoint_overhead_ratio(write_sec, interval_sec)
    lost_sec = preemptions * interval_sec / 2.0
    recovery_sec = preemptions * restart_sec

    return RunCost(
        wall_sec=wall,
        total=wall * node_rate,
        compute=compute_sec * node_rate,
        checkpoint=checkpoint_sec * node_rate,
        recovery=recovery_sec * node_rate,
        lost_work=lost_sec * node_rate,
        preemptions=preemptions,
    )


def cost_per_1b_tokens(total: float, total_tokens: float) -> float | None:
    if total_tokens <= 0:
        return None
    return total / (total_tokens / 1e9)


def cost_per_1k_steps(total: float, steps: float) -> float | None:
    if steps <= 0:
        return None
    return total / (steps / 1000.0)
