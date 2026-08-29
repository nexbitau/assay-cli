import math

import pytest

from assay import config, model


# --------------------------------------------------------------------- basics

def test_checkpoint_overhead_is_write_over_interval():
    assert model.checkpoint_overhead_ratio(100, 1000) == pytest.approx(0.1)


def test_zero_interval_rejected():
    with pytest.raises(ValueError):
        model.checkpoint_overhead_ratio(100, 0)


def test_on_demand_wall_adds_checkpoint_time():
    assert model.on_demand_wall_time(1000, 10, 100) == pytest.approx(1100)


# ------------------------------------------------------------------ spot wall

def test_no_preemptions_matches_on_demand():
    base = model.on_demand_wall_time(1000, 10, 100)
    assert model.spot_wall_time(1000, 10, 100, 60, 0.0) == pytest.approx(base)


def test_spot_wall_solves_the_implicit_equation():
    """W must satisfy W = base + lambda*W*(T/2 + r) — the fixed point."""
    compute, c, T, r, lam = 1_104_000, 135, 2820.0, 180, 0.15
    W = model.spot_wall_time(compute, c, T, r, lam)
    base = model.on_demand_wall_time(compute, c, T)
    assert W == pytest.approx(base + (lam / 3600) * W * (T / 2 + r))


def test_higher_preemption_rate_costs_more_time():
    a = model.spot_wall_time(1_000_000, 100, 1800, 120, 0.05)
    b = model.spot_wall_time(1_000_000, 100, 1800, 120, 0.25)
    assert b > a


def test_run_that_cannot_finish_returns_none():
    """lambda*(T/2 + r) >= 1: recovery outpaces progress."""
    assert model.spot_wall_time(1_000_000, 100, 7200, 600, 1.0) is None


def test_price_run_propagates_the_non_viable_case():
    assert model.price_run(1_000_000, 100, 7200, 600, 1.0, 1.45, 8) is None


# ----------------------------------------------------------------- break-even

def test_break_even_makes_the_two_placements_cost_the_same():
    od, sp, T, r = 3.70, 1.45, 2820.0, 180
    lam = model.break_even_preempt_rate(od, sp, T, r)
    spot = model.price_run(1_104_000, 135, T, r, lam, sp, 8)
    od_wall = model.on_demand_wall_time(1_104_000, 135, T)
    assert spot.total == pytest.approx(od_wall * od * 8 / 3600, rel=1e-9)


def test_no_break_even_when_spot_is_not_cheaper():
    assert model.break_even_preempt_rate(2.00, 2.00, 1800, 120) is None
    assert model.break_even_preempt_rate(2.00, 3.00, 1800, 120) is None


# ------------------------------------------------------------- young and daly

def test_optimal_interval_matches_closed_form():
    lam = 0.15
    mtbf = 3600 / lam
    assert model.optimal_checkpoint_interval(135, lam) == pytest.approx(
        math.sqrt(2 * 135 * mtbf))


def test_optimal_interval_undefined_without_failures():
    assert model.optimal_checkpoint_interval(135, 0) is None


def test_optimal_interval_actually_minimises_cost():
    """Sanity: perturbing T* in either direction should not be cheaper."""
    compute, c, r, lam, price, gpus = 1_104_000, 135, 180, 0.30, 1.45, 8
    t = model.optimal_checkpoint_interval(c, lam)
    at = lambda T: model.price_run(compute, c, T, r, lam, price, gpus).total
    assert at(t) <= at(t * 0.6) and at(t) <= at(t * 1.6)


# ---------------------------------------------------------------------- config

def test_target_steps_derived_from_tokens():
    run = config.parse({
        "gpu_count": 8, "price_per_gpu_hour": 3.7, "step_time_sec": 2.0,
        "checkpoint_write_sec": 100, "restart_sec": 60,
        "checkpoint_interval_steps": 500,
        "target_tokens": 2_097_152_000, "seq_len": 8192, "global_batch": 256,
    })
    assert run.target_steps == 1000


def test_interval_in_steps_becomes_seconds():
    run = config.parse({
        "gpu_count": 1, "price_per_gpu_hour": 1.0, "step_time_sec": 2.0,
        "checkpoint_write_sec": 10, "restart_sec": 5,
        "checkpoint_interval_steps": 300, "target_steps": 1000,
    })
    assert run.checkpoint_interval_sec == 600


def test_missing_required_field_exits():
    with pytest.raises(SystemExit):
        config.parse({"gpu_count": 8})


def test_mini_yaml_reads_comments_nulls_and_numbers():
    d = config._mini_yaml(
        "a: 1  # trailing\n\nb: 2.5\nc: null\nd: hello\ne: true\n")
    assert d == {"a": 1, "b": 2.5, "c": None, "d": "hello", "e": True}


def test_mini_yaml_rejects_nesting_with_a_useful_message():
    with pytest.raises(ValueError, match="PyYAML"):
        config._mini_yaml("parallelism:\n  tp: 8\n")
