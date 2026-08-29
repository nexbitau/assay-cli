"""Terminal output.

Money is never printed to more precision than it is known to. Any figure
resting on an unmeasured parameter is labelled, with the range that parameter's
plausible values produce.
"""

from __future__ import annotations

from . import model
from .config import Run

# Plausible-λ sweep used when the preemption rate has not been measured.
# Wide on purpose; a canary narrows it.
LAMBDA_LOW, LAMBDA_HIGH = 0.05, 0.30

W = 44   # label column width


def _money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def _row(label: str, value: str, indent: int = 0) -> str:
    return f"{' ' * indent}{label:<{W - indent}}{value:>14}"


def _mins(sec: float) -> str:
    return f"{sec / 60:.0f} min" if sec < 5400 else f"{sec / 3600:.1f} h"


def analyze(run: Run) -> str:
    wall = model.on_demand_wall_time(
        run.compute_sec, run.checkpoint_write_sec, run.checkpoint_interval_sec)
    rate = run.node_price_per_hour / 3600.0
    ckpt_ratio = model.checkpoint_overhead_ratio(
        run.checkpoint_write_sec, run.checkpoint_interval_sec)

    total = wall * rate
    compute = run.compute_sec * rate
    ckpt = total - compute

    out = [
        f"Current  {run.gpu_count} × {run.gpu_type} on-demand "
        f"@ ${run.price_per_gpu_hour:.2f}/GPU-h",
        "",
        _row("Expected total cost", _money(total)),
        _row("compute", _money(compute), 2),
        _row(f"checkpoint overhead ({ckpt_ratio * 100:.1f}%)", _money(ckpt), 2),
        "",
        _row("Wall time", f"{wall / 3600:,.1f} h"),
        _row("Steps", f"{run.target_steps:,}"),
    ]

    per_1k = model.cost_per_1k_steps(total, run.target_steps)
    if per_1k:
        out.append(_row("Cost / 1k steps", _money(per_1k)))
    if run.total_tokens:
        per_1b = model.cost_per_1b_tokens(total, run.total_tokens)
        out.append(_row("Tokens", f"{run.total_tokens / 1e9:,.0f} B"))
        out.append(_row("Cost / 1B tokens", _money(per_1b)))
    return "\n".join(out)


def _spot_at(run: Run, lam: float):
    return model.price_run(
        run.compute_sec, run.checkpoint_write_sec, run.checkpoint_interval_sec,
        run.restart_sec, lam, run.spot_price_per_gpu_hour, run.gpu_count)


def spot(run: Run) -> str:
    if run.spot_price_per_gpu_hour is None:
        return ("assay: no spot_price_per_gpu_hour in the run description — "
                "nothing to compare against.")

    on_demand = model.on_demand_wall_time(
        run.compute_sec, run.checkpoint_write_sec, run.checkpoint_interval_sec)
    od_total = on_demand * run.node_price_per_hour / 3600.0

    measured = run.preempt_per_hour is not None
    lam = run.preempt_per_hour if measured else (LAMBDA_LOW + LAMBDA_HIGH) / 2

    out = [
        f"Current  {run.gpu_count} × {run.gpu_type} on-demand "
        f"@ ${run.price_per_gpu_hour:.2f}/GPU-h",
        _row("Expected total cost", _money(od_total)),
        "",
        f"Spot  @ ${run.spot_price_per_gpu_hour:.2f}/GPU-h"
        + ("" if measured else "   (λ assumed, see below)"),
    ]

    c = _spot_at(run, lam)
    if c is None:
        out += [
            "",
            "  VERDICT: spot is not viable for this workload.",
            f"  At λ = {lam:.2f}/hr, recovery consumes wall time at least as fast",
            "  as the run produces it, so the run never completes. Lengthen the",
            "  checkpoint interval or cut restart time before reconsidering.",
        ]
        return "\n".join(out)

    price_saving = 1 - run.spot_price_per_gpu_hour / run.price_per_gpu_hour
    out += [
        _row("GPU price saving", f"-{price_saving * 100:.0f}%", 2),
        _row("expected preemptions", f"{c.preemptions:.1f}", 2),
        _row("checkpoint I/O", "+" + _money(c.checkpoint), 2),
        _row("expected lost work", "+" + _money(c.lost_work), 2),
        _row("recovery overhead", "+" + _money(c.recovery), 2),
        "  " + "─" * (W + 12),
        _row("expected total", _money(c.total), 2),
        _row("expected saving vs on-demand",
             f"{(1 - c.total / od_total) * 100:.1f}%", 2),
        "",
    ]

    t_opt = model.optimal_checkpoint_interval(run.checkpoint_write_sec, lam)
    if t_opt is None:
        out.append(_row("Optimal checkpoint interval", "n/a (λ = 0)"))
    else:
        out.append(_row("Optimal checkpoint interval", _mins(t_opt)))
        delta = _spot_at(run, lam)
        alt = model.price_run(
            run.compute_sec, run.checkpoint_write_sec, t_opt, run.restart_sec,
            lam, run.spot_price_per_gpu_hour, run.gpu_count)
        if alt and delta and alt.total < delta.total:
            out.append(f"  currently {_mins(run.checkpoint_interval_sec)} — "
                       f"costing {_money(delta.total - alt.total)} per run")
        mtbf = 3600.0 / lam
        if t_opt > 0.4 * mtbf:
            out.append("  (interval is a large fraction of MTBF; Young/Daly is "
                       "approximate here)")

    be = model.break_even_preempt_rate(
        run.price_per_gpu_hour, run.spot_price_per_gpu_hour,
        run.checkpoint_interval_sec, run.restart_sec)
    out.append(_row("Break-even preemption rate",
                    f"{be:.2f} /hr" if be else "n/a (spot not cheaper)"))

    if not measured:
        lo, hi = _spot_at(run, LAMBDA_LOW), _spot_at(run, LAMBDA_HIGH)
        lo_s = _money(lo.total) if lo else "n/a"
        hi_s = _money(hi.total) if hi else "does not finish"
        spread = (hi.total - lo.total) if (lo and hi) else None
        out += [
            "",
            f"!  Preemption rate for {run.provider}: ESTIMATED — no measured data",
            f"   Range at λ = {LAMBDA_LOW:.2f}–{LAMBDA_HIGH:.2f}/hr"
            f"{'':>6}{lo_s} – {hi_s}",
            *( [f"   Decision spread{'':>24}{_money(spread)}"] if spread else [] ),
            "   A short measured canary on this provider collapses this range.",
        ]
    return "\n".join(out)
