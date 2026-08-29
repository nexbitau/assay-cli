# assay

**What a training run costs, and what it would cost somewhere else.**

Most teams choose where to train from a price-per-GPU-hour table. That number
is wrong once checkpoint writes, restarts and preemption losses land on it.
`assay` computes the real figure, and tells you whether spot is economically
rational for *this* workload rather than in general.

Part of [Assay](https://assay.nexbit.au).

```console
$ assay analyze run.yaml
Current  8 × H100-SXM-80GB on-demand @ $3.70/GPU-h

Expected total cost                                 $9,512
  compute                                           $9,077
  checkpoint overhead (4.8%)                          $434

Wall time                                          321.3 h
Cost / 1B tokens                                     $7.56
```

```console
$ assay spot run.yaml
Spot  @ $1.45/GPU-h   (λ assumed, see below)
  GPU price saving                                    -61%
  expected preemptions                                60.9
  expected lost work                                 +$277
  ...
  expected total                                    $4,040
  expected saving vs on-demand                       57.5%

Optimal checkpoint interval                         39 min
Break-even preemption rate                        1.38 /hr

!  Preemption rate for provider-a: ESTIMATED — no measured data
   Range at λ = 0.05–0.30/hr      $3,812 – $4,297
   Decision spread                        $485
   A short measured canary on this provider collapses this range.
```

## Install

```bash
pip install assay-cli
```

No runtime dependencies. PyYAML is used if you happen to have it; otherwise a
small built-in reader handles the flat `key: value` files below. JSON works too.

## Run description

```yaml
gpu_type: H100-SXM-80GB
gpu_count: 8
provider: provider-a
price_per_gpu_hour: 3.70
spot_price_per_gpu_hour: 1.45

step_time_sec: 1.84
target_steps: 600000          # or target_tokens, with seq_len and global_batch
seq_len: 8192
global_batch: 256

checkpoint_write_sec: 135
checkpoint_interval_steps: 1533
restart_sec: 180

preemptions_per_hour: null
```

## The model

Checkpoint overhead is `write / interval` extra wall time per unit of work.

Preemptions cost time proportional to the wall time you are solving for, so the
naive formulation is implicit. Solving it:

```
W = compute · (1 + c/T) / (1 − λ·(T/2 + r))
```

for checkpoint write `c`, interval `T`, restart `r` and preemption rate `λ`.
When `λ·(T/2 + r) ≥ 1`, recovery consumes wall time at least as fast as the run
produces it and the run never completes. `assay` reports that as a verdict
rather than an error.

Optimal checkpoint interval is Young/Daly, `T* = sqrt(2·c·MTBF)`. The
approximation degrades when `T*` approaches MTBF, and `assay` says so.

Break-even preemption rate has a closed form, since both placements share a
base wall time:

```
λ* = (1 − spot_price/on_demand_price) / (T/2 + r)
```

## What it will not do

Guess at a number and present it as fact. `preemptions_per_hour` and
`checkpoint_write_sec` are provider-specific and unpublished. Leave the first
null and `assay` prints the cost range your uncertainty produces, and what it is
worth to measure.

It makes no network calls. The only file it reads is the run description you
pass on the command line.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

## License

Apache-2.0.
