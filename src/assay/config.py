"""Run description: loading, validation, derived quantities.

Input is YAML or JSON. Uses PyYAML if it is installed; if not, a fallback
parser handles the flat `key: value` subset these files use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SEC_PER_HOUR = 3600.0

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _coerce(raw: str):
    v = raw.strip()
    if not v or v in {"null", "~", "None"}:
        return None
    if v.lower() in {"true", "yes"}:
        return True
    if v.lower() in {"false", "no"}:
        return False
    if _NUM.match(v):
        return float(v) if any(c in v for c in ".eE") else int(v)
    return v.strip("'\"")


def _mini_yaml(text: str) -> dict:
    """Flat `key: value` YAML, with comments and blank lines. Nothing more."""
    out: dict = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if "#" in line:
            line = line[: line.index("#")]
        if not line.strip():
            continue
        if line[0].isspace():
            raise ValueError(
                f"line {lineno}: nested keys need PyYAML installed "
                f"(pip install pyyaml), or use JSON"
            )
        if ":" not in line:
            raise ValueError(f"line {lineno}: expected 'key: value', got {line!r}")
        key, _, val = line.partition(":")
        out[key.strip()] = _coerce(val)
    return out


def load(path: str | Path) -> dict:
    text = Path(path).read_text()
    if str(path).endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # optional
        return yaml.safe_load(text) or {}
    except ImportError:
        return _mini_yaml(text)


@dataclass(frozen=True)
class Run:
    gpu_count: int
    price_per_gpu_hour: float
    spot_price_per_gpu_hour: float | None
    step_time_sec: float
    target_steps: int
    checkpoint_write_sec: float
    checkpoint_interval_sec: float
    restart_sec: float
    preempt_per_hour: float | None      # None = unmeasured
    seq_len: int | None
    global_batch: int | None
    gpu_type: str
    provider: str

    @property
    def compute_sec(self) -> float:
        """Useful compute only — no checkpoint, restart or lost work."""
        return self.step_time_sec * self.target_steps

    @property
    def total_tokens(self) -> float | None:
        if not self.seq_len or not self.global_batch:
            return None
        return float(self.seq_len) * self.global_batch * self.target_steps

    @property
    def node_price_per_hour(self) -> float:
        return self.price_per_gpu_hour * self.gpu_count


_REQUIRED = ("gpu_count", "price_per_gpu_hour", "step_time_sec",
             "checkpoint_write_sec", "restart_sec")


def parse(d: dict) -> Run:
    missing = [k for k in _REQUIRED if d.get(k) is None]
    if missing:
        raise SystemExit(f"assay: missing required field(s): {', '.join(missing)}")

    steps = d.get("target_steps")
    if steps is None:
        tokens, seq, batch = d.get("target_tokens"), d.get("seq_len"), d.get("global_batch")
        if not (tokens and seq and batch):
            raise SystemExit(
                "assay: need target_steps, or target_tokens with seq_len and global_batch"
            )
        steps = int(round(float(tokens) / (float(seq) * float(batch))))

    # Interval may be given in steps or seconds; steps is what a config knows.
    if d.get("checkpoint_interval_sec"):
        interval = float(d["checkpoint_interval_sec"])
    elif d.get("checkpoint_interval_steps"):
        interval = float(d["checkpoint_interval_steps"]) * float(d["step_time_sec"])
    else:
        raise SystemExit(
            "assay: need checkpoint_interval_steps or checkpoint_interval_sec"
        )
    if interval <= 0:
        raise SystemExit("assay: checkpoint interval must be positive")

    return Run(
        gpu_count=int(d["gpu_count"]),
        price_per_gpu_hour=float(d["price_per_gpu_hour"]),
        spot_price_per_gpu_hour=(
            float(d["spot_price_per_gpu_hour"])
            if d.get("spot_price_per_gpu_hour") is not None else None
        ),
        step_time_sec=float(d["step_time_sec"]),
        target_steps=int(steps),
        checkpoint_write_sec=float(d["checkpoint_write_sec"]),
        checkpoint_interval_sec=interval,
        restart_sec=float(d["restart_sec"]),
        preempt_per_hour=(
            float(d["preemptions_per_hour"])
            if d.get("preemptions_per_hour") is not None else None
        ),
        seq_len=int(d["seq_len"]) if d.get("seq_len") else None,
        global_batch=int(d["global_batch"]) if d.get("global_batch") else None,
        gpu_type=str(d.get("gpu_type", "GPU")),
        provider=str(d.get("provider", "this provider")),
    )
