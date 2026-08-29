"""assay — what a training run costs, and what it would cost somewhere else."""

from __future__ import annotations

import argparse
import sys

from . import config, render

__version__ = "0.1.0"

EPILOG = """\
examples:
  assay analyze run.yaml     what this run costs today
  assay spot run.yaml        spot verdict and checkpoint policy

assay runs entirely on your machine. It makes no network calls and writes
nothing outside the paths you give it.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="assay", description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"assay {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_ in (("analyze", "cost of the run as configured"),
                        ("spot", "spot vs on-demand, and checkpoint interval")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("run", help="path to a run description (.yaml or .json)")

    args = p.parse_args(argv)

    try:
        run = config.parse(config.load(args.run))
    except FileNotFoundError:
        print(f"assay: no such file: {args.run}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"assay: {args.run}: {e}", file=sys.stderr)
        return 2

    print(render.analyze(run) if args.cmd == "analyze" else render.spot(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
