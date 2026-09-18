"""Entry point for the ``slurm-top`` / ``stop`` commands.

Dispatch happens before ``slurm_top.app`` is imported so that ``--json`` works
in an interpreter without ``textual`` installed -- the VS Code extension may be
pointed at a bare system Python.
"""

import sys
from typing import List, Optional

from . import __version__

USAGE = """usage: slurm-top [--json [JSON OPTIONS]] [--version] [--help]

Run the terminal dashboard, or with --json print cluster state as JSON for
another front end (such as the VS Code extension).

  slurm-top                     run the TUI
  slurm-top --json              print one snapshot and exit
  slurm-top --json --watch 3    stream one snapshot per line every 3 seconds
  slurm-top --json --job 1234   details for one job
  slurm-top --json --node n01   details for one node
  slurm-top --json --node n01 --probe-cpu
                                read that node's CPU model and clock
  slurm-top --json --pin 1234   pin a job to the top of both front ends
  slurm-top --json --help       all JSON options
"""


def main(argv: Optional[List[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "--json":
        from .export import main as export_main

        raise SystemExit(export_main(args[1:]))

    if args and args[0] in ("-h", "--help"):
        sys.stdout.write(USAGE)
        raise SystemExit(0)

    if args and args[0] in ("-V", "--version"):
        sys.stdout.write(f"slurm-monitor-top {__version__}\n")
        raise SystemExit(0)

    if args:
        sys.stderr.write(f"slurm-top: unrecognized arguments: {' '.join(args)}\n\n")
        sys.stderr.write(USAGE)
        raise SystemExit(2)

    from .app import main as app_main

    app_main()


__all__ = ["main"]
