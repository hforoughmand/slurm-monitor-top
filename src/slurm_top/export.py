"""JSON snapshots of the cluster, for non-terminal front ends.

The VS Code extension cannot reuse the Textual widgets, but it must not
re-implement the squeue/sinfo/GRES parsing either -- the TRES and memory-unit
handling in :mod:`slurm_top.data` is the fiddly part. So the extension spawns
this module and reads the same numbers the TUI renders.

Two modes:

* one-shot  -- ``slurm-top --json`` prints a single snapshot object.
* streaming -- ``slurm-top --json --watch 3`` prints one snapshot per line
  (newline-delimited JSON) every 3 seconds until stdin closes or the process is
  killed. The extension uses this so a refresh costs no interpreter startup.
"""

import argparse
import dataclasses
import json
import os
import socket
import sys
import time
from typing import Any, Dict, List

from .data import (
    DiskUsage,
    Job,
    Node,
    _format_mb_human,
    _parse_gpu_count,
    _parse_gpu_inventory,
    _parse_gpu_per_type,
    _parse_int,
    _parse_mem_to_mb,
    collect_job_info,
    collect_node_info,
    parse_disks,
    parse_sinfo,
    parse_squeue,
    sort_jobs,
    summarize_gpus,
    summarize_jobs,
)

# Bumped when the snapshot shape changes incompatibly, so an extension talking
# to an older installed slurm-top can say so instead of rendering blanks.
SCHEMA_VERSION = 1


def current_user() -> str:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if user:
        return user
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return ""


def job_dict(job: Job) -> Dict[str, Any]:
    """A job plus the numbers the TUI derives while rendering.

    The GPU count lives in a TRES string (``cpu=16,mem=64G,gres/gpu:a100=2``)
    that only :mod:`slurm_top.data` knows how to read. Deriving it here keeps
    every front end on one parser instead of each growing its own.
    """
    row = dataclasses.asdict(job)
    row["gpu_count"] = _parse_gpu_count(job.gpus)
    row["gpu_types"] = _parse_gpu_per_type(job.gpus)
    row["cpu_count"] = _parse_int(job.ncpus)
    row["mem_mb"] = _parse_mem_to_mb(job.mem)
    return row


def node_dict(node: Node) -> Dict[str, Any]:
    """A machine with its CPU, memory and GPU columns as numbers.

    Everything a front end would otherwise have to derive: the GRES strings
    expanded into per-type counts, memory in both MB and human form, and load
    as a fraction of the node's cores so it can be drawn as a bar.
    """
    row = dataclasses.asdict(node)
    row["cpus_total_n"] = _parse_int(node.cpus_total)
    row["cpus_alloc_n"] = _parse_int(node.cpus_alloc)
    row["cpus_idle_n"] = _parse_int(node.cpus_idle)
    for key in ("mem_total", "mem_reserved", "mem_free"):
        mb = _parse_int(getattr(node, key))
        row[f"{key}_mb"] = mb
        row[f"{key}_human"] = _format_mb_human(mb)

    inventory = _parse_gpu_inventory(node.gres)
    allocated = _parse_gpu_inventory(node.gres_used)
    row["gpu_inventory"] = inventory
    row["gpu_allocated"] = allocated
    row["gpu_total"] = sum(inventory.values())
    row["gpu_used"] = sum(allocated.values())
    row["gpu_free"] = max(0, row["gpu_total"] - row["gpu_used"])
    row["gpu_types"] = sorted(inventory)

    try:
        load = float(node.cpu_load)
    except (TypeError, ValueError):
        load = 0.0
    row["cpu_load_n"] = load
    # Load above the core count means the node is oversubscribed; keep the
    # ratio uncapped so a front end can show that rather than hide it.
    row["cpu_load_ratio"] = load / row["cpus_total_n"] if row["cpus_total_n"] else 0.0
    return row


def disk_dict(disk: DiskUsage) -> Dict[str, Any]:
    """A filesystem with its sizes as numbers so a consumer can sort on them."""
    row = dataclasses.asdict(disk)
    row["usage_pct"] = _parse_int(disk.usage_percent.rstrip("%"))
    for key in ("size", "used", "avail"):
        row[f"{key}_mb"] = _parse_mem_to_mb(getattr(disk, key))
    return row


def snapshot() -> Dict[str, Any]:
    """One full cluster reading: jobs, nodes, disks and the derived summaries."""
    user = current_user()
    jobs = sort_jobs(parse_squeue())
    nodes = parse_sinfo()
    disks = parse_disks()
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": time.time(),
        "user": user,
        "host": socket.gethostname(),
        "jobs": [job_dict(j) for j in jobs],
        "nodes": [node_dict(n) for n in nodes],
        "disks": [disk_dict(d) for d in disks],
        "gpu": summarize_gpus(nodes, jobs),
        "summary": summarize_jobs(jobs, user),
    }


def job_detail(job_id: str) -> Dict[str, Any]:
    """Everything the job-details popup shows, for one job."""
    job, detail, usage = collect_job_info(job_id)
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": time.time(),
        "kind": "job",
        "job_id": job_id,
        "job": job_dict(job) if job is not None else None,
        "detail": detail,
        "usage": usage,
    }


def node_detail(node_name: str) -> Dict[str, Any]:
    """Everything the node-details popup shows, for one node."""
    detail, jobs = collect_node_info(node_name)
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": time.time(),
        "kind": "node",
        "node": node_name,
        "detail": detail,
        "jobs": [job_dict(j) for j in jobs],
    }


def _emit(payload: Dict[str, Any], pretty: bool) -> None:
    json.dump(payload, sys.stdout, indent=2 if pretty else None)
    sys.stdout.write("\n")
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurm-top --json",
        description="Print Slurm cluster state as JSON instead of running the TUI.",
    )
    parser.add_argument("--job", metavar="JOBID", help="emit details for one job and exit")
    parser.add_argument("--node", metavar="NAME", help="emit details for one node and exit")
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="keep running, emitting one snapshot per line every SECONDS",
    )
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    return parser


def main(argv: "List[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

    if args.job:
        _emit(job_detail(args.job), args.pretty)
        return 0
    if args.node:
        _emit(node_detail(args.node), args.pretty)
        return 0

    if args.watch is None:
        _emit(snapshot(), args.pretty)
        return 0

    # Streaming mode. --pretty is ignored: consumers split on newlines, and an
    # indented object would span many of them.
    interval = max(1.0, args.watch)
    while True:
        started = time.time()
        try:
            _emit(snapshot(), False)
        except BrokenPipeError:
            return 0
        except Exception as exc:  # keep the stream alive across a transient failure
            try:
                _emit({"schema": SCHEMA_VERSION, "timestamp": time.time(), "error": str(exc)}, False)
            except BrokenPipeError:
                return 0
        elapsed = time.time() - started
        time.sleep(max(0.0, interval - elapsed))


if __name__ == "__main__":
    sys.exit(main())
