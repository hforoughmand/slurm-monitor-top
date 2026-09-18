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
    OUTPUT_TAIL_LINES,
    job_output,
    job_output_paths,
    job_usage_metrics,
    cpu_counts,
    cpu_speed,
    cpu_topology,
    describe_cpu,
    describe_cpu_count,
    describe_cpu_speeds,
    load_cpu_info,
    load_pinned_jobs,
    parse_disks,
    parse_sinfo,
    parse_squeue,
    probe_node_cpu,
    set_job_pinned,
    short_cpu_model,
    sort_jobs,
    summarize_gpus,
    summarize_jobs,
    toggle_job_pin,
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


def cpu_dict(node: Node, cpu_info: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """What a node's processors are, as far as we know.

    ``sockets``/``cores``/``threads`` always come from Slurm. ``model`` and
    ``speed`` are filled in only once someone has probed that node, because
    Slurm does not carry them at all -- see :func:`slurm_top.data.probe_node_cpu`.
    """
    info = (cpu_info or {}).get(node.name) or {}
    speed, speed_kind = cpu_speed(info)
    counts = cpu_counts(node.sockets, node.cores_per_socket, node.threads_per_core, node.cpus_total)
    return {
        "sockets": _parse_int(node.sockets),
        "cores_per_socket": _parse_int(node.cores_per_socket),
        "threads_per_core": _parse_int(node.threads_per_core),
        # Derived so a front end never has to decide whether "CPUs" means
        # cores or threads: `logical` is what Slurm allocates, `cores` is the
        # silicon, `processors` is how many chips are in the machine.
        "cpus_total": counts["logical"],
        "cores": counts["cores"],
        "processors": counts["processors"],
        "count_summary": describe_cpu_count(counts, info),
        "speeds": [{"label": label, "value": value} for label, value in describe_cpu_speeds(info)],
        "topology": cpu_topology(node),
        "model": info.get("model", ""),
        "model_short": short_cpu_model(info),
        "speed": speed,
        # "nominal" (from the model name), "max" (a boost ceiling) or
        # "current" (whatever the governor was doing); empty when unknown.
        "speed_kind": speed_kind,
        "vendor": info.get("vendor", ""),
        "arch": info.get("arch", ""),
        "summary": describe_cpu(info),
        "source": info.get("source", ""),
        "probed_at": _parse_int(info.get("probed_at", "")),
        "known": bool(info.get("model")),
    }


def node_dict(node: Node, cpu_info: "Dict[str, Dict[str, str]] | None" = None) -> Dict[str, Any]:
    """A machine with its CPU, memory and GPU columns as numbers.

    Everything a front end would otherwise have to derive: the GRES strings
    expanded into per-type counts, memory in both MB and human form, and load
    as a fraction of the node's cores so it can be drawn as a bar.
    """
    row = dataclasses.asdict(node)
    row["cpu"] = cpu_dict(node, cpu_info if cpu_info is not None else load_cpu_info())
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
    # Both are file reads that only re-parse when the file changed, so they
    # cost nothing on a 3-second poll.
    cpu_info = load_cpu_info()
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": time.time(),
        "user": user,
        "host": socket.gethostname(),
        "pinned": load_pinned_jobs(),
        "jobs": [job_dict(j) for j in jobs],
        "nodes": [node_dict(n, cpu_info) for n in nodes],
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
        # Requested against used, ready to draw; see `job_usage_metrics`.
        "metrics": job_usage_metrics(job, detail, usage),
        # Where the job's output goes, and whether the file is there. The
        # contents are a separate request (`--job-output`): a detail lookup
        # runs every few seconds and must not read a log file each time.
        "output": job_output_paths(detail, job),
    }


def job_output_detail(job_id: str, stream: str, lines: int) -> Dict[str, Any]:
    """The tail of one of a job's output files."""
    payload = job_output(job_id, stream=stream, lines=lines)
    payload.update({"schema": SCHEMA_VERSION, "timestamp": time.time(), "kind": "job-output"})
    return payload


def node_detail(node_name: str, probe_cpu: bool = False, allow_srun: bool = True) -> Dict[str, Any]:
    """Everything the node-details popup shows, for one node.

    ``probe_cpu`` goes and reads the CPU model off the node, which costs an ssh
    connection or a one-second job; it is never done for a plain lookup.
    """
    detail, jobs = collect_node_info(node_name)
    cpu_error = ""
    if probe_cpu:
        _, cpu_error = probe_node_cpu(node_name, allow_srun=allow_srun)
    info = load_cpu_info().get(node_name, {})
    speed, speed_kind = cpu_speed(info)
    counts = cpu_counts(
        detail.get("Sockets", ""),
        detail.get("CoresPerSocket", ""),
        detail.get("ThreadsPerCore", ""),
        detail.get("CPUTot", ""),
    )
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": time.time(),
        "kind": "node",
        "node": node_name,
        "detail": detail,
        "jobs": [job_dict(j) for j in jobs],
        "cpu": {
            "sockets": counts["processors"],
            "cores_per_socket": counts["cores_per_processor"],
            "threads_per_core": counts["threads_per_core"],
            "cpus_total": counts["logical"],
            "cores": counts["cores"],
            "processors": counts["processors"],
            "count_summary": describe_cpu_count(counts, info),
            "speeds": [{"label": label, "value": value} for label, value in describe_cpu_speeds(info)],
            "model": info.get("model", ""),
            "model_short": short_cpu_model(info),
            "speed": speed,
            "speed_kind": speed_kind,
            "vendor": info.get("vendor", ""),
            "arch": info.get("arch", "") or detail.get("Arch", ""),
            "summary": describe_cpu(info),
            "source": info.get("source", ""),
            "probed_at": _parse_int(info.get("probed_at", "")),
            "known": bool(info.get("model")),
            "error": cpu_error,
        },
    }


def pin(job_id: str, action: str) -> Dict[str, Any]:
    """Pin, unpin or toggle one job, and report the new set."""
    if action == "toggle":
        pinned, is_pinned = toggle_job_pin(job_id)
    else:
        is_pinned = action == "pin"
        pinned = set_job_pinned(job_id, is_pinned)
    return {
        "schema": SCHEMA_VERSION,
        "timestamp": time.time(),
        "kind": "pin",
        "job_id": job_id,
        "pinned": pinned,
        "is_pinned": is_pinned,
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
    parser.add_argument(
        "--job-output",
        metavar="JOBID",
        help="emit the tail of one job's output file and exit. Slurm keeps the "
             "path but not the contents, so this reads the file itself and can "
             "only see what the login node can see.",
    )
    parser.add_argument(
        "--stream",
        choices=("stdout", "stderr"),
        default="stdout",
        help="with --job-output: which of the two files to read (default: stdout)",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=OUTPUT_TAIL_LINES,
        metavar="N",
        help=f"with --job-output: how many trailing lines to read (default: {OUTPUT_TAIL_LINES})",
    )
    parser.add_argument("--node", metavar="NAME", help="emit details for one node and exit")
    parser.add_argument(
        "--probe-cpu",
        action="store_true",
        help="with --node: read the CPU model and clock off the node itself "
             "(over ssh, else a one-second job) and remember it. Slurm does not "
             "report either, so this is the only way to get them.",
    )
    parser.add_argument(
        "--no-srun",
        action="store_true",
        help="with --probe-cpu: try ssh only, never submit a job",
    )
    parser.add_argument(
        "--pin",
        metavar="JOBID",
        help="pin a job so both front ends list it first, then exit",
    )
    parser.add_argument("--unpin", metavar="JOBID", help="remove a pin, then exit")
    parser.add_argument(
        "--toggle-pin", metavar="JOBID", help="flip one job's pin, then exit"
    )
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

    for flag, action in (("pin", "pin"), ("unpin", "unpin"), ("toggle_pin", "toggle")):
        job_id = getattr(args, flag)
        if job_id:
            _emit(pin(job_id, action), args.pretty)
            return 0

    if args.job:
        _emit(job_detail(args.job), args.pretty)
        return 0
    if args.job_output:
        _emit(job_output_detail(args.job_output, args.stream, args.lines), args.pretty)
        return 0
    if args.node:
        _emit(
            node_detail(args.node, probe_cpu=args.probe_cpu, allow_srun=not args.no_srun),
            args.pretty,
        )
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
