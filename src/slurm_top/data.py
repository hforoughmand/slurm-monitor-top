"""Slurm data collection and parsing.

Pure-stdlib layer shared by the Textual TUI (:mod:`slurm_top.app`) and the JSON
exporter (:mod:`slurm_top.export`) that backs the VS Code extension. Nothing in
here may import ``rich`` or ``textual``: the exporter has to run in whatever
interpreter the editor can reach, which is often a bare system Python.
"""

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    job_id: str
    user: str
    state: str
    partition: str
    name: str
    nodes: str
    ncpus: str
    mem: str
    gpus: str
    time_used: str
    node_list: str


@dataclass
class Node:
    name: str
    state: str
    cpus_total: str
    cpus_alloc: str
    cpus_idle: str
    mem_total: str
    mem_reserved: str
    mem_free: str
    gres: str
    # Appended with defaults so existing positional construction and any
    # consumer that ignores these fields keep working.
    partition: str = ""
    cpu_load: str = ""
    reason: str = ""
    gres_used: str = ""
    # Sockets/cores/threads come free with sinfo's %z. The CPU *model* and its
    # clock do not: Slurm carries neither, so those need `probe_node_cpu`.
    sockets: str = ""
    cores_per_socket: str = ""
    threads_per_core: str = ""


@dataclass
class DiskUsage:
    usage_percent: str
    mount: str
    fs_type: str
    size: str
    used: str = ""
    avail: str = ""


# Hard cap on every external command. Without this a stuck `df` (stale NFS
# mount) or a slow slurmctld would block whatever thread the call runs on.
CMD_TIMEOUT = 15


def run_cmd(cmd: str) -> str:
    try:
        out = subprocess.check_output(
            shlex.split(cmd), stderr=subprocess.DEVNULL, text=True, timeout=CMD_TIMEOUT
        )
        return out
    except Exception:
        return ""


def run_cmd_argv(argv: List[str], timeout: float = CMD_TIMEOUT) -> str:
    try:
        return subprocess.check_output(
            argv, stderr=subprocess.DEVNULL, text=True, timeout=timeout
        )
    except Exception:
        return ""


def run_cmd_output(argv: List[str], timeout: float = CMD_TIMEOUT) -> "tuple[bool, str, str]":
    """(succeeded, stdout, stderr) - for callers that must report *why* it failed."""
    try:
        completed = subprocess.run(
            argv, check=False, text=True, capture_output=True, timeout=timeout
        )
        return completed.returncode == 0, completed.stdout or "", (completed.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return False, "", f"timed out after {timeout:.0f}s"
    except Exception as exc:
        return False, "", str(exc)


def run_cmd_checked(args: List[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            args, check=False, text=True, capture_output=True, timeout=CMD_TIMEOUT
        )
        ok = completed.returncode == 0
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        lines = [f"exit_code={completed.returncode}"]
        if stdout:
            lines.append(f"stdout: {stdout}")
        if stderr:
            lines.append(f"stderr: {stderr}")
        if not stdout and not stderr:
            lines.append("no output")
        return ok, " | ".join(lines)
    except Exception as exc:
        return False, str(exc)


def _config_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".config", "slurm-monitor-top")


def _config_path() -> str:
    return os.path.join(_config_dir(), "config.json")


def load_config() -> Dict[str, object]:
    """Load persisted settings from ~/.config/slurm-monitor-top/config.json."""
    try:
        with open(_config_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(config: Dict[str, object]) -> None:
    """Persist settings to ~/.config/slurm-monitor-top/config.json (best effort)."""
    try:
        os.makedirs(_config_dir(), exist_ok=True)
        with open(_config_path(), "w") as fh:
            json.dump(config, fh, indent=2)
    except Exception:
        pass


# Use --Format=tres-alloc: %b in -o/--format is a vestigial mapping to tres-per-node
# (not allocated GRES), so GPU type / usage counts would stay empty on modern Slurm.
_SQUEUE_FORMAT = (
    "jobid:|,username:|,state:|,partition:|,name:|,numnodes:|,"
    "numcpus:|,minmemory:|,tres-alloc:|,timeused:|,nodelist:"
)


def _job_from_line(line: str) -> Optional[Job]:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 11:
        return None
    return Job(
        job_id=parts[0],
        user=parts[1],
        state=parts[2],
        partition=parts[3],
        name=parts[4],
        nodes=parts[5],
        ncpus=parts[6],
        mem=parts[7],
        gpus=parts[8],
        time_used=parts[9],
        node_list=parts[10],
    )


def parse_squeue() -> List[Job]:
    raw = run_cmd_argv(["squeue", "-a", "-h", f"--Format={_SQUEUE_FORMAT}"])
    jobs: List[Job] = []
    for line in raw.strip().splitlines():
        job = _job_from_line(line)
        if job is not None:
            jobs.append(job)
    return jobs


def fetch_job(job_id: str) -> Optional[Job]:
    """Re-query squeue for a single job; returns None if it is no longer queued."""
    raw = run_cmd_argv(["squeue", "-a", "-h", "-j", job_id, f"--Format={_SQUEUE_FORMAT}"])
    found = [job for job in (_job_from_line(l) for l in raw.strip().splitlines()) if job is not None]
    for job in found:
        if job.job_id == job_id:
            return job
    # An array task asked for as `12345_7` comes back under its own JobId
    # (`12350`), so the ids will not match even though squeue answered about
    # exactly the job we named.
    return found[0] if len(found) == 1 else None


def _short_time(value: Optional[str]) -> str:
    """Trim a Slurm timestamp (2026-05-18T14:15:43) to a compact form."""
    v = (value or "").strip()
    if not v or v in {"Unknown", "N/A", "(null)", "None"}:
        return "-"
    v = v.replace("T", " ")
    return re.sub(r"(\d\d:\d\d):\d\d$", r"\1", v)


_NO_VALUE = {"", "unlimited", "partition_limit", "invalid", "n/a", "unknown", "none", "(null)"}

# AveCPU on a step Slurm never accounted for comes back as an overflowed
# counter (213503982334-14:25:51). Anything past a century is that, not work.
_ABSURD_SECONDS = 100 * 365 * 24 * 3600.0


def parse_duration(value: str) -> float:
    """Seconds from a Slurm duration, or -1.0 when there is no number in it.

    Covers every shape Slurm prints: `D-HH:MM:SS`, `D-HH:MM`, `HH:MM:SS`,
    `MM:SS`, `MM:SS.mmm` and a bare count of seconds. `UNLIMITED` and friends
    are "no limit", which is not a number either, so they get -1.0 too.
    """
    text = (value or "").strip()
    if text.lower() in _NO_VALUE:
        return -1.0
    days = 0.0
    if "-" in text:
        head, _, text = text.partition("-")
        try:
            days = float(head)
        except ValueError:
            return -1.0
    parts = text.split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return -1.0
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        # With a day part in front, `1-02:03` is hours and minutes; without
        # one, `02:03` is squeue's minutes and seconds.
        hours, minutes, seconds = (numbers[0], numbers[1], 0.0) if days else (0.0, numbers[0], numbers[1])
    elif len(numbers) == 1:
        hours, minutes, seconds = (numbers[0], 0.0, 0.0) if days else (0.0, 0.0, numbers[0])
    else:
        return -1.0
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return -1.0 if total >= _ABSURD_SECONDS else total


def format_duration(seconds: float) -> str:
    """Seconds back to Slurm's own `D-HH:MM:SS` / `HH:MM:SS` notation."""
    if seconds < 0:
        return "-"
    total = int(round(seconds))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _human_mem(value: str) -> str:
    mb = _parse_mem_to_mb(value)
    return _format_mb_human(mb) if mb > 0 else (value or "-")


def _tres_value(tres: str, key: str) -> str:
    for part in (tres or "").split(","):
        name, sep, val = part.strip().partition("=")
        if sep and name.strip() == key:
            return val.strip()
    return ""


def _parse_scontrol_kv(text: str) -> Dict[str, str]:
    """Parse `scontrol show job` output into a key/value dict.

    Splits only at whitespace that precedes a `Key=` token so values
    containing spaces (Command, SubmitLine, ...) stay intact.
    """
    result: Dict[str, str] = {}
    for token in re.split(r"\s+(?=[A-Za-z][\w/:.]*=)", text.strip()):
        key, sep, value = token.partition("=")
        if sep:
            result.setdefault(key.strip(), value.strip())
    return result


def fetch_job_detail(job_id: str) -> Dict[str, str]:
    """Detailed key/value fields for a job from `scontrol show job -d`."""
    raw = run_cmd_argv(["scontrol", "show", "job", "-d", job_id])
    if not raw.strip() or "Invalid job id" in raw:
        return {}
    return _parse_scontrol_kv(raw)


# sstat reports one row per step. The `extern` step is the container Slurm
# wraps around the job; it does no work and its AveCPU is a well-known garbage
# value (213503982334-14:25:51), so it never contributes to a usage figure.
_EXTERN_STEP = ".extern"


def fetch_job_usage(job_id: str, numeric_id: str = "") -> Dict[str, str]:
    """Live resource usage of a running job from `sstat` (best effort).

    Returns the step with the largest MaxRSS, plus the CPU time summed over
    every real step. Empty when sstat has no data (pending job, not owned by
    the user, no running steps yet).

    ``numeric_id`` is the plain JobId `scontrol` reports. It matters for array
    tasks: `sstat -j 12345_7` does not error, it silently ignores the filter
    and answers for *every* running step on the cluster, so without the real id
    - and the guard below that drops rows belonging to other jobs - an array
    task would show a stranger's memory.
    """
    wanted = {i for i in (job_id, numeric_id) if i}
    raw = run_cmd_argv([
        "sstat", "-a", "-P", "-n",
        "--format=JobID,MaxRSS,MaxVMSize,AveCPU,NTasks,TRESUsageInTot",
        "-j", numeric_id or job_id,
    ])
    best: Dict[str, str] = {}
    best_rss = -1.0
    cpu_seconds = 0.0
    steps = 0
    for line in raw.strip().splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 5 or not cols[0]:
            continue
        step_id = cols[0]
        if step_id.split(".")[0] not in wanted:
            continue
        if step_id.endswith(_EXTERN_STEP):
            continue
        steps += 1
        rss = _parse_mem_to_mb(cols[1])
        if rss > best_rss:
            best_rss = rss
            best = {
                "MaxRSS": cols[1], "MaxVMSize": cols[2],
                "AveCPU": cols[3], "NTasks": cols[4],
            }
        cpu_seconds += _step_cpu_seconds(cols[3], cols[4], cols[5] if len(cols) > 5 else "")
    if best and cpu_seconds > 0:
        # Named TotalCPU because that is what sacct calls the same quantity.
        best["TotalCPU"] = format_duration(cpu_seconds)
        best["Steps"] = str(steps)
    return best


def _step_cpu_seconds(ave_cpu: str, ntasks: str, tres_usage: str) -> float:
    """CPU time one step has burned, in seconds.

    Prefers TRESUsageInTot's `cpu=`, which is already the total over the
    step's tasks; AveCPU x NTasks is the fallback for a Slurm that does not
    report the TRES breakdown.
    """
    total = parse_duration(_tres_value(tres_usage, "cpu"))
    if total >= 0:
        return total
    ave = parse_duration(ave_cpu)
    if ave < 0:
        return 0.0
    return ave * max(1, _parse_int(ntasks))


def collect_job_info(job_id: str) -> tuple[Optional[Job], Dict[str, str], Dict[str, str]]:
    """Gather squeue summary + scontrol detail + sstat usage for one job."""
    job = fetch_job(job_id)
    if job is None:
        return None, {}, {}
    detail = fetch_job_detail(job_id)
    usage = (
        # scontrol resolves an array task to its own JobId; sstat needs that one.
        fetch_job_usage(job_id, detail.get("JobId", ""))
        if job.state.upper().startswith("R")
        else {}
    )
    return job, detail, usage


# --------------------------------------------------------------------------
# What a job asked for against what it is using
#
# Slurm hands out whole CPUs and whole gigabytes for the job's lifetime, so a
# job that reserved 16 cores and runs one thread wastes fifteen of them for
# days. Nothing in squeue or scontrol says so: the request is there, the usage
# is in sstat, and the comparison is what the metrics below are.
# --------------------------------------------------------------------------

# Which end of the scale is the bad one, for whoever draws the bar:
# a job near its time or memory limit is in danger of being killed, while a
# job using a fraction of its cores is wasting the other machines' turn.
RISK_HIGH = "high"
RISK_LOW = "low"


def _parse_slurm_time(value: str) -> float:
    """A Slurm timestamp (2026-05-18T14:15:43) as epoch seconds; 0 if unusable."""
    text = (value or "").strip()
    if not text or text.lower() in _NO_VALUE:
        return 0.0
    try:
        return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return 0.0


def job_memory_request_mb(detail: Dict[str, str], job: Optional[Job], nodes: int, cpus: int) -> "tuple[int, str]":
    """Memory the job holds on one node, in MB, and how that was worked out.

    Per node is the figure worth comparing against, because MaxRSS is a
    per-task peak, not a sum across the allocation.
    """
    per_node = _parse_mem_to_mb(detail.get("MinMemoryNode", ""))
    if per_node > 0:
        return per_node, "per node"
    per_cpu = _parse_mem_to_mb(detail.get("MinMemoryCPU", ""))
    if per_cpu > 0:
        cpus_per_node = max(1, cpus // max(1, nodes))
        return per_cpu * cpus_per_node, f"per node ({_format_mb_human(per_cpu)}/CPU x {cpus_per_node})"
    for key in ("AllocTRES", "ReqTRES", "TRES"):
        total = _parse_mem_to_mb(_tres_value(detail.get(key, ""), "mem"))
        if total > 0:
            return int(total / max(1, nodes)), "per node" if nodes > 1 else ""
    fallback = _parse_mem_to_mb(job.mem if job else "")
    return fallback, ""


def job_usage_metrics(
    job: Optional[Job], detail: Dict[str, str], usage: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Request-versus-use figures for one job, ready to draw as bars.

    Each entry is either a ``bar`` (it has a percentage), a ``fact`` (a number
    worth showing that nothing bounds) or a ``note`` (why a bar is missing).
    The caller decides what a bar looks like; this decides what is true.
    """
    metrics: List[Dict[str, Any]] = []
    state = (detail.get("JobState", "") or (job.state if job else "")).upper()
    running = state.startswith("R")

    cpus = _parse_int(detail.get("NumCPUs", "")) or _parse_int(job.ncpus if job else "")
    nodes = max(1, _parse_int(detail.get("NumNodes", "")) or _parse_int(job.nodes if job else ""))
    elapsed = parse_duration(detail.get("RunTime", "") or (job.time_used if job else ""))
    limit = parse_duration(detail.get("TimeLimit", ""))

    if not running:
        submitted = _parse_slurm_time(detail.get("SubmitTime", ""))
        if submitted:
            metrics.append({
                "key": "queued", "label": "Queued", "kind": "fact", "percent": None,
                "value": format_duration(max(0.0, time.time() - submitted)),
                "total": "", "risk": "",
                "note": f"waiting for {detail.get('Reason', '') or 'the scheduler'}",
            })
        if limit > 0:
            metrics.append({
                "key": "limit", "label": "Limit", "kind": "fact", "percent": None,
                "value": format_duration(limit), "total": "", "risk": "",
                "note": f"{cpus} CPU(s) on {nodes} node(s) once it starts",
            })
        metrics.append({
            "key": "note", "kind": "note", "label": "", "percent": None,
            "value": "", "total": "", "risk": "",
            "note": "Nothing is running yet, so there is nothing to measure.",
        })
        return metrics

    if elapsed >= 0 and limit > 0:
        metrics.append({
            "key": "time", "label": "Time", "kind": "bar",
            "percent": 100.0 * elapsed / limit, "risk": RISK_HIGH,
            "value": format_duration(elapsed), "total": format_duration(limit),
            "note": f"{format_duration(max(0.0, limit - elapsed))} left",
        })
    elif elapsed >= 0:
        metrics.append({
            "key": "time", "label": "Time", "kind": "fact", "percent": None,
            "value": format_duration(elapsed), "total": "", "risk": "",
            "note": "no time limit on this job",
        })

    cpu_used = parse_duration(usage.get("TotalCPU", ""))
    if cpu_used < 0:
        cpu_used = _step_cpu_seconds(usage.get("AveCPU", ""), usage.get("NTasks", ""), "")
        cpu_used = cpu_used if cpu_used > 0 else -1.0
    cpu_held = elapsed * cpus if elapsed > 0 and cpus > 0 else 0.0
    if cpu_used >= 0 and cpu_held > 0:
        busy = cpu_used / elapsed
        metrics.append({
            "key": "cpu", "label": "CPU", "kind": "bar",
            "percent": 100.0 * cpu_used / cpu_held, "risk": RISK_LOW,
            "value": format_duration(cpu_used), "total": format_duration(cpu_held),
            "note": f"{busy:.1f} of {cpus} cores busy (avg)",
        })

    mem_used = _parse_mem_to_mb(usage.get("MaxRSS", ""))
    mem_held, mem_basis = job_memory_request_mb(detail, job, nodes, cpus)
    if mem_used > 0 and mem_held > 0:
        metrics.append({
            "key": "memory", "label": "Memory", "kind": "bar",
            "percent": 100.0 * mem_used / mem_held, "risk": RISK_HIGH,
            "value": _format_mb_human(mem_used),
            # The basis rides along with the figure it qualifies: "of 64.0G per
            # node" says more in the same space than a note repeating it.
            "total": (_format_mb_human(mem_held) + " " + mem_basis).strip(),
            "note": "peak RSS",
        })
    elif mem_held > 0:
        metrics.append({
            "key": "memory", "label": "Memory", "kind": "fact", "percent": None,
            "value": (_format_mb_human(mem_held) + " " + mem_basis).strip(),
            "total": "", "risk": "",
            "note": "reserved; no live reading",
        })

    gpus = _parse_gpu_count(detail.get("TresPerNode", "") or detail.get("AllocTRES", "") or (job.gpus if job else ""))
    if gpus:
        metrics.append({
            "key": "gpu", "label": "GPUs", "kind": "fact", "percent": None,
            "value": str(gpus), "total": "", "risk": "",
            # Utilisation would have to come from the GPUs themselves; Slurm
            # accounts for the reservation and nothing else.
            "note": "held for the whole run - Slurm never measures their use",
        })

    if not usage:
        metrics.append({
            "key": "note", "kind": "note", "label": "", "percent": None,
            "value": "", "total": "", "risk": "",
            "note": "No live CPU or memory reading: sstat answers only for your own running jobs.",
        })
    return metrics


# --------------------------------------------------------------------------
# Job output
#
# Slurm never shows a job's stdout/stderr: it only records where the file was
# opened. `scontrol` gives us that path, and the rest is ordinary file reading
# from the login node - which is also the catch, since a job writing to local
# scratch on a compute node leaves a path that exists nowhere we can see.
# --------------------------------------------------------------------------

OUTPUT_TAIL_LINES = 200
# How far back we read to find those lines. A job that prints a progress bar
# without newlines can produce a multi-gigabyte file, and a tail must never
# pull one of those through NFS.
OUTPUT_TAIL_BYTES = 256 * 1024

_DISCARDED_PATHS = {"/dev/null"}


def expand_job_path(pattern: str, detail: Dict[str, str], job: Optional[Job] = None) -> str:
    """Fill in the `%j`/`%x`/`%A_%a` placeholders Slurm allows in -o/-e paths.

    `scontrol` has normally substituted them already; this is for the versions
    and the corner cases where it hands back the raw pattern, so the path we
    show is one you can actually open.
    """
    text = (pattern or "").strip()
    if not text:
        return ""
    array_job = detail.get("ArrayJobId", "") or detail.get("JobId", "")
    values = {
        "A": array_job,
        "a": detail.get("ArrayTaskId", ""),
        "j": detail.get("JobId", "") or (job.job_id if job else ""),
        "J": detail.get("JobId", "") or (job.job_id if job else ""),
        "x": detail.get("JobName", "") or (job.name if job else ""),
        "u": (detail.get("UserId", "") or (job.user if job else "")).split("(")[0],
        "N": (detail.get("BatchHost", "") or (job.node_list if job else "")).split(",")[0],
        "n": "0",
        "t": "0",
        "%": "%",
    }
    out = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "%" or index + 1 >= len(text):
            out.append(char)
            index += 1
            continue
        # A width may sit between the % and the letter: %4j pads the job id.
        width = ""
        cursor = index + 1
        while cursor < len(text) and text[cursor].isdigit():
            width += text[cursor]
            cursor += 1
        key = text[cursor] if cursor < len(text) else ""
        if key not in values:
            out.append(char)
            index += 1
            continue
        value = values[key]
        if width and value.isdigit():
            value = value.zfill(int(width))
        out.append(value)
        index = cursor + 1
    resolved = "".join(out)
    work_dir = detail.get("WorkDir", "")
    if resolved and not os.path.isabs(resolved) and work_dir:
        resolved = os.path.join(work_dir, resolved)
    return resolved


def job_output_paths(detail: Dict[str, str], job: Optional[Job] = None) -> Dict[str, Dict[str, object]]:
    """Where this job's stdout and stderr go, as `{"stdout": {...}, "stderr": {...}}`.

    An empty StdErr means the batch script asked for one file, not two, so the
    stderr entry points at the stdout path and says it is merged - otherwise
    the UI would offer an empty stream that never exists.
    """
    stdout_path = expand_job_path(detail.get("StdOut", ""), detail, job)
    stderr_path = expand_job_path(detail.get("StdErr", ""), detail, job)
    merged = bool(stdout_path) and (not stderr_path or stderr_path == stdout_path)
    streams = {
        "stdout": dict(stat_output_file(stdout_path), stream="stdout", merged=False),
        "stderr": dict(
            stat_output_file(stdout_path if merged else stderr_path),
            stream="stderr",
            merged=merged,
        ),
    }
    # scontrol prints StdOut/StdErr only to the job's owner, so for anyone
    # else's job there is no path to be missing - say which of the two it is.
    if not stdout_path and not _job_is_mine(detail, job):
        for stream in streams.values():
            stream["error"] = "only the job's owner can see where its output goes"
    return streams


def _job_is_mine(detail: Dict[str, str], job: Optional[Job]) -> bool:
    me = os.environ.get("USER", "")
    owner = (detail.get("UserId", "") or (job.user if job else "")).split("(")[0]
    return not me or not owner or me == owner


def stat_output_file(path: str) -> Dict[str, object]:
    """Whether an output file is there and how big it is - no contents read."""
    info: Dict[str, object] = {
        "path": path, "exists": False, "size": 0, "modified": 0.0, "error": "",
    }
    if not path:
        info["error"] = "Slurm did not record a path for this stream"
        return info
    if path in _DISCARDED_PATHS:
        info["error"] = "discarded by the job (/dev/null)"
        return info
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        info["error"] = "not there yet - the job has not written to it"
        return info
    except PermissionError:
        info["error"] = "no permission to read it"
        return info
    except OSError as exc:
        info["error"] = str(exc)
        return info
    info["exists"] = True
    info["size"] = int(stat.st_size)
    info["modified"] = float(stat.st_mtime)
    return info


def read_file_tail(
    path: str, lines: int = OUTPUT_TAIL_LINES, max_bytes: int = OUTPUT_TAIL_BYTES
) -> Dict[str, object]:
    """The last `lines` lines of a file, plus what we know about the file.

    Reads only the final `max_bytes`, so following a job that has been printing
    for a week costs the same as following one that started a minute ago.
    """
    info = stat_output_file(path)
    info.update({"text": "", "line_count": 0, "truncated": False})
    if not info["exists"]:
        return info
    size = int(info["size"])
    start = max(0, size - max_bytes)
    try:
        with open(path, "rb") as handle:
            if start:
                handle.seek(start)
            raw = handle.read()
    except OSError as exc:
        info["error"] = str(exc)
        return info
    text = raw.decode("utf-8", errors="replace")
    if start:
        # The seek almost certainly landed mid-line; drop that fragment rather
        # than show half a line as if it were whole.
        text = text.split("\n", 1)[-1]
    found = text.splitlines()
    kept = found[-lines:] if lines > 0 else found
    info["text"] = "\n".join(kept)
    info["line_count"] = len(kept)
    info["truncated"] = bool(start) or len(kept) < len(found)
    return info


def job_output(
    job_id: str, stream: str = "stdout", lines: int = OUTPUT_TAIL_LINES
) -> Dict[str, object]:
    """Tail one of a job's two output streams, with the paths of both."""
    job = fetch_job(job_id)
    detail = fetch_job_detail(job_id)
    streams = job_output_paths(detail, job)
    key = "stderr" if stream in ("err", "stderr") else "stdout"
    chosen = streams[key]
    tail = read_file_tail(str(chosen.get("path", "")), lines=lines)
    tail.update({"stream": key, "merged": chosen.get("merged", False)})
    return {
        "job_id": job_id,
        "state": (detail.get("JobState", "") or (job.state if job else "")),
        "streams": streams,
        "output": tail,
    }


def fetch_node_detail(node_name: str) -> Dict[str, str]:
    """Detailed key/value fields for a node from `scontrol show node`."""
    raw = run_cmd_argv(["scontrol", "show", "node", node_name])
    if not raw.strip() or "not found" in raw.lower():
        return {}
    return _parse_scontrol_kv(raw)


def fetch_node_jobs(node_name: str) -> List[Job]:
    """Jobs Slurm currently places on one node (`squeue -w`)."""
    raw = run_cmd_argv(
        ["squeue", "-a", "-h", "-w", node_name, f"--Format={_SQUEUE_FORMAT}"]
    )
    jobs: List[Job] = []
    for line in raw.strip().splitlines():
        job = _job_from_line(line)
        if job is not None:
            jobs.append(job)
    return sort_jobs(jobs)


def collect_node_info(node_name: str) -> "tuple[Dict[str, str], List[Job]]":
    """scontrol detail + the jobs on that node, for the node details modal."""
    return fetch_node_detail(node_name), fetch_node_jobs(node_name)


# --------------------------------------------------------------------------
# Pinned jobs
#
# Kept in the shared config file rather than in each front end, so a job you
# pin in the terminal is already at the top of the list in the editor.
# --------------------------------------------------------------------------

# A pin outlives the job it was put on - a finished job is simply absent from
# squeue - so without an expiry the file would grow forever.
PIN_MAX_AGE = 30 * 24 * 3600


def load_pinned_jobs() -> List[str]:
    """Job ids the user pinned, most recently pinned last."""
    raw = load_config().get("pinned_jobs")
    now = time.time()
    fresh: List[tuple[float, str]] = []
    if isinstance(raw, dict):
        for job_id, stamp in raw.items():
            try:
                when = float(stamp)
            except Exception:
                continue
            if now - when <= PIN_MAX_AGE:
                fresh.append((when, str(job_id)))
    elif isinstance(raw, list):  # tolerate a hand-edited plain list
        fresh = [(now, str(job_id)) for job_id in raw]
    return [job_id for _, job_id in sorted(fresh)]


def set_job_pinned(job_id: str, pinned: bool) -> List[str]:
    """Pin or unpin one job; returns the new list."""
    config = load_config()
    raw = config.get("pinned_jobs")
    current: Dict[str, float] = {}
    if isinstance(raw, dict):
        for key, stamp in raw.items():
            try:
                current[str(key)] = float(stamp)
            except Exception:
                continue
    elif isinstance(raw, list):
        current = {str(key): time.time() for key in raw}

    now = time.time()
    current = {k: v for k, v in current.items() if now - v <= PIN_MAX_AGE}
    if pinned:
        current[str(job_id)] = now
    else:
        current.pop(str(job_id), None)

    config["pinned_jobs"] = current
    save_config(config)
    return [key for key, _ in sorted(current.items(), key=lambda kv: kv[1])]


def toggle_job_pin(job_id: str) -> "tuple[List[str], bool]":
    """Flip one job's pin; returns ``(pinned ids, is now pinned)``."""
    now_pinned = str(job_id) not in set(load_pinned_jobs())
    return set_job_pinned(job_id, now_pinned), now_pinned


def apply_pins(jobs: List[Job], pinned: "set[str] | List[str]") -> List[Job]:
    """Move pinned jobs to the front, keeping the order of both groups.

    Applied after sorting rather than folded into the sort key, so reversing
    the sort direction does not send the pinned rows to the bottom.
    """
    ids = set(pinned)
    if not ids:
        return list(jobs)
    front = [j for j in jobs if j.job_id in ids]
    rest = [j for j in jobs if j.job_id not in ids]
    return front + rest


# --------------------------------------------------------------------------
# CPU model and clock speed
#
# Slurm knows how many CPUs a node has and how they are arranged (sockets,
# cores, threads) but not *what* they are: neither `sinfo` nor `scontrol show
# node` carries a model name or a clock. The only place that information exists
# is the node itself, so it has to be read there once and remembered.
# --------------------------------------------------------------------------

# Hardware does not change under you, so a reading stays good for a long time;
# the expiry is only there so a re-imaged or replaced node eventually corrects
# itself.
CPU_INFO_MAX_AGE = 90 * 24 * 3600

_READ_CPU = "lscpu 2>/dev/null || cat /proc/cpuinfo"

# lscpu label -> our key. /proc/cpuinfo uses the same "label : value" shape for
# the two fields that matter there, so one parser covers the fallback too.
_LSCPU_KEYS = {
    "model name": "model",
    "architecture": "arch",
    "vendor id": "vendor",
    "cpu(s)": "cpus",
    "socket(s)": "sockets",
    "core(s) per socket": "cores_per_socket",
    "thread(s) per core": "threads_per_core",
    "cpu mhz": "mhz",
    "cpu max mhz": "max_mhz",
    "cpu min mhz": "min_mhz",
}


def _parse_sct(value: str) -> "tuple[str, str, str]":
    """Split sinfo's %z (``sockets:cores:threads``) into its three numbers."""
    parts = (value or "").strip().split(":")
    if len(parts) != 3:
        return "", "", ""
    return tuple(p.strip() for p in parts)  # type: ignore[return-value]


def _cpu_info_path() -> str:
    return os.path.join(_config_dir(), "cpu-info.json")


# Reloaded only when the file's mtime moves, so the JSON exporter can call this
# on every snapshot without a stat-plus-parse per node, and a probe made in the
# terminal UI still shows up in the editor a second later.
_CPU_INFO_MEMO: Dict[str, object] = {"mtime": None, "data": {}}


def load_cpu_info() -> Dict[str, Dict[str, str]]:
    """Per-node CPU readings collected by earlier probes."""
    path = _cpu_info_path()
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        _CPU_INFO_MEMO["mtime"] = None
        _CPU_INFO_MEMO["data"] = {}
        return {}
    if _CPU_INFO_MEMO["mtime"] == mtime:
        return _CPU_INFO_MEMO["data"]  # type: ignore[return-value]
    data: Dict[str, Dict[str, str]] = {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
        nodes = raw.get("nodes") if isinstance(raw, dict) else None
        if isinstance(nodes, dict):
            now = time.time()
            for name, info in nodes.items():
                if not isinstance(info, dict):
                    continue
                age = now - float(info.get("probed_at") or 0)
                if age <= CPU_INFO_MAX_AGE:
                    data[str(name)] = {k: str(v) for k, v in info.items()}
    except Exception:
        data = {}
    _CPU_INFO_MEMO["mtime"] = mtime
    _CPU_INFO_MEMO["data"] = data
    return data


def save_cpu_info(node_name: str, info: Dict[str, str]) -> None:
    """Remember one node's reading (best effort)."""
    cache = dict(load_cpu_info())
    cache[node_name] = info
    try:
        os.makedirs(_config_dir(), exist_ok=True)
        with open(_cpu_info_path(), "w") as fh:
            json.dump({"version": 1, "nodes": cache}, fh, indent=2)
    except Exception:
        return
    _CPU_INFO_MEMO["mtime"] = None  # force a reload on the next read


def _parse_lscpu(text: str) -> Dict[str, str]:
    """Pull the fields we show out of `lscpu` or /proc/cpuinfo output."""
    info: Dict[str, str] = {}
    for line in (text or "").splitlines():
        label, sep, value = line.partition(":")
        if not sep:
            continue
        key = _LSCPU_KEYS.get(label.strip().lower())
        value = value.strip()
        # /proc/cpuinfo repeats a block per logical CPU; the first wins.
        if key and value and key not in info:
            info[key] = value
    return info


def _round_mhz(value: str) -> str:
    try:
        return f"{float(value):.0f}"
    except Exception:
        return ""


def cpu_speed(info: Dict[str, str]) -> "tuple[str, str]":
    """``(speed, kind)`` for a reading, e.g. ``("2.20GHz", "nominal")``.

    Prefers the nominal clock printed in the model name. The *current* MHz is
    whatever the governor happened to be doing during the probe - often an idle
    800MHz on a machine that runs at 3GHz - and the max is a boost ceiling one
    core reaches, not a speed the node sustains. Which one we ended up with is
    returned alongside so a caller can label it rather than pass a boost figure
    off as the real speed.
    """
    model = (info or {}).get("model", "")
    match = re.search(r"@\s*([\d.]+)\s*GHz", model, re.IGNORECASE)
    if match:
        return f"{match.group(1)}GHz", "nominal"
    for key, kind in (("max_mhz", "max"), ("mhz", "current")):
        mhz = _round_mhz((info or {}).get(key, ""))
        if mhz:
            return f"{float(mhz) / 1000:.2f}GHz", kind
    return "", ""


def cpu_speed_ghz(info: Dict[str, str]) -> str:
    """Just the number from :func:`cpu_speed`."""
    return cpu_speed(info)[0]


def short_cpu_model(info: Dict[str, str]) -> str:
    """``Intel(R) Xeon(R) Bronze 3104 CPU @ 1.70GHz`` -> ``Xeon Bronze 3104``.

    The trademark noise and the repeated clock cost a third of the column
    width and say nothing; the speed is shown beside this, not inside it.
    """
    model = (info or {}).get("model", "").strip()
    if not model:
        return ""
    model = re.sub(r"\(R\)|\(TM\)|\(tm\)", "", model)
    model = re.sub(r"\s*@.*$", "", model)
    model = re.sub(r"\b(CPU|Processor|Genuine|Intel|AMD)\b", "", model)
    model = re.sub(r"\b\d+-Core\b", "", model)
    return re.sub(r"\s+", " ", model).strip(" -")


def describe_cpu(info: Dict[str, str]) -> str:
    """One line for a table cell: ``Xeon Bronze 3104 @ 1.70GHz``.

    A clock we had to take from the boost ceiling is written ``max 3.35GHz``,
    so it is never mistaken for the speed the node actually runs at.
    """
    model = short_cpu_model(info)
    speed, kind = cpu_speed(info or {})
    if speed and kind != "nominal":
        speed = f"{kind} {speed}"
    if model and speed:
        separator = " @ " if kind == "nominal" else ", "
        return f"{model}{separator}{speed}"
    return model or speed or ""


def cpu_topology(node: Node) -> str:
    """``2 x 64C/2T`` - sockets by cores per socket by threads per core."""
    if not node.sockets or not node.cores_per_socket:
        return ""
    threads = node.threads_per_core or "1"
    return f"{node.sockets} x {node.cores_per_socket}C/{threads}T"


def cpu_counts(
    sockets: str = "",
    cores_per_socket: str = "",
    threads_per_core: str = "",
    cpus_total: str = "",
) -> Dict[str, int]:
    """How many of each thing a node's processors come to.

    One socket holds one physical processor, so ``processors`` is the number of
    chips you would find in the machine; ``cores`` is what they add up to, and
    ``logical`` is what Slurm hands out and what ``nproc`` reports. Slurm's own
    CPU total wins when we have it, because a node with cores reserved for the
    OS reports fewer than sockets x cores x threads.
    """
    chips = _parse_int(sockets)
    per_chip = _parse_int(cores_per_socket)
    threads = _parse_int(threads_per_core) or 1
    cores = chips * per_chip
    return {
        "processors": chips,
        "cores_per_processor": per_chip,
        "cores": cores,
        "threads_per_core": threads,
        "logical": _parse_int(cpus_total) or cores * threads,
    }


def describe_cpu_count(counts: Dict[str, int], info: Dict[str, str]) -> str:
    """``2 x EPYC 7763`` - how many processors, and of what.

    Empty until the node has been probed: the count alone (``2 x``) says
    nothing you cannot already read off the socket number.
    """
    model = short_cpu_model(info or {})
    if not model:
        return ""
    chips = (counts or {}).get("processors", 0)
    return f"{chips} x {model}" if chips else model


def describe_cpu_speeds(info: Dict[str, str]) -> "List[tuple[str, str]]":
    """``[("nominal", "2.45GHz"), ("max", "3.53GHz"), ("now", "2.10GHz")]``.

    Every clock we have, each labelled with what it actually is, rather than one
    number that could be any of the three. ``nominal`` is the speed printed in
    the model name - what the machine runs at - while ``max`` is a boost ceiling
    and ``now`` is whatever the governor was doing when we looked.
    """
    speeds: List[tuple[str, str]] = []
    nominal, kind = cpu_speed(info or {})
    if nominal and kind == "nominal":
        speeds.append(("nominal", nominal))
    for key, label in (("max_mhz", "max"), ("mhz", "now")):
        mhz = _round_mhz((info or {}).get(key, ""))
        if mhz:
            speeds.append((label, f"{float(mhz) / 1000:.2f}GHz"))
    return speeds


def probe_node_cpu(node_name: str, allow_srun: bool = True) -> "tuple[Dict[str, str], str]":
    """Read a node's CPU model and clock, and remember the answer.

    Returns ``(info, error)``; ``info`` is empty when every route failed.

    Two routes, cheapest first:

    * **ssh** - instant, but clusters running ``pam_slurm_adopt`` only let you
      onto a node where you already have a job, so it works for your own
      machines and nowhere else.
    * **srun** - a one-second job that prints ``lscpu``. It works anywhere you
      can submit, but it needs a free slice on that node, so it is refused on a
      full one and is never run unasked.
    """
    errors: List[str] = []

    def last_line(text: str) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        # ssh prints host-key notices before the real failure, and srun repeats
        # its own prefix; the last line is the one that says what went wrong.
        return lines[-1].replace("srun: error: ", "") if lines else ""

    ok, out, err = run_cmd_output(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5", node_name, _READ_CPU,
        ],
        timeout=20,
    )
    info = _parse_lscpu(out) if ok else {}
    source = "ssh"
    if not info.get("model"):
        errors.append(f"ssh: {last_line(err) or 'no CPU information'}")
        if not allow_srun:
            return {}, "; ".join(errors)
        ok, out, err = run_cmd_output(
            [
                "srun", "--nodelist", node_name, "--ntasks=1", "--cpus-per-task=1",
                "--mem=16M", "--time=1", "--oversubscribe", "--immediate=20",
                "--job-name=slurm-top-cpu", "-Q", "sh", "-c", _READ_CPU,
            ],
            timeout=45,
        )
        info = _parse_lscpu(out) if ok else {}
        source = "srun"
        if not info.get("model"):
            errors.append(f"srun: {last_line(err) or 'no CPU information'}")
            return {}, "; ".join(errors)

    info["source"] = source
    info["probed_at"] = str(int(time.time()))
    save_cpu_info(node_name, info)
    return info, ""


def parse_gres_used() -> Dict[str, str]:
    """Allocated GRES per node, from `sinfo -O GresUsed`.

    `sinfo -o %G` reports what a node *has*, never what is handed out, so
    without this second call a node with four idle GPUs and one with four busy
    ones look identical. Kept separate because %G-style format strings have no
    equivalent field; a failure here just leaves the column empty.
    """
    raw = run_cmd_argv(["sinfo", "-h", "-N", "-O", "NodeHost:64,GresUsed:128"])
    used: Dict[str, str] = {}
    for line in raw.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        used[parts[0].strip()] = parts[1].strip()
    return used


def parse_sinfo() -> List[Node]:
    format_str = "%n|%t|%c|%C|%m|%e|%G|%O|%P|%z|%E"
    raw = run_cmd(f"sinfo -o '{format_str}'")
    gres_used = parse_gres_used()
    lines = raw.strip().splitlines()
    nodes: List[Node] = []
    for line in lines[1:]:
        # Bounded split: a drain reason is free text and may itself contain a
        # pipe, so everything after the tenth separator belongs to it.
        parts = line.split("|", 10)
        if len(parts) != 11:
            continue
        name = parts[0].strip()
        state = parts[1].strip()
        cpus_total = parts[2].strip()
        c_state = parts[3].strip()
        mem_total = parts[4].strip()
        mem_free = parts[5].strip()
        gres = parts[6].strip()
        cpu_load = parts[7].strip()
        partition = parts[8].strip()
        sockets, cores_per_socket, threads_per_core = _parse_sct(parts[9])
        reason = parts[10].strip()

        cpus_alloc = ""
        cpus_idle = ""
        try:
            alloc, idle, *_ = c_state.split("/")
            cpus_alloc = alloc
            cpus_idle = idle
        except Exception:
            pass

        mem_reserved = ""
        total_mb = _parse_int(mem_total)
        free_mb = _parse_int(mem_free)
        if total_mb > 0 and free_mb >= 0:
            mem_reserved = str(max(0, total_mb - free_mb))

        nodes.append(
            Node(
                name=name,
                state=state,
                cpus_total=cpus_total,
                cpus_alloc=cpus_alloc,
                cpus_idle=cpus_idle,
                mem_total=mem_total,
                mem_reserved=mem_reserved,
                mem_free=mem_free,
                gres=gres,
                partition=partition,
                cpu_load=cpu_load,
                reason="" if reason.lower() in {"none", "(null)", "n/a"} else reason,
                gres_used=gres_used.get(name, ""),
                sockets=sockets,
                cores_per_socket=cores_per_socket,
                threads_per_core=threads_per_core,
            )
        )
    return nodes


def parse_disks() -> List[DiskUsage]:
    # `target` goes last on purpose: a mount point containing a space would
    # otherwise shift every column after it, and everything before it is a
    # single token, so the remainder of the line is the mount path.
    raw = run_cmd("df -h --output=pcent,fstype,size,used,avail,target")
    lines = raw.strip().splitlines()
    disks: List[DiskUsage] = []
    for line in lines[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        disks.append(
            DiskUsage(
                usage_percent=parts[0].strip(),
                fs_type=parts[1].strip(),
                size=parts[2].strip(),
                used=parts[3].strip(),
                avail=parts[4].strip(),
                mount=parts[5].strip(),
            )
        )
    return disks


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _parse_mem_to_mb(value: str) -> int:
    v = value.strip().upper()
    if not v:
        return 0
    num = ""
    unit = ""
    for ch in v:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            unit += ch
    if not num:
        return 0
    try:
        base = float(num)
    except Exception:
        return 0
    unit = unit or "M"
    if unit.startswith("G"):
        return int(base * 1024)
    if unit.startswith("T"):
        return int(base * 1024 * 1024)
    if unit.startswith("K"):
        return int(base / 1024)
    return int(base)


def _parse_gpu_count(value: str) -> int:
    text = (value or "").strip().lower()
    if not text or text in {"(null)", "n/a"}:
        return 0
    total = 0
    for match in re.finditer(r"gpu(?::[^:,=]+)?[:=](\d+)", text):
        try:
            total += int(match.group(1))
        except Exception:
            continue
    return total


def _parse_gpu_per_type(value: str) -> Dict[str, int]:
    text = (value or "").strip().lower()
    if not text or text in {"(null)", "n/a"}:
        return {}
    per_type: Dict[str, int] = {}
    for match in re.finditer(r"gpu(?::([^:,=]+))?[:=](\d+)", text):
        gpu_type = (match.group(1) or "generic").strip() or "generic"
        try:
            count = int(match.group(2))
        except Exception:
            continue
        per_type[gpu_type] = per_type.get(gpu_type, 0) + count
    return per_type


def _parse_gpu_inventory(gres: str) -> Dict[str, int]:
    text = (gres or "").strip().lower()
    if not text or text in {"(null)", "n/a"}:
        return {}
    return _parse_gpu_per_type(text)


def _format_mb_human(mb: int) -> str:
    if mb < 1024:
        return f"{mb}M"
    gb = mb / 1024
    if gb < 1024:
        return f"{gb:.1f}G"
    tb = gb / 1024
    return f"{tb:.2f}T"


def _job_id_sort_key(job_id: str) -> int:
    digits = "".join(ch for ch in job_id if ch.isdigit())
    if not digits:
        return 10**12
    try:
        return int(digits)
    except Exception:
        return 10**12


def _job_state_rank(state: str) -> int:
    st = state.upper()
    if st.startswith("R"):
        return 0
    if st.startswith("CG"):
        return 1
    if st.startswith("P"):
        return 2
    return 3


def sort_jobs(jobs: List[Job]) -> List[Job]:
    return sorted(jobs, key=lambda j: (_job_state_rank(j.state), _job_id_sort_key(j.job_id)))


def summarize_jobs(jobs: List[Job], current_user: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    summary: Dict[str, Dict[str, Dict[str, int]]] = {
        "all": {"running": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}, "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}},
        "me": {"running": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}, "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}},
        "others": {"running": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}, "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}},
    }
    for j in jobs:
        owner = "me" if j.user == current_user else "others"
        st = j.state.upper()
        if st.startswith("R"):
            key = "running"
        elif st.startswith("P"):
            key = "pending"
        else:
            continue
        cpus = _parse_int(j.ncpus)
        mem_mb = _parse_mem_to_mb(j.mem)
        gpus = _parse_gpu_count(j.gpus)
        for bucket in ("all", owner):
            summary[bucket][key]["jobs"] += 1
            summary[bucket][key]["cpus"] += cpus
            summary[bucket][key]["mem_mb"] += mem_mb
            summary[bucket][key]["gpus"] += gpus
    return summary


def summarize_gpus(nodes: List[Node], jobs: List[Job]) -> Dict[str, object]:
    per_type: Dict[str, int] = {}
    total = 0
    for n in nodes:
        inv = _parse_gpu_inventory(n.gres)
        for gpu_type, count in inv.items():
            per_type[gpu_type] = per_type.get(gpu_type, 0) + count
            total += count

    active = 0
    reserved = 0
    per_type_stats: Dict[str, Dict[str, int]] = {
        gpu_type: {"total": total_count, "active": 0, "reserved": 0, "free_est": total_count}
        for gpu_type, total_count in per_type.items()
    }
    for j in jobs:
        gpus = _parse_gpu_count(j.gpus)
        per_job_types = _parse_gpu_per_type(j.gpus)
        st = j.state.upper()
        if st.startswith("R"):
            active += gpus
            for gpu_type, count in per_job_types.items():
                bucket = per_type_stats.setdefault(gpu_type, {"total": 0, "active": 0, "reserved": 0, "free_est": 0})
                bucket["active"] += count
        elif st.startswith("P"):
            reserved += gpus
            for gpu_type, count in per_job_types.items():
                bucket = per_type_stats.setdefault(gpu_type, {"total": 0, "active": 0, "reserved": 0, "free_est": 0})
                bucket["reserved"] += count
    for bucket in per_type_stats.values():
        bucket["free_est"] = max(0, bucket.get("total", 0) - bucket.get("active", 0))

    return {"total": total, "types_count": len(per_type), "per_type": per_type, "per_type_stats": per_type_stats, "active": active, "reserved": reserved, "free_est": max(0, total - active)}
