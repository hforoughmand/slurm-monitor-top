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
from dataclasses import dataclass
from typing import Dict, List, Optional


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


def run_cmd_argv(argv: List[str]) -> str:
    try:
        return subprocess.check_output(
            argv, stderr=subprocess.DEVNULL, text=True, timeout=CMD_TIMEOUT
        )
    except Exception:
        return ""


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
    for line in raw.strip().splitlines():
        job = _job_from_line(line)
        if job is not None and job.job_id == job_id:
            return job
    return None


def _short_time(value: Optional[str]) -> str:
    """Trim a Slurm timestamp (2026-05-18T14:15:43) to a compact form."""
    v = (value or "").strip()
    if not v or v in {"Unknown", "N/A", "(null)", "None"}:
        return "-"
    v = v.replace("T", " ")
    return re.sub(r"(\d\d:\d\d):\d\d$", r"\1", v)


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


def fetch_job_usage(job_id: str) -> Dict[str, str]:
    """Live resource usage of a running job from `sstat` (best effort).

    Returns the step with the largest MaxRSS; empty when sstat has no data
    (pending job, not owned by the user, no running steps yet).
    """
    raw = run_cmd_argv([
        "sstat", "-a", "-P", "-n",
        "--format=MaxRSS,MaxVMSize,AveCPU,NTasks",
        "-j", job_id,
    ])
    best: Dict[str, str] = {}
    best_rss = -1.0
    for line in raw.strip().splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 4 or not cols[0]:
            continue
        rss = _parse_mem_to_mb(cols[0])
        if rss > best_rss:
            best_rss = rss
            best = {"MaxRSS": cols[0], "MaxVMSize": cols[1], "AveCPU": cols[2], "NTasks": cols[3]}
    return best


def collect_job_info(job_id: str) -> tuple[Optional[Job], Dict[str, str], Dict[str, str]]:
    """Gather squeue summary + scontrol detail + sstat usage for one job."""
    job = fetch_job(job_id)
    if job is None:
        return None, {}, {}
    detail = fetch_job_detail(job_id)
    usage = fetch_job_usage(job_id) if job.state.upper().startswith("R") else {}
    return job, detail, usage


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
    format_str = "%n|%t|%c|%C|%m|%e|%G|%O|%P|%E"
    raw = run_cmd(f"sinfo -o '{format_str}'")
    gres_used = parse_gres_used()
    lines = raw.strip().splitlines()
    nodes: List[Node] = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) != 10:
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
        reason = parts[9].strip()

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
