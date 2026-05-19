import asyncio
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.events import Key
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, Static


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


@dataclass
class DiskUsage:
    usage_percent: str
    mount: str
    fs_type: str
    size: str


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


def parse_sinfo() -> List[Node]:
    format_str = "%n|%t|%c|%C|%m|%e|%G"
    raw = run_cmd(f"sinfo -o '{format_str}'")
    lines = raw.strip().splitlines()
    nodes: List[Node] = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) != 7:
            continue
        name = parts[0].strip()
        state = parts[1].strip()
        cpus_total = parts[2].strip()
        c_state = parts[3].strip()
        mem_total = parts[4].strip()
        mem_free = parts[5].strip()
        gres = parts[6].strip()

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
            )
        )
    return nodes


def parse_disks() -> List[DiskUsage]:
    raw = run_cmd("df -h --output=pcent,target,fstype,size")
    lines = raw.strip().splitlines()
    disks: List[DiskUsage] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        disks.append(
            DiskUsage(
                usage_percent=parts[0].strip(),
                mount=parts[1].strip(),
                fs_type=parts[2].strip(),
                size=parts[3].strip(),
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


class JobsView(DataTable[str]):
    BINDINGS = [
        ("s", "open_sort_menu", "Sort"),
        ("d", "toggle_sort_direction", "Asc/Desc"),
        ("f", "cycle_owner_filter", "Owner"),
        ("enter", "open_details", "Details"),
    ]
    jobs: reactive[List[Job]] = reactive([])  # type: ignore
    owner_filter: reactive[str] = reactive("all")  # type: ignore
    state_filter: reactive[str] = reactive("all")  # type: ignore
    sort_key: reactive[str] = reactive("state")  # type: ignore
    sort_desc: reactive[bool] = reactive(False)  # type: ignore
    user: str = os.environ.get("USER", "")
    _display_jobs: List[Job]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._display_jobs = []

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_columns("JOBID", "USER", "STATE", "PART", "NAME", "NODES", "CPUS", "GPUS", "MEM", "TIME")
        self.refresh_table()

    def _build_title(self) -> str:
        return "Slurm Jobs " f"(Enter: details, s: sort menu, d: {'desc' if self.sort_desc else 'asc'}, " f"f: owner={self.owner_filter}, state={self.state_filter})"

    def _update_title(self) -> None:
        selected = self.get_selected_job()
        hint = f" | selected {selected.job_id}: Enter opens details" if selected else ""
        self.title = self._build_title() + hint

    def _include_owner(self, job: Job) -> bool:
        if self.owner_filter == "all":
            return True
        if self.owner_filter == "me":
            return job.user == self.user
        return job.user != self.user

    def _include_state(self, job: Job) -> bool:
        st = job.state.upper()
        if self.state_filter == "all":
            return True
        if self.state_filter == "running":
            return st.startswith("R")
        if self.state_filter == "pending":
            return st.startswith("P")
        return not st.startswith("R") and not st.startswith("P")

    def _sort_value(self, job: Job):
        if self.sort_key == "jobid":
            return _job_id_sort_key(job.job_id)
        if self.sort_key == "user":
            return job.user.lower()
        if self.sort_key == "partition":
            return job.partition.lower()
        if self.sort_key == "cpus":
            return _parse_int(job.ncpus)
        if self.sort_key == "gpus":
            return _parse_gpu_count(job.gpus)
        if self.sort_key == "mem":
            return _parse_mem_to_mb(job.mem)
        if self.sort_key == "time":
            return job.time_used
        if self.sort_key == "state":
            return (_job_state_rank(job.state), _job_id_sort_key(job.job_id))
        return job.job_id

    def refresh_table(self) -> None:
        previous_scroll_x = self.scroll_x
        previous_scroll_y = self.scroll_y
        previous_row = self.cursor_row if self.cursor_row is not None else 0
        selected_job_id = None
        selected = self.get_selected_job()
        if selected:
            selected_job_id = selected.job_id

        self.clear(columns=False)
        self._display_jobs = [j for j in self.jobs if self._include_owner(j) and self._include_state(j)]
        self._display_jobs = sorted(self._display_jobs, key=self._sort_value, reverse=self.sort_desc)
        for j in self._display_jobs:
            style = None
            if j.state.upper().startswith("R"):
                style = "green"
            elif j.state.upper().startswith("P"):
                style = "yellow"
            elif j.state.upper().startswith("F"):
                style = "red"
            self.add_row(j.job_id, j.user, Text(j.state, style=style), j.partition, j.name, j.nodes, j.ncpus, str(_parse_gpu_count(j.gpus)), j.mem, j.time_used)

        self._update_title()
        if not self._display_jobs:
            return

        if selected_job_id:
            for row_idx, job in enumerate(self._display_jobs):
                if job.job_id == selected_job_id:
                    self.move_cursor(row=row_idx)
                    self.scroll_to(x=previous_scroll_x, y=previous_scroll_y, animate=False)
                    self._update_title()
                    return

        if previous_row is not None and previous_row >= 0:
            self.move_cursor(row=min(previous_row, len(self._display_jobs) - 1))
            self.scroll_to(x=previous_scroll_x, y=previous_scroll_y, animate=False)
            self._update_title()
            return
        self.move_cursor(row=0)
        self.scroll_to(x=previous_scroll_x, y=previous_scroll_y, animate=False)
        self._update_title()

    def get_selected_job(self) -> Optional[Job]:
        row = self.cursor_row
        if row is None or row < 0 or row >= len(self._display_jobs):
            return None
        return self._display_jobs[row]

    def watch_jobs(self, _old: List[Job], _new: List[Job]) -> None:
        self.refresh_table()

    def watch_owner_filter(self, _old: str, _new: str) -> None:
        self.refresh_table()

    def watch_state_filter(self, _old: str, _new: str) -> None:
        self.refresh_table()

    def watch_sort_key(self, _old: str, _new: str) -> None:
        self.refresh_table()

    def watch_sort_desc(self, _old: bool, _new: bool) -> None:
        self.refresh_table()

    async def action_open_sort_menu(self) -> None:
        await self.app.action_open_sort_picker()

    async def action_toggle_sort_direction(self) -> None:
        await self.app.action_toggle_sort_direction()

    async def action_cycle_owner_filter(self) -> None:
        await self.app.action_cycle_owner_filter()

    async def action_open_details(self) -> None:
        await self.app.action_open_selected_job()

    def on_data_table_row_highlighted(self) -> None:
        self._update_title()


class NodesView(Static):
    can_focus = True
    # layout=True: the panel height is auto; without a relayout on change the
    # widget stays stuck at the height of the initial (empty) render.
    nodes: reactive[List[Node]] = reactive([], layout=True)  # type: ignore

    def render(self) -> Table:
        table = Table(box=None, show_edge=False, pad_edge=False)
        table.add_column("NODE", style="cyan")
        table.add_column("STATE", style="bold")
        table.add_column("CPUS(T)")
        table.add_column("CPUS(alloc)")
        table.add_column("CPUS(idle)")
        table.add_column("MEM(total)")
        table.add_column("MEM(resv)")
        table.add_column("MEM(free)")
        table.add_column("GPUs(total)")
        for n in self.nodes:
            state_style = "green" if n.state.startswith("idle") else "yellow"
            table.add_row(
                n.name,
                Text(n.state, style=state_style),
                n.cpus_total,
                n.cpus_alloc,
                n.cpus_idle,
                _format_mb_human(_parse_int(n.mem_total)),
                _format_mb_human(_parse_int(n.mem_reserved)),
                _format_mb_human(_parse_int(n.mem_free)),
                str(sum(_parse_gpu_inventory(n.gres).values())),
            )
        return table


class GpuStatusView(DataTable[str]):
    BINDINGS = [("enter", "open_gpu_jobs", "GPU jobs")]
    stats: reactive[Dict[str, object]] = reactive({})  # type: ignore
    jobs: reactive[List[Job]] = reactive([])  # type: ignore
    _row_gpu_types: List[Optional[str]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._row_gpu_types = []

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_columns("TYPE", "TOTAL", "ACTIVE", "RESERVED", "FREE")
        self.refresh_table()

    def refresh_table(self) -> None:
        selected_gpu = self.get_selected_gpu_type()
        s = self.stats or {"total": 0, "types_count": 0, "per_type": {}, "active": 0, "reserved": 0, "free_est": 0}
        self.clear(columns=False)
        self._row_gpu_types = []
        self.add_row("ALL", str(s.get("total", 0)), str(s.get("active", 0)), str(s.get("reserved", 0)), str(s.get("free_est", 0)))
        self._row_gpu_types.append(None)

        per_type_stats = s.get("per_type_stats", {})
        if isinstance(per_type_stats, dict):
            for gpu_type, stats in sorted(per_type_stats.items()):
                if isinstance(stats, dict):
                    self.add_row(gpu_type, str(stats.get("total", 0)), str(stats.get("active", 0)), str(stats.get("reserved", 0)), str(stats.get("free_est", 0)))
                    self._row_gpu_types.append(gpu_type)

        if not self._row_gpu_types:
            return
        if selected_gpu:
            for idx, gpu_type in enumerate(self._row_gpu_types):
                if gpu_type == selected_gpu:
                    self.move_cursor(row=idx)
                    return
        self.move_cursor(row=0)

    def get_selected_gpu_type(self) -> Optional[str]:
        row = self.cursor_row
        if row is None or row < 0 or row >= len(self._row_gpu_types):
            return None
        return self._row_gpu_types[row]

    def watch_stats(self, _old: Dict[str, object], _new: Dict[str, object]) -> None:
        self.refresh_table()

    async def action_open_gpu_jobs(self) -> None:
        await self.app.action_open_selected_gpu_jobs()


class DiskUsageView(Static):
    can_focus = True
    # layout=True: see NodesView — auto-height panel needs a relayout on change.
    disks: reactive[List[DiskUsage]] = reactive([], layout=True)  # type: ignore

    def render(self) -> Table:
        table = Table(box=None, show_edge=False, pad_edge=False)
        table.add_column("USAGE")
        table.add_column("PATH", style="cyan")
        table.add_column("TYPE")
        table.add_column("SPACE")
        for d in self.disks:
            table.add_row(d.usage_percent, d.mount, d.fs_type, d.size)
        return table


class SummaryBar(Static):
    can_focus = True
    summary: reactive[Dict[str, Dict[str, Dict[str, int]]]] = reactive({})  # type: ignore

    def render(self) -> Table:
        empty = {"running": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}, "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}}
        s = self.summary or {"all": empty, "me": empty, "others": empty}
        table = Table(box=None, show_edge=False, pad_edge=False)
        table.add_column("Owner", style="bold")
        table.add_column("Running", style="green")
        table.add_column("Pending", style="yellow")

        def fmt_block(bucket: str, state: str) -> str:
            data = s.get(bucket, {}).get(state, {})
            return f"{data.get('jobs', 0)} / {data.get('gpus', 0)} / {data.get('cpus', 0)} / {_format_mb_human(data.get('mem_mb', 0))}"

        table.add_row("All", fmt_block("all", "running"), fmt_block("all", "pending"))
        table.add_row("Me", fmt_block("me", "running"), fmt_block("me", "pending"))
        table.add_row("Others", fmt_block("others", "running"), fmt_block("others", "pending"))
        return table


class JobDetailsModal(ModalScreen[None]):
    BINDINGS = [
        ("enter", "dismiss", "Close"),
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("c", "cancel_job", "Cancel"),
        ("h", "hold_job", "Hold"),
        ("u", "release_job", "Release"),
        ("r", "requeue_job", "Requeue"),
        ("f", "manual_refresh", "Refresh"),
        ("a", "toggle_auto_update", "Auto-update"),
    ]

    AUTO_UPDATE_INTERVAL = 3.0
    # Column index of the auto-update toggle in the #job-actions row.
    _AUTO_COLUMN = 5

    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job
        self.detail: Dict[str, str] = {}
        self.usage: Dict[str, str] = {}
        self.auto_update = bool(load_config().get("job_details_auto_update", False))
        self._auto_timer: Optional[Timer] = None

    _EMPTY_FIELDS = {"", "(null)", "N/A", "None", "Unknown"}

    @staticmethod
    def _state_style(state: str) -> str:
        s = state.upper()
        if s.startswith("R"):
            return "bold green"
        if s.startswith("P"):
            return "bold yellow"
        if s.startswith(("CA", "F", "TO", "NF", "OOM", "DL", "BF")):
            return "bold red"
        if s.startswith("C"):
            return "bold cyan"
        return "bold"

    def _field(self, key: str, default: str = "-") -> str:
        value = (self.detail.get(key) or "").strip()
        return default if value in self._EMPTY_FIELDS else value

    def _main_text(self) -> Text:
        """The compact key/value block (everything except WkDir / Cmd)."""
        j = self.job
        u = self.usage
        gd = self._field
        dim = "dim"

        state = gd("JobState", j.state)
        kind = "alloc" if state.upper().startswith("R") else "req"

        tres = ""
        for key in ("AllocTRES", "TRES", "ReqTRES"):
            cand = (self.detail.get(key) or "").strip()
            if cand and cand not in self._EMPTY_FIELDS:
                tres = cand
                break

        mem = _tres_value(tres, "mem") or gd("MinMemoryNode", j.mem)
        nodelist = gd("NodeList", j.node_list or "-")
        gpu_count = _parse_gpu_count(j.gpus)

        rows = [
            Text.assemble(
                ("Job ", dim), (j.job_id, "bold"),
                "    ", (state, self._state_style(state)),
                "    ", ("Reason ", dim), gd("Reason"),
            ),
            Text.assemble(("Name   ", dim), gd("JobName", j.name)),
            Text.assemble(
                ("User   ", dim), gd("UserId", j.user),
                "    ", ("Account ", dim), gd("Account"),
                "    ", ("QOS ", dim), gd("QOS"),
                "    ", ("Prio ", dim), gd("Priority"),
            ),
            Text.assemble(
                ("Part   ", dim), gd("Partition", j.partition),
                "    ", ("Nodes ", dim), f"{gd('NumNodes', j.nodes)} [{nodelist}]",
                "    ", ("Batch ", dim), gd("BatchHost"),
            ),
            Text.assemble(
                ("Time   ", dim), ("run ", dim), gd("RunTime", j.time_used),
                "    ", ("limit ", dim), gd("TimeLimit"),
            ),
            Text.assemble(
                ("Sched  ", dim),
                ("submit ", dim), _short_time(self.detail.get("SubmitTime")),
                "    ", ("start ", dim), _short_time(self.detail.get("StartTime")),
            ),
            Text.assemble(
                ("CPUs   ", dim), f"{gd('NumCPUs', j.ncpus)} ", (kind, dim),
                "    ", ("per-task ", dim), gd("CPUs/Task"),
                "    ", ("tasks ", dim), gd("NumTasks"),
            ),
            Text.assemble(
                ("Memory ", dim), _human_mem(mem), " ", (f"({kind})", dim)
            ),
        ]
        if gpu_count:
            rows.append(Text.assemble(
                ("GPUs   ", dim), str(gpu_count), "  ", (f"({j.gpus})", dim)
            ))
        if u.get("MaxRSS"):
            rows.append(Text.assemble(
                ("Usage  ", dim), ("MaxRSS ", dim), _human_mem(u.get("MaxRSS", "")),
                "    ", ("MaxVM ", dim), _human_mem(u.get("MaxVMSize", "")),
                "    ", ("AveCPU ", dim), u.get("AveCPU") or "-",
                ("   sstat live", dim),
            ))
        if tres:
            rows.append(Text.assemble(("TRES   ", dim), tres))
        return Text("\n").join(rows)

    def _paths_text(self) -> Text:
        """WkDir + Cmd: the two long lines, shown in one shared scroller."""
        # srun --wrap jobs leave Command empty; fall back to the submit line.
        command = self._field("Command")
        if command == "-":
            command = self._field("SubmitLine")
        return Text("\n").join([
            Text.assemble(("WkDir  ", "dim"), self._field("WorkDir")),
            Text.assemble(("Cmd    ", "dim"), command),
        ])

    def _auto_label(self) -> str:
        return f"a Auto-update: {'ON' if self.auto_update else 'OFF'}"

    def compose(self) -> ComposeResult:
        # The compact block never scrolls; WkDir + Cmd share one horizontal
        # scroller so a single scrollbar covers just those two long lines.
        with Vertical(id="job-details-box"):
            yield Static(self._main_text(), id="job-details-body")
            with HorizontalScroll(id="job-paths-scroll"):
                yield Static(self._paths_text(), id="job-paths")
        yield DataTable(id="job-actions")
        yield Static(
            f"Status: auto-update {'ON' if self.auto_update else 'OFF'}",
            id="job-details-status",
        )

    def on_mount(self) -> None:
        actions = self.query_one("#job-actions", DataTable)
        actions.cursor_type = "cell"
        actions.zebra_stripes = False
        actions.show_header = False
        actions.add_columns("", "", "", "", "", "", "")
        actions.add_row(
            "c Cancel", "h Hold", "u Release", "r Requeue",
            "f Refresh", self._auto_label(), "Enter/Esc/q Close",
        )
        actions.cursor_background_priority = "css"
        actions.cursor_foreground_priority = "css"
        actions.move_cursor(row=0, column=6)
        self.set_focus(actions)
        self.query_one("#job-details-box", Vertical).border_title = "Job details"
        # Auto-update timer is Textual-managed (stopped on unmount); start it
        # paused unless the persisted setting has auto-update on.
        self._auto_timer = self.set_interval(
            self.AUTO_UPDATE_INTERVAL, self._auto_tick, pause=not self.auto_update
        )
        # squeue gave only a summary; pull full detail right away.
        self._request_refresh("opened")

    def _set_status(self, text: str) -> None:
        self.query_one("#job-details-status", Static).update(text)

    def _refresh_body(self) -> None:
        self.query_one("#job-details-body", Static).update(self._main_text())
        self.query_one("#job-paths", Static).update(self._paths_text())

    def _request_refresh(self, source: str) -> None:
        # Runs as a Textual worker: cancelled automatically when the modal
        # closes, and exclusive so refreshes never pile up.
        self.run_worker(
            self._do_refresh(source), group="job-detail-refresh", exclusive=True
        )

    def _auto_tick(self) -> None:
        self._request_refresh("auto")

    async def _do_refresh(self, source: str) -> None:
        job, detail, usage = await asyncio.to_thread(collect_job_info, self.job.job_id)
        if not self.is_mounted:
            return
        stamp = time.strftime("%H:%M:%S")
        if job is None:
            self._set_status(
                f"Status: job {self.job.job_id} no longer in queue "
                f"(finished/cancelled) - checked {stamp}"
            )
            return
        self.job = job
        self.detail = detail
        self.usage = usage
        self._refresh_body()
        self._set_status(f"Status: updated ({source}) at {stamp}")

    def action_manual_refresh(self) -> None:
        self._request_refresh("manual")

    def action_toggle_auto_update(self) -> None:
        self.auto_update = not self.auto_update
        config = load_config()
        config["job_details_auto_update"] = self.auto_update
        save_config(config)
        actions = self.query_one("#job-actions", DataTable)
        actions.update_cell_at(Coordinate(0, self._AUTO_COLUMN), self._auto_label())
        if self._auto_timer is not None:
            if self.auto_update:
                self._auto_timer.resume()
            else:
                self._auto_timer.pause()
        if self.auto_update:
            self._request_refresh("auto")
        else:
            self._set_status("Status: auto-update OFF")

    async def _run_action_by_column(self, column: int) -> None:
        if column == 0:
            await self.action_cancel_job()
            return
        if column == 1:
            await self.action_hold_job()
            return
        if column == 2:
            await self.action_release_job()
            return
        if column == 3:
            await self.action_requeue_job()
            return
        if column == 4:
            self.action_manual_refresh()
            return
        if column == 5:
            self.action_toggle_auto_update()
            return
        if column == 6:
            self.dismiss()
            return

    async def _run_job_action(self, command: List[str], action_name: str) -> None:
        ok, output = await asyncio.to_thread(run_cmd_checked, command)
        if not self.is_mounted:
            return
        status = f"Status: {action_name} {'OK' if ok else 'FAILED'} - {output}"
        self.query_one("#job-details-status", Static).update(status)
        if ok:
            await self.app.refresh_data()

    async def action_cancel_job(self) -> None:
        await self._run_job_action(["scancel", self.job.job_id], "cancel")

    async def action_hold_job(self) -> None:
        await self._run_job_action(["scontrol", "hold", self.job.job_id], "hold")

    async def action_release_job(self) -> None:
        await self._run_job_action(["scontrol", "release", self.job.job_id], "release")

    async def action_requeue_job(self) -> None:
        await self._run_job_action(["scontrol", "requeue", self.job.job_id], "requeue")

    async def on_key(self, event: Key) -> None:
        if event.key != "enter":
            return
        actions = self.query_one("#job-actions", DataTable)
        if self.focused is not actions:
            return
        event.stop()
        column = actions.cursor_column
        if column is None:
            return
        await self._run_action_by_column(column)

    async def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        if event.data_table.id != "job-actions":
            return
        await self._run_action_by_column(event.coordinate.column)


class SortPickerModal(ModalScreen[None]):
    BINDINGS = [
        ("enter", "apply_selected", "Apply"),
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    OPTIONS = [
        ("state", "State", "1"),
        ("jobid", "Job ID", "2"),
        ("user", "User", "3"),
        ("partition", "Partition", "4"),
        ("cpus", "CPUs", "5"),
        ("gpus", "GPUs", "6"),
        ("mem", "Memory", "7"),
        ("time", "Time", "8"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("Sort by: choose row + Enter, or press hotkey 1..8", id="sort-help")
        yield DataTable(id="sort-table")

    def on_mount(self) -> None:
        table = self.query_one("#sort-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Key", "Field")
        for _, label, hotkey in self.OPTIONS:
            table.add_row(hotkey, label)
        table.move_cursor(row=0)
        self.set_focus(table)

    async def _apply_sort_index(self, index: int) -> None:
        if index < 0 or index >= len(self.OPTIONS):
            return
        sort_key, label, _ = self.OPTIONS[index]
        app = self.app
        if isinstance(app, SlurmHtop):
            app.jobs_view.sort_key = sort_key
            app.notify(f"Sort by {label.lower()}")
        self.dismiss()

    async def action_apply_selected(self) -> None:
        table = self.query_one("#sort-table", DataTable)
        row = table.cursor_row
        if row is None:
            return
        await self._apply_sort_index(row)

    async def on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.stop()
            await self.action_apply_selected()
            return
        if event.key in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            event.stop()
            await self._apply_sort_index(int(event.key) - 1)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "sort-table":
            return
        await self._apply_sort_index(event.cursor_row)


class GpuJobsModal(ModalScreen[None]):
    BINDINGS = [
        ("enter", "dismiss", "Close"),
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def __init__(self, gpu_type: str, jobs: List[Job]) -> None:
        super().__init__()
        self.gpu_type = gpu_type
        self.jobs = jobs

    def compose(self) -> ComposeResult:
        yield Static(f"GPU type: {self.gpu_type} (using + reserving jobs)", id="gpu-jobs-title")
        yield DataTable(id="gpu-jobs-table")

    def on_mount(self) -> None:
        table = self.query_one("#gpu-jobs-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("MODE", "JOBID", "USER", "STATE", "PART", "NAME", "NODES", "CPUS", "GPUS", "TIME")
        rows = []
        for job in self.jobs:
            per_types = _parse_gpu_per_type(job.gpus)
            count = per_types.get(self.gpu_type, 0)
            if count <= 0:
                continue
            st = job.state.upper()
            if st.startswith("R"):
                mode = "USING"
            elif st.startswith("P"):
                mode = "RESERVING"
            else:
                continue
            rows.append((mode, job))

        rows.sort(key=lambda item: (0 if item[0] == "USING" else 1, _job_id_sort_key(item[1].job_id)))
        for mode, job in rows:
            table.add_row(mode, job.job_id, job.user, job.state, job.partition, job.name, job.nodes, job.ncpus, str(_parse_gpu_count(job.gpus)), job.time_used)

        if table.row_count == 0:
            table.add_row("-", "-", "-", "-", "-", "No using/reserving jobs for this GPU type", "-", "-", "-", "-")
        table.move_cursor(row=0)
        self.set_focus(table)


class SlurmHtop(App):
    TITLE = "slurm-top"
    CSS = """
    Screen { layout: vertical; }
    #main-split { height: 3fr; }
    #left-column { width: 3fr; height: 1fr; }
    #nodes-column { width: 2fr; height: 1fr; }
    #bottom-row { height: 1fr; }
    #gpu-column, #disk-column, #summary-column { width: 1fr; height: 1fr; }
    #jobs-scroll { height: 1fr; }
    #jobs-scroll, #nodes-scroll, #gpu-scroll, #disk-scroll, #summary-scroll {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-color: $panel-darken-1;
        scrollbar-color-hover: $panel;
        scrollbar-color-active: $accent;
        scrollbar-corner-color: $surface;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
    }
    #jobs, #gpu-status {
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-color: $panel-darken-1;
        scrollbar-color-hover: $panel;
        scrollbar-color-active: $accent;
        scrollbar-corner-color: $surface;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
    }
    #summary { height: auto; }
    #summary {
        content-align: center middle;
    }
    #nodes { height: auto; }
    #disk-usage { height: auto; }
    /* GpuStatusView is a DataTable: with height:auto a short panel clips
       trailing rows AND collapses virtual_size, hiding GPU types with no
       scrollbar. Fill the scroll viewport so the table scrolls its rows. */
    #gpu-status { height: 1fr; }
    #jobs-scroll, #nodes-scroll, #gpu-scroll, #disk-scroll, #summary-scroll {
        border: round $panel;
        padding: 0 1;
    }
    JobDetailsModal {
        align: center middle;
    }
    SortPickerModal {
        align: center middle;
    }
    GpuJobsModal {
        align: center middle;
    }
    #job-details-box {
        width: 96;
        height: auto;
        max-height: 80%;
        border: round $accent;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }
    #job-details-body {
        width: 1fr;
        height: auto;
    }
    #job-paths-scroll {
        width: 1fr;
        height: 3;
        overflow-y: hidden;
        scrollbar-size-horizontal: 1;
    }
    #job-paths-scroll:focus {
        background: $boost;
    }
    #job-paths {
        width: auto;
        height: 2;
    }
    #job-details-status {
        width: 96;
        border: round $boost;
        padding: 0 2;
        background: $surface;
    }
    #job-actions {
        width: 96;
        height: 3;
        border: round $boost;
        background: $surface;
    }
    #job-actions:focus {
        border: round $accent;
    }
    #job-actions > .datatable--cursor {
        background: $accent 60%;
        color: $text;
        text-style: bold;
    }
    #sort-help {
        width: 44;
        border: round $panel;
        padding: 0 1;
    }
    #sort-table {
        width: 44;
        height: 10;
        border: round $accent;
    }
    #gpu-jobs-title {
        width: 130;
        border: round $panel;
        padding: 0 1;
        content-align: center middle;
    }
    #gpu-jobs-table {
        width: 130;
        height: 20;
        border: round $accent;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-color: $panel-darken-1;
        scrollbar-color-hover: $panel;
        scrollbar-color-active: $accent;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("s", "open_sort_picker", "Sort"),
        ("d", "toggle_sort_direction", "Asc/Desc"),
        ("f", "cycle_owner_filter", "Owner"),
        ("alt+left", "shrink_focused_panel", "Pane-"),
        ("alt+right", "grow_focused_panel", "Pane+"),
        ("0", "reset_layout", "Reset"),
    ]

    REFRESH_INTERVAL = 3.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jobs_view = JobsView(id="jobs")
        self.nodes_view = NodesView(id="nodes")
        self.gpu_status_view = GpuStatusView(id="gpu-status")
        self.disk_usage_view = DiskUsageView(id="disk-usage")
        self.summary_bar = SummaryBar(id="summary")
        self.top_row_ratio = 3
        self.bottom_row_ratio = 1
        self.top_left_ratio = 3
        self.top_right_ratio = 2
        self.bottom_ratios = [1, 1, 1]  # gpu, disk, summary

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-split"):
            with Vertical(id="left-column"):
                with VerticalScroll(id="jobs-scroll"):
                    yield self.jobs_view
            with Vertical(id="nodes-column"):
                with VerticalScroll(id="nodes-scroll"):
                    yield self.nodes_view
        with Horizontal(id="bottom-row"):
            with Vertical(id="gpu-column"):
                with VerticalScroll(id="gpu-scroll"):
                    yield self.gpu_status_view
            with Vertical(id="disk-column"):
                with VerticalScroll(id="disk-scroll"):
                    yield self.disk_usage_view
            with Vertical(id="summary-column"):
                with VerticalScroll(id="summary-scroll"):
                    yield self.summary_bar
        yield Footer()

    async def on_mount(self) -> None:
        self.set_focus(self.jobs_view)
        self._apply_layout_ratios()
        self.query_one("#jobs-scroll", VerticalScroll).border_title = "Jobs"
        self.query_one("#nodes-scroll", VerticalScroll).border_title = "Nodes"
        self.query_one("#gpu-scroll", VerticalScroll).border_title = "GPU status"
        self.query_one("#disk-scroll", VerticalScroll).border_title = "Disks"
        self.query_one("#summary-scroll", VerticalScroll).border_title = "Job statistics (jobs / GPUs / CPUs / MEM)"
        # Textual-managed timer (stopped automatically on shutdown) instead of a
        # raw asyncio task; the refresh itself runs in a worker.
        self._refresh_tick()
        self.set_interval(self.REFRESH_INTERVAL, self._refresh_tick)

    def _apply_layout_ratios(self) -> None:
        self.query_one("#main-split", Horizontal).styles.height = f"{self.top_row_ratio}fr"
        self.query_one("#bottom-row", Horizontal).styles.height = f"{self.bottom_row_ratio}fr"
        self.query_one("#left-column", Vertical).styles.width = f"{self.top_left_ratio}fr"
        self.query_one("#nodes-column", Vertical).styles.width = f"{self.top_right_ratio}fr"
        self.query_one("#gpu-column", Vertical).styles.width = f"{self.bottom_ratios[0]}fr"
        self.query_one("#disk-column", Vertical).styles.width = f"{self.bottom_ratios[1]}fr"
        self.query_one("#summary-column", Vertical).styles.width = f"{self.bottom_ratios[2]}fr"

    def _grow_split(self, left_attr: str, right_attr: str) -> None:
        setattr(self, left_attr, getattr(self, left_attr) + 1)
        right = getattr(self, right_attr)
        if right > 1:
            setattr(self, right_attr, right - 1)
        self._apply_layout_ratios()

    def _shrink_split(self, left_attr: str, right_attr: str) -> None:
        left = getattr(self, left_attr)
        if left <= 1:
            return
        setattr(self, left_attr, left - 1)
        setattr(self, right_attr, getattr(self, right_attr) + 1)
        self._apply_layout_ratios()

    def _focused_panel(self) -> str:
        focused = self.focused
        if focused is None:
            return "jobs"
        ids = set()
        node = focused
        while node is not None:
            if node.id:
                ids.add(node.id)
            node = node.parent
        if "jobs" in ids or "jobs-scroll" in ids:
            return "jobs"
        if "nodes" in ids or "nodes-scroll" in ids:
            return "nodes"
        if "gpu-status" in ids or "gpu-scroll" in ids:
            return "gpu"
        if "disk-usage" in ids or "disk-scroll" in ids:
            return "disk"
        return "summary"

    def _focused_bottom_index(self) -> int:
        panel = self._focused_panel()
        if panel == "gpu":
            return 0
        if panel == "disk":
            return 1
        return 2

    def _grow_bottom_focused(self) -> None:
        idx = self._focused_bottom_index()
        donors = [i for i, v in enumerate(self.bottom_ratios) if i != idx and v > 1]
        if not donors:
            return
        donor = max(donors, key=lambda i: self.bottom_ratios[i])
        self.bottom_ratios[idx] += 1
        self.bottom_ratios[donor] -= 1
        self._apply_layout_ratios()

    def _shrink_bottom_focused(self) -> None:
        idx = self._focused_bottom_index()
        if self.bottom_ratios[idx] <= 1:
            return
        receiver = (idx + 1) % len(self.bottom_ratios)
        self.bottom_ratios[idx] -= 1
        self.bottom_ratios[receiver] += 1
        self._apply_layout_ratios()

    def _refresh_tick(self) -> None:
        # exclusive: a still-running refresh is cancelled rather than piling up.
        self.run_worker(
            self.refresh_data(), group="cluster-refresh", exclusive=True
        )

    @staticmethod
    def _collect_cluster_data() -> "tuple[List[Job], List[Node], List[DiskUsage]]":
        jobs = sort_jobs(parse_squeue())
        nodes = parse_sinfo()
        disks = parse_disks()
        return jobs, nodes, disks

    async def refresh_data(self) -> None:
        # Run the blocking squeue/sinfo/df calls off the event-loop thread so a
        # slow or stuck command never freezes the UI (or wedges shutdown).
        jobs, nodes, disks = await asyncio.to_thread(self._collect_cluster_data)
        if not self.is_running:
            return
        with self.batch_update():
            self.jobs_view.jobs = jobs
            self.nodes_view.nodes = nodes
            self.disk_usage_view.disks = disks
            self.gpu_status_view.jobs = jobs
            self.gpu_status_view.stats = summarize_gpus(nodes, jobs)
            self.summary_bar.summary = summarize_jobs(jobs, self.jobs_view.user)

    async def action_refresh(self) -> None:
        await self.refresh_data()

    async def action_open_sort_picker(self) -> None:
        await self.push_screen(SortPickerModal())

    async def action_toggle_sort_direction(self) -> None:
        self.jobs_view.sort_desc = not self.jobs_view.sort_desc

    async def action_cycle_owner_filter(self) -> None:
        options = ["all", "me", "others"]
        idx = options.index(self.jobs_view.owner_filter) if self.jobs_view.owner_filter in options else 0
        self.jobs_view.owner_filter = options[(idx + 1) % len(options)]

    async def action_open_selected_job(self) -> None:
        selected_job = self.jobs_view.get_selected_job()
        if not selected_job:
            self.notify("No job selected")
            return
        await self.push_screen(JobDetailsModal(selected_job))

    async def action_open_selected_gpu_jobs(self) -> None:
        gpu_type = self.gpu_status_view.get_selected_gpu_type()
        if not gpu_type:
            self.notify("Select a GPU type row first")
            return
        await self.push_screen(GpuJobsModal(gpu_type, self.jobs_view.jobs))

    async def action_grow_focused_panel(self) -> None:
        panel = self._focused_panel()
        if panel == "jobs":
            self._grow_split("top_left_ratio", "top_right_ratio")
            return
        if panel == "nodes":
            self._grow_split("top_right_ratio", "top_left_ratio")
            return
        self._grow_bottom_focused()

    async def action_shrink_focused_panel(self) -> None:
        panel = self._focused_panel()
        if panel == "jobs":
            self._shrink_split("top_left_ratio", "top_right_ratio")
            return
        if panel == "nodes":
            self._shrink_split("top_right_ratio", "top_left_ratio")
            return
        self._shrink_bottom_focused()

    async def action_reset_layout(self) -> None:
        self.top_row_ratio = 3
        self.bottom_row_ratio = 1
        self.top_left_ratio = 3
        self.top_right_ratio = 2
        self.bottom_ratios = [1, 1, 1]
        self._apply_layout_ratios()

def main() -> None:
    SlurmHtop().run()
