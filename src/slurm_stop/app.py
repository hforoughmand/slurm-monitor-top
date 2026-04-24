import asyncio
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, List

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static


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


def run_cmd(cmd: str) -> str:
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL, text=True)
        return out
    except Exception:
        return ""


def parse_squeue() -> List[Job]:
    format_str = "%i|%u|%T|%P|%j|%D|%C|%m|%b|%M"
    raw = run_cmd(f"squeue -a -o '{format_str}'")
    lines = raw.strip().splitlines()
    jobs: List[Job] = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) != 10:
            continue
        jobs.append(
            Job(
                job_id=parts[0].strip(),
                user=parts[1].strip(),
                state=parts[2].strip(),
                partition=parts[3].strip(),
                name=parts[4].strip(),
                nodes=parts[5].strip(),
                ncpus=parts[6].strip(),
                mem=parts[7].strip(),
                gpus=parts[8].strip(),
                time_used=parts[9].strip(),
            )
        )
    return jobs


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


class JobsView(Static):
    jobs: reactive[List[Job]] = reactive([])  # type: ignore
    owner_filter: reactive[str] = reactive("all")  # type: ignore
    state_filter: reactive[str] = reactive("all")  # type: ignore
    sort_key: reactive[str] = reactive("state")  # type: ignore
    sort_desc: reactive[bool] = reactive(False)  # type: ignore
    sort_pick_mode: reactive[bool] = reactive(False)  # type: ignore
    user: str = os.environ.get("USER", "")

    def render(self) -> Table:
        title = "Slurm Jobs " f"(s: pick sort, d: {'desc' if self.sort_desc else 'asc'}, " f"f: owner={self.owner_filter}, state={self.state_filter})"
        if self.sort_pick_mode:
            title += " | pick sort: 1-state 2-jobid 3-user 4-part 5-cpus 6-gpus 7-mem 8-time"
        table = Table(title=title)
        table.add_column("JOBID", justify="right", no_wrap=True)
        table.add_column("USER", style="cyan")
        table.add_column("STATE", style="bold")
        table.add_column("PART", style="magenta")
        table.add_column("NAME", overflow="fold")
        table.add_column("NODES")
        table.add_column("CPUS")
        table.add_column("GPUS")
        table.add_column("MEM")
        table.add_column("TIME")

        def include_owner(job: Job) -> bool:
            if self.owner_filter == "all":
                return True
            if self.owner_filter == "me":
                return job.user == self.user
            return job.user != self.user

        def include_state(job: Job) -> bool:
            st = job.state.upper()
            if self.state_filter == "all":
                return True
            if self.state_filter == "running":
                return st.startswith("R")
            if self.state_filter == "pending":
                return st.startswith("P")
            return not st.startswith("R") and not st.startswith("P")

        def sort_value(job: Job):
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

        jobs = [j for j in self.jobs if include_owner(j) and include_state(j)]
        jobs = sorted(jobs, key=sort_value, reverse=self.sort_desc)
        for j in jobs:
            style = None
            if j.state.upper().startswith("R"):
                style = "green"
            elif j.state.upper().startswith("P"):
                style = "yellow"
            elif j.state.upper().startswith("F"):
                style = "red"
            table.add_row(j.job_id, j.user, Text(j.state, style=style), j.partition, j.name, j.nodes, j.ncpus, str(_parse_gpu_count(j.gpus)), j.mem, j.time_used)
        return table


class NodesView(Static):
    nodes: reactive[List[Node]] = reactive([])  # type: ignore

    def render(self) -> Table:
        table = Table(title="Nodes")
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


class GpuStatusView(Static):
    stats: reactive[Dict[str, object]] = reactive({})  # type: ignore

    def render(self) -> Table:
        s = self.stats or {"total": 0, "types_count": 0, "per_type": {}, "active": 0, "reserved": 0, "free_est": 0}
        table = Table(title="GPU status")
        table.add_column("Category", style="bold")
        table.add_column("Total")
        table.add_column("Active")
        table.add_column("Reserved")
        table.add_column("Free est.")
        table.add_row("ALL", str(s.get("total", 0)), str(s.get("active", 0)), str(s.get("reserved", 0)), str(s.get("free_est", 0)))
        table.add_row("Types", str(s.get("types_count", 0)), "-", "-", "-")
        per_type_stats = s.get("per_type_stats", {})
        if isinstance(per_type_stats, dict):
            for gpu_type, stats in sorted(per_type_stats.items()):
                if isinstance(stats, dict):
                    table.add_row(gpu_type, str(stats.get("total", 0)), str(stats.get("active", 0)), str(stats.get("reserved", 0)), str(stats.get("free_est", 0)))
        return table


class SummaryBar(Static):
    summary: reactive[Dict[str, Dict[str, Dict[str, int]]]] = reactive({})  # type: ignore

    def render(self) -> Table:
        empty = {"running": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}, "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}}
        s = self.summary or {"all": empty, "me": empty, "others": empty}
        table = Table(title="Job statistics (jobs / GPUs / CPUs / MEM)")
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


class SlurmHtop(App):
    CSS = """
    Screen { layout: vertical; }
    #main-split { height: 1fr; }
    #left-column { width: 2fr; height: 1fr; }
    #nodes-column { width: 1fr; height: 1fr; }
    #jobs-scroll { height: 1fr; }
    #summary { height: auto; }
    #nodes-scroll { height: 2fr; }
    #gpu-scroll { height: 1fr; }
    #nodes { height: auto; }
    #gpu-status { height: auto; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Ref"),
        ("s", "toggle_sort_pick_mode", "Sort"),
        ("d", "toggle_sort_direction", "Asc/Desc"),
        ("f", "cycle_owner_filter", "Owner"),
    ]

    REFRESH_INTERVAL = 3.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jobs_view = JobsView(id="jobs")
        self.nodes_view = NodesView(id="nodes")
        self.gpu_status_view = GpuStatusView(id="gpu-status")
        self.summary_bar = SummaryBar(id="summary")
        self._task = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-split"):
            with Vertical(id="left-column"):
                with VerticalScroll(id="jobs-scroll"):
                    yield self.jobs_view
                yield self.summary_bar
            with Vertical(id="nodes-column"):
                with VerticalScroll(id="nodes-scroll"):
                    yield self.nodes_view
                with VerticalScroll(id="gpu-scroll"):
                    yield self.gpu_status_view
        yield Footer()

    async def on_mount(self) -> None:
        self._task = asyncio.create_task(self.poll_loop())

    async def poll_loop(self) -> None:
        while True:
            await self.refresh_data()
            await asyncio.sleep(self.REFRESH_INTERVAL)

    async def refresh_data(self) -> None:
        jobs = sort_jobs(parse_squeue())
        nodes = parse_sinfo()
        self.jobs_view.jobs = jobs
        self.nodes_view.nodes = nodes
        self.gpu_status_view.stats = summarize_gpus(nodes, jobs)
        self.summary_bar.summary = summarize_jobs(jobs, self.jobs_view.user)

    async def action_refresh(self) -> None:
        await self.refresh_data()

    async def action_toggle_sort_pick_mode(self) -> None:
        self.jobs_view.sort_pick_mode = not self.jobs_view.sort_pick_mode

    async def action_toggle_sort_direction(self) -> None:
        self.jobs_view.sort_desc = not self.jobs_view.sort_desc

    async def action_cycle_owner_filter(self) -> None:
        options = ["all", "me", "others"]
        idx = options.index(self.jobs_view.owner_filter) if self.jobs_view.owner_filter in options else 0
        self.jobs_view.owner_filter = options[(idx + 1) % len(options)]


def main() -> None:
    SlurmHtop().run()
