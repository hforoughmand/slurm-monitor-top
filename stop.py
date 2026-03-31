import asyncio
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Dict

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input
from textual.reactive import reactive
from textual.containers import Horizontal, VerticalScroll


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


def run_cmd(cmd: str) -> str:
    try:
        out = subprocess.check_output(
            shlex.split(cmd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out
    except Exception:
        return ""


def parse_squeue() -> List[Job]:
    # Customize the output format as needed
    format_str = "%i|%u|%T|%P|%j|%D|%C|%m|%M"
    raw = run_cmd(f"squeue -a -o '{format_str}'")
    lines = raw.strip().splitlines()
    jobs: List[Job] = []

    # Skip header if present
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) != 9:
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
                time_used=parts[8].strip(),
            )
        )
    return jobs


def parse_sinfo() -> List[Node]:
    format_str = "%n|%t|%c|%C|%m|%e"
    raw = run_cmd(f"sinfo -o '{format_str}'")
    lines = raw.strip().splitlines()
    nodes: List[Node] = []

    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) != 6:
            continue
        name = parts[0].strip()
        state = parts[1].strip()
        cpus_total = parts[2].strip()
        c_state = parts[3].strip()  # format: "alloc/idle/other"
        mem_total = parts[4].strip()
        mem_free = parts[5].strip()

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
            )
        )
    return nodes


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _parse_mem_to_mb(value: str) -> int:
    """Parse Slurm-style memory strings (e.g. 4000M, 4G) into MB."""
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
    # Default MB for M, K, or unknown
    if unit.startswith("K"):
        return int(base / 1024)
    return int(base)


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
    # Prioritize active jobs, then older jobs (smaller id) on top.
    return sorted(jobs, key=lambda j: (_job_state_rank(j.state), _job_id_sort_key(j.job_id)))


def summarize_jobs(jobs: List[Job], current_user: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    # Nested dict: owner -> state -> metrics
    summary: Dict[str, Dict[str, Dict[str, int]]] = {
        "all": {
            "running": {"jobs": 0, "cpus": 0, "mem_mb": 0},
            "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0},
        },
        "me": {
            "running": {"jobs": 0, "cpus": 0, "mem_mb": 0},
            "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0},
        },
        "others": {
            "running": {"jobs": 0, "cpus": 0, "mem_mb": 0},
            "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0},
        },
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

        for bucket in ("all", owner):
            summary[bucket][key]["jobs"] += 1
            summary[bucket][key]["cpus"] += cpus
            summary[bucket][key]["mem_mb"] += mem_mb

    return summary


class JobsView(Static):
    jobs: reactive[List[Job]] = reactive([])  # type: ignore
    only_me: reactive[bool] = reactive(False)  # type: ignore
    search_query: reactive[str] = reactive("")  # type: ignore
    user: str = os.environ.get("USER", "")

    def render(self) -> Table:
        title = "Slurm Jobs (f: my jobs, /: search)"
        if self.search_query:
            title += f" | filter='{self.search_query}'"
        table = Table(title=title)
        table.add_column("JOBID", justify="right", no_wrap=True)
        table.add_column("USER", style="cyan")
        table.add_column("STATE", style="bold")
        table.add_column("PART", style="magenta")
        table.add_column("NAME", overflow="fold")
        table.add_column("NODES")
        table.add_column("CPUS")
        table.add_column("MEM")
        table.add_column("TIME")

        query = self.search_query.lower().strip()
        for j in self.jobs:
            if self.only_me and j.user != self.user:
                continue
            if query:
                haystack = " ".join(
                    [
                        j.job_id,
                        j.user,
                        j.state,
                        j.partition,
                        j.name,
                        j.nodes,
                        j.ncpus,
                        j.mem,
                        j.time_used,
                    ]
                ).lower()
                if query not in haystack:
                    continue
            style = None
            if j.state.upper().startswith("R"):
                style = "green"
            elif j.state.upper().startswith("P"):
                style = "yellow"
            elif j.state.upper().startswith("F"):
                style = "red"

            table.add_row(
                j.job_id,
                j.user,
                Text(j.state, style=style),
                j.partition,
                j.name,
                j.nodes,
                j.ncpus,
                j.mem,
                j.time_used,
            )
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
            )
        return table


class SummaryBar(Static):
    summary: reactive[Dict[str, Dict[str, Dict[str, int]]]] = reactive({})  # type: ignore

    def render(self) -> Table:
        # Fallback structure if no data yet
        empty = {
            "running": {"jobs": 0, "cpus": 0, "mem_mb": 0},
            "pending": {"jobs": 0, "cpus": 0, "mem_mb": 0},
        }
        s = self.summary or {
            "all": empty,
            "me": empty,
            "others": empty,
        }

        table = Table(title="Job statistics (jobs / CPUs / MEM)")
        table.add_column("Owner", style="bold")
        table.add_column("Running", style="green")
        table.add_column("Pending", style="yellow")

        def fmt_block(bucket: str, state: str) -> str:
            data = s.get(bucket, {}).get(state, {})
            jobs = data.get("jobs", 0)
            cpus = data.get("cpus", 0)
            mem_mb = data.get("mem_mb", 0)
            return f"{jobs} / {cpus} / {_format_mb_human(mem_mb)}"

        def add_row(label: str, key: str) -> None:
            running = fmt_block(key, "running")
            pending = fmt_block(key, "pending")
            table.add_row(label, running, pending)

        add_row("All", "all")
        add_row("Me", "me")
        add_row("Others", "others")

        return table


class SlurmHtop(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #search {
        display: none;
    }
    #search.-active {
        display: block;
    }
    #main-row {
        height: 1fr;
    }
    #jobs {
        width: 2fr;
    }
    #nodes {
        width: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh now"),
        ("f", "toggle_only_me", "Toggle my jobs"),
        ("slash", "focus_search", "Search jobs"),
        ("escape", "clear_search", "Clear search"),
    ]

    REFRESH_INTERVAL = 3.0  # seconds

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.jobs_view = JobsView(id="jobs")
        self.nodes_view = NodesView(id="nodes")
        self.summary_bar = SummaryBar(id="summary")
        self.search_input = Input(
            placeholder="Type to filter jobs... (Esc to clear)"
            ,
            id="search",
        )
        self._task = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.search_input
        with Horizontal(id="main-row"):
            with VerticalScroll():
                yield self.jobs_view
            with VerticalScroll():
                yield self.nodes_view
        yield self.summary_bar
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
        self.summary_bar.summary = summarize_jobs(jobs, self.jobs_view.user)

    async def action_refresh(self) -> None:
        await self.refresh_data()

    async def action_toggle_only_me(self) -> None:
        self.jobs_view.only_me = not self.jobs_view.only_me

    async def action_focus_search(self) -> None:
        self.search_input.add_class("-active")
        self.search_input.focus()

    async def action_clear_search(self) -> None:
        self.search_input.value = ""
        self.jobs_view.search_query = ""
        self.search_input.remove_class("-active")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is self.search_input:
            self.jobs_view.search_query = event.value

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is self.search_input:
            self.search_input.remove_class("-active")


if __name__ == "__main__":
    SlurmHtop().run()