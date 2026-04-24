import asyncio
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.reactive import reactive
from textual.screen import ModalScreen
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


def run_cmd_checked(args: List[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(args, check=False, text=True, capture_output=True)
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
                    self._update_title()
                    return
        self.move_cursor(row=0)
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


class JobDetailsModal(ModalScreen[None]):
    BINDINGS = [
        ("enter", "dismiss", "Close"),
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("c", "cancel_job", "Cancel"),
        ("h", "hold_job", "Hold"),
        ("u", "release_job", "Release"),
        ("r", "requeue_job", "Requeue"),
    ]

    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job

    def compose(self) -> ComposeResult:
        details = [
            f"Job ID   : {self.job.job_id}",
            f"User     : {self.job.user}",
            f"State    : {self.job.state}",
            f"Part     : {self.job.partition}",
            f"Name     : {self.job.name}",
            f"Nodes    : {self.job.nodes}",
            f"CPUs     : {self.job.ncpus}",
            f"GPUs     : {_parse_gpu_count(self.job.gpus)} ({self.job.gpus})",
            f"Memory   : {self.job.mem}",
            f"Run Time : {self.job.time_used}",
        ]
        yield Static("\n".join(details), id="job-details-body")
        yield DataTable(id="job-actions")
        yield Static("Status: waiting for action", id="job-details-status")

    def on_mount(self) -> None:
        actions = self.query_one("#job-actions", DataTable)
        actions.cursor_type = "cell"
        actions.zebra_stripes = False
        actions.show_header = False
        actions.add_columns("", "", "", "", "")
        actions.add_row("c Cancel", "h Hold", "u Release", "r Requeue", "Enter/Esc/q Close")
        actions.cursor_background_priority = "css"
        actions.cursor_foreground_priority = "css"
        actions.move_cursor(row=0, column=4)
        self.set_focus(actions)

    async def _run_action_by_column(self, column: int) -> None:
        self.query_one("#job-details-status", Static).update("Status: running action...")
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
            self.dismiss()
            return

    async def _run_job_action(self, command: List[str], action_name: str) -> None:
        ok, output = run_cmd_checked(command)
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


class SlurmHtop(App):
    TITLE = "slurm-top"
    CSS = """
    Screen { layout: vertical; }
    #main-split { height: 1fr; }
    #left-column { width: 3fr; height: 1fr; }
    #nodes-column { width: 2fr; height: 1fr; }
    #jobs-scroll { height: 1fr; }
    #summary { height: auto; }
    #summary, #gpu-status {
        content-align: center middle;
    }
    #nodes-scroll { height: 1fr; }
    #nodes { height: auto; }
    #gpu-status { height: auto; }
    JobDetailsModal {
        align: center middle;
    }
    SortPickerModal {
        align: center middle;
    }
    #job-details-body {
        width: 80;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    #job-details-status {
        width: 80;
        border: round $boost;
        padding: 0 2;
        background: $surface;
    }
    #job-actions {
        width: 80;
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
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("s", "open_sort_picker", "Sort"),
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
                yield self.gpu_status_view
        yield Footer()

    async def on_mount(self) -> None:
        self.set_focus(self.jobs_view)
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

def main() -> None:
    SlurmHtop().run()
