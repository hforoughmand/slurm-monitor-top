import asyncio
import os
import re
import time
from typing import Dict, List, Optional

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import (
    Horizontal,
    HorizontalScroll,
    ScrollableContainer,
    Vertical,
    VerticalScroll,
)
from textual.coordinate import Coordinate
from textual.events import Key
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, Input, Static

from .data import (
    OUTPUT_TAIL_LINES,
    RISK_LOW,
    DiskUsage,
    Job,
    Node,
    _format_mb_human,
    _human_mem,
    apply_pins,
    _job_id_sort_key,
    _job_state_rank,
    _parse_gpu_count,
    _parse_gpu_inventory,
    _parse_gpu_per_type,
    _parse_int,
    _parse_mem_to_mb,
    _short_time,
    _tres_value,
    collect_job_info,
    collect_node_info,
    cpu_counts,
    cpu_speed,
    cpu_topology,
    describe_cpu,
    describe_cpu_count,
    describe_cpu_speeds,
    fetch_job_detail,
    job_output_paths,
    job_usage_metrics,
    load_config,
    load_cpu_info,
    load_pinned_jobs,
    parse_disks,
    parse_sinfo,
    parse_squeue,
    probe_node_cpu,
    read_file_tail,
    run_cmd_checked,
    save_config,
    short_cpu_model,
    sort_jobs,
    summarize_gpus,
    summarize_jobs,
    toggle_job_pin,
)



# Usage bars. Block characters rather than a widget: these are drawn inside a
# Rich Text block that also carries the numbers, and one string is far cheaper
# to rebuild every three seconds than a row of widgets.
BAR_WIDTH = 20
# The bar is drawn as coloured background on ordinary spaces, not out of the
# block characters (U+2588 / U+2591) the first version used. A space is a
# space in every font: the blocks depend on one that has them, sizes them to
# the cell and does not leave a seam between neighbours, and plenty of
# terminal fonts fail at least one of those - the shaded block in particular
# comes out as dithered noise or an empty box. Nothing can ask a terminal
# which glyphs its font holds, so the only robust answer is not to need any.
#
# What *can* be detected is colour, so a terminal with none falls back to
# these two, which are ASCII and therefore always drawable.
_BAR_ASCII_FULL = "#"
_BAR_ASCII_EMPTY = "-"
# Background for the unfilled part: dark enough to read as "empty" on the
# app's own surface, light enough to show where the bar ends.
_BAR_EMPTY_STYLE = "on grey23"


def _human_bytes(size: int) -> str:
    """Byte counts the way a file listing shows them: 812B, 26.2K, 1.4G."""
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def _usage_style(percent: Optional[float], risk: str) -> str:
    """Colour for a usage figure, by which end of the scale is the bad one.

    A job at 95% of its memory is about to be killed; a job at 5% of its cores
    is wasting fifteen of them. Both deserve red, at opposite ends.
    """
    if percent is None:
        return "dim"
    if risk == RISK_LOW:
        if percent < 25:
            return "red"
        if percent < 60:
            return "yellow"
        return "green"
    if percent >= 90:
        return "red"
    if percent >= 75:
        return "yellow"
    return "green"


def _usage_bar(
    percent: Optional[float], risk: str, width: int = BAR_WIDTH, color: bool = True
) -> Text:
    """One metric as a band, clamped so a figure over 100% still fits the frame.

    ``color`` is what the terminal told us it can do; see the note by
    :data:`_BAR_ASCII_FULL` for why that is the one thing worth asking it.
    """
    style = _usage_style(percent, risk)
    filled = 0 if percent is None else int(round(max(0.0, min(100.0, percent)) / 100.0 * width))
    if not color:
        return Text.assemble(
            ("[", "dim"),
            _BAR_ASCII_FULL * filled,
            (_BAR_ASCII_EMPTY * (width - filled), "dim"),
            ("]", "dim"),
        )
    return Text.assemble(
        ("[", "dim"),
        (" " * filled, f"on {style}" if style != "dim" else _BAR_EMPTY_STYLE),
        (" " * (width - filled), _BAR_EMPTY_STYLE),
        ("]", "dim"),
    )


def _usage_rows(metrics: "List[Dict[str, object]]", color: bool = True) -> "List[Text]":
    """One line per metric: label, bar, percentage, the two numbers, the why."""
    rows: List[Text] = []
    for metric in metrics:
        kind = metric.get("kind")
        note = str(metric.get("note", ""))
        if kind == "note":
            rows.append(Text(note, style="dim italic"))
            continue
        label = f"{str(metric.get('label', '')):<7}"
        if kind == "bar":
            percent = float(metric.get("percent") or 0.0)
            rows.append(Text.assemble(
                (label, "dim"),
                _usage_bar(percent, str(metric.get("risk", "")), color=color),
                (f" {percent:5.1f}%  ", _usage_style(percent, str(metric.get("risk", "")))),
                f"{metric.get('value', '')} of {metric.get('total', '')}",
                (f"   {note}", "dim"),
            ))
            continue
        rows.append(Text.assemble(
            (label, "dim"), str(metric.get("value", "")), (f"   {note}", "dim")
        ))
    return rows


class StableTable(DataTable[str]):
    """DataTable whose viewport survives a full rebuild.

    `clear()` zeroes the scroll offset and resets the cursor, and the
    cursor-coordinate watcher then schedules a scroll-into-view no matter what
    `move_cursor(scroll=False)` asked for. So the saved offset has to be
    re-applied *after* the next refresh, once the virtual size is settled and
    those queued callbacks have run - otherwise the table snaps back to the
    selected row every time the data refreshes.
    """

    def _view_snapshot(self) -> "tuple[float, float]":
        return self.scroll_x, self.scroll_y

    def _restore_view(self, snapshot: "tuple[float, float]", row: Optional[int] = None) -> None:
        if row is not None and row >= 0:
            self.move_cursor(row=row, scroll=False)
        x, y = snapshot

        def apply() -> None:
            self.scroll_to(x=x, y=y, animate=False, force=True)

        apply()
        # Queued after the cursor watcher's own scroll callback, so this wins.
        self.call_after_refresh(apply)

    def _set_panel_title(self, text: str) -> None:
        """Write the title onto the bordered scroll container we sit in."""
        parent = self.parent
        if parent is not None and hasattr(parent, "border_title"):
            parent.border_title = text

    def row_fields(self) -> "List[tuple[str, str]]":
        """(column label, plain text) pairs for the row under the cursor."""
        row = self.cursor_row
        if row is None or row < 0 or row >= self.row_count:
            return []
        labels = [str(getattr(col, "label", "")) for col in self.columns.values()]
        values = [cell.plain if isinstance(cell, Text) else str(cell) for cell in self.get_row_at(row)]
        return list(zip(labels, values))


class JobsView(StableTable):
    BINDINGS = [
        ("s", "open_sort_menu", "Sort"),
        ("d", "toggle_sort_direction", "Asc/Desc"),
        ("f", "cycle_owner_filter", "Owner"),
        ("p", "toggle_pin", "Pin"),
        ("o", "open_output", "Output"),
        ("enter", "open_details", "Details"),
    ]
    # always_update: every poll must rebuild the table, so a job that appears
    # while a filter is active is picked up even if Textual considers the new
    # list equal to the old one.
    jobs: reactive[List[Job]] = reactive([], always_update=True)  # type: ignore
    owner_filter: reactive[str] = reactive("all")  # type: ignore
    state_filter: reactive[str] = reactive("all")  # type: ignore
    search_filter: reactive[str] = reactive("")  # type: ignore
    sort_key: reactive[str] = reactive("state")  # type: ignore
    sort_desc: reactive[bool] = reactive(False)  # type: ignore
    # Shared with the editor extension through the config file, so a job pinned
    # in either front end is already on top in the other.
    pinned: reactive[frozenset] = reactive(frozenset())  # type: ignore
    user: str = os.environ.get("USER", "")
    _display_jobs: List[Job]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._display_jobs = []

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_columns("P", "JOBID", "USER", "STATE", "PART", "NAME", "NODES", "CPUS", "GPUS", "MEM", "TIME")
        self.pinned = frozenset(load_pinned_jobs())
        self.refresh_table()

    def _update_title(self) -> None:
        parts = [f"Jobs {len(self._display_jobs)}/{len(self.jobs)}"]
        parts.append(f"owner={self.owner_filter}")
        if self.state_filter != "all":
            parts.append(f"state={self.state_filter}")
        parts.append(f"sort={self.sort_key} {'desc' if self.sort_desc else 'asc'}")
        if self.pinned:
            parts.append(f"pinned={sum(1 for j in self._display_jobs if j.job_id in self.pinned)}")
        if self.search_filter:
            parts.append(f'find="{self.search_filter}"')
        self._set_panel_title(" | ".join(parts))

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

    def _include_search(self, job: Job) -> bool:
        """Space separated terms, all of which must appear somewhere in the row."""
        terms = self.search_filter.lower().split()
        if not terms:
            return True
        haystack = " ".join([
            job.job_id, job.user, job.state, job.partition, job.name,
            job.nodes, job.ncpus, job.mem, job.gpus, job.time_used, job.node_list,
        ]).lower()
        return all(term in haystack for term in terms)

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
        snapshot = self._view_snapshot()
        previous_row = self.cursor_row if self.cursor_row is not None else 0
        selected = self.get_selected_job()
        selected_job_id = selected.job_id if selected else None

        self.clear(columns=False)
        self._display_jobs = [
            j for j in self.jobs
            if self._include_owner(j) and self._include_state(j) and self._include_search(j)
        ]
        self._display_jobs = sorted(self._display_jobs, key=self._sort_value, reverse=self.sort_desc)
        self._display_jobs = apply_pins(self._display_jobs, self.pinned)
        for j in self._display_jobs:
            style = None
            if j.state.upper().startswith("R"):
                style = "green"
            elif j.state.upper().startswith("P"):
                style = "yellow"
            elif j.state.upper().startswith("F"):
                style = "red"
            is_pinned = j.job_id in self.pinned
            self.add_row(
                Text("*" if is_pinned else "", style="bold magenta"),
                Text(j.job_id, style="bold" if is_pinned else None),
                j.user, Text(j.state, style=style), j.partition, j.name, j.nodes,
                j.ncpus, str(_parse_gpu_count(j.gpus)), j.mem, j.time_used,
            )

        self._update_title()
        if not self._display_jobs:
            return

        row = min(max(previous_row, 0), len(self._display_jobs) - 1)
        if selected_job_id is not None:
            for idx, job in enumerate(self._display_jobs):
                if job.job_id == selected_job_id:
                    row = idx
                    break
        self._restore_view(snapshot, row)
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

    def watch_search_filter(self, _old: str, _new: str) -> None:
        self.refresh_table()

    def watch_sort_key(self, _old: str, _new: str) -> None:
        self.refresh_table()

    def watch_sort_desc(self, _old: bool, _new: bool) -> None:
        self.refresh_table()

    def watch_pinned(self, _old: frozenset, _new: frozenset) -> None:
        self.refresh_table()

    def action_toggle_pin(self) -> None:
        job = self.get_selected_job()
        if job is None:
            return
        pinned, is_pinned = toggle_job_pin(job.job_id)
        self.pinned = frozenset(pinned)
        self.app.notify(
            f"{'Pinned' if is_pinned else 'Unpinned'} job {job.job_id}", timeout=2
        )

    async def action_open_sort_menu(self) -> None:
        await self.app.action_open_sort_picker()

    async def action_toggle_sort_direction(self) -> None:
        await self.app.action_toggle_sort_direction()

    async def action_cycle_owner_filter(self) -> None:
        await self.app.action_cycle_owner_filter()

    async def action_open_details(self) -> None:
        await self.app.action_open_selected_job()

    async def action_open_output(self) -> None:
        await self.app.action_open_selected_job_output()

    def on_data_table_row_highlighted(self) -> None:
        self._update_title()


class NodesView(StableTable):
    BINDINGS = [("enter", "open_node_details", "Node details")]
    nodes: reactive[List[Node]] = reactive([], always_update=True)  # type: ignore
    # Probed CPU models, keyed by node name. Empty until someone asks for a
    # node's CPU in the details modal, so the column falls back to the layout
    # Slurm does report.
    cpu_info: reactive[Dict[str, Dict[str, str]]] = reactive({}, always_update=True)  # type: ignore
    _display_nodes: List[Node]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._display_nodes = []

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_columns(
            "NODE", "STATE", "CPUS(T)", "CPUS(alloc)", "CPUS(idle)",
            "MEM(total)", "MEM(resv)", "MEM(free)", "GPUs(total)", "CPU",
        )
        self.cpu_info = load_cpu_info()
        self.refresh_table()

    def _cpu_cell(self, node: Node) -> Text:
        """The CPU model if we have read it, otherwise the layout sinfo gives."""
        info = (self.cpu_info or {}).get(node.name) or {}
        described = describe_cpu(info)
        if described:
            return Text(described)
        return Text(cpu_topology(node) or "-", style="dim")

    def refresh_table(self) -> None:
        snapshot = self._view_snapshot()
        previous_row = self.cursor_row if self.cursor_row is not None else 0
        selected = self.get_selected_node()
        selected_name = selected.name if selected else None

        self.clear(columns=False)
        self._display_nodes = list(self.nodes)
        for n in self._display_nodes:
            state_style = "green" if n.state.startswith("idle") else "yellow"
            self.add_row(
                Text(n.name, style="cyan"),
                Text(n.state, style=state_style),
                n.cpus_total,
                n.cpus_alloc,
                n.cpus_idle,
                _format_mb_human(_parse_int(n.mem_total)),
                _format_mb_human(_parse_int(n.mem_reserved)),
                _format_mb_human(_parse_int(n.mem_free)),
                str(sum(_parse_gpu_inventory(n.gres).values())),
                self._cpu_cell(n),
            )

        self._set_panel_title(f"Nodes {len(self._display_nodes)} (Enter: details)")
        if not self._display_nodes:
            return

        row = min(max(previous_row, 0), len(self._display_nodes) - 1)
        if selected_name is not None:
            for idx, node in enumerate(self._display_nodes):
                if node.name == selected_name:
                    row = idx
                    break
        self._restore_view(snapshot, row)

    def get_selected_node(self) -> Optional[Node]:
        row = self.cursor_row
        if row is None or row < 0 or row >= len(self._display_nodes):
            return None
        return self._display_nodes[row]

    def watch_nodes(self, _old: List[Node], _new: List[Node]) -> None:
        self.refresh_table()

    def watch_cpu_info(self, _old: Dict[str, Dict[str, str]], _new: Dict[str, Dict[str, str]]) -> None:
        self.refresh_table()

    async def action_open_node_details(self) -> None:
        await self.app.action_open_selected_node()


class GpuStatusView(StableTable):
    BINDINGS = [("enter", "open_gpu_jobs", "GPU jobs")]
    stats: reactive[Dict[str, object]] = reactive({}, always_update=True)  # type: ignore
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
        snapshot = self._view_snapshot()
        previous_row = self.cursor_row if self.cursor_row is not None else 0
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

        self._set_panel_title("GPU status (Enter: jobs)")
        if not self._row_gpu_types:
            return

        row = min(max(previous_row, 0), len(self._row_gpu_types) - 1)
        if selected_gpu:
            for idx, gpu_type in enumerate(self._row_gpu_types):
                if gpu_type == selected_gpu:
                    row = idx
                    break
        self._restore_view(snapshot, row)

    def get_selected_gpu_type(self) -> Optional[str]:
        row = self.cursor_row
        if row is None or row < 0 or row >= len(self._row_gpu_types):
            return None
        return self._row_gpu_types[row]

    def watch_stats(self, _old: Dict[str, object], _new: Dict[str, object]) -> None:
        self.refresh_table()

    async def action_open_gpu_jobs(self) -> None:
        await self.app.action_open_selected_gpu_jobs()


class DiskUsageView(StableTable):
    disks: reactive[List[DiskUsage]] = reactive([], always_update=True)  # type: ignore
    _display_disks: List[DiskUsage]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._display_disks = []

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_columns("USAGE", "PATH", "TYPE", "SPACE")
        self.refresh_table()

    def refresh_table(self) -> None:
        snapshot = self._view_snapshot()
        previous_row = self.cursor_row if self.cursor_row is not None else 0
        selected = self.get_selected_disk()
        selected_mount = selected.mount if selected else None

        self.clear(columns=False)
        self._display_disks = list(self.disks)
        for d in self._display_disks:
            self.add_row(d.usage_percent, Text(d.mount, style="cyan"), d.fs_type, d.size)

        self._set_panel_title(f"Disks {len(self._display_disks)}")
        if not self._display_disks:
            return

        row = min(max(previous_row, 0), len(self._display_disks) - 1)
        if selected_mount is not None:
            for idx, disk in enumerate(self._display_disks):
                if disk.mount == selected_mount:
                    row = idx
                    break
        self._restore_view(snapshot, row)

    def get_selected_disk(self) -> Optional[DiskUsage]:
        row = self.cursor_row
        if row is None or row < 0 or row >= len(self._display_disks):
            return None
        return self._display_disks[row]

    def watch_disks(self, _old: List[DiskUsage], _new: List[DiskUsage]) -> None:
        self.refresh_table()


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
    """One job in full, with the actions that can be taken on it.

    The two actions that destroy running work - cancel and requeue - sit behind
    a modifier; everything a bare letter does either reads something or is
    undone by another letter next to it. `c` used to cancel, which was the worst
    possible place for it: `c` copies in every other panel of this app, so the
    muscle memory for "copy this row" landed on `scancel`. It copies here too
    now, and cancelling costs a deliberate ^X.

    ctrl+h, ctrl+i, ctrl+j and ctrl+m are Backspace, Tab, Enter and Return at
    the terminal level, which is why hold and requeue do not simply gain a
    modifier on the letter they already use.
    """

    BINDINGS = [
        ("enter", "dismiss", "Close"),
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("ctrl+x", "cancel_job", "Cancel"),
        ("ctrl+r", "requeue_job", "Requeue"),
        ("h", "hold_job", "Hold"),
        ("u", "release_job", "Release"),
        ("f", "manual_refresh", "Refresh"),
        ("r", "manual_refresh", "Refresh"),
        ("a", "toggle_auto_update", "Auto-update"),
        ("p", "toggle_pin", "Pin"),
        ("o", "open_output", "Output"),
        ("y", "copy_job", "Copy"),
        ("c", "copy_job", "Copy"),
    ]

    AUTO_UPDATE_INTERVAL = 3.0
    # Column indexes of the two toggles in the #job-actions row, whose labels
    # are rewritten in place when they flip.
    _AUTO_COLUMN = 5
    _PIN_COLUMN = 6
    _CLOSE_COLUMN = 9

    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job
        self.detail: Dict[str, str] = {}
        self.usage: Dict[str, str] = {}
        self.auto_update = bool(load_config().get("job_details_auto_update", False))
        self.is_pinned = job.job_id in set(load_pinned_jobs())
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
            # MaxRSS and the CPU time are in the bars below; this row keeps the
            # readings they leave out rather than printing the same number twice.
            rows.append(Text.assemble(
                ("Usage  ", dim), ("MaxVM ", dim), _human_mem(u.get("MaxVMSize", "")),
                "    ", ("steps ", dim), u.get("Steps") or "1",
                ("   live from sstat", dim),
            ))
        if tres:
            rows.append(Text.assemble(("TRES   ", dim), tres))
        return Text("\n").join(rows)

    def _usage_text(self) -> Text:
        """The request-versus-use bars: how much of the reservation is working."""
        rows = _usage_rows(
            job_usage_metrics(self.job, self.detail, self.usage),
            # A monochrome terminal gets the ASCII bars instead of a band of
            # colour it would render as an empty gap.
            color=bool(getattr(self.app.console, "color_system", "truecolor")),
        )
        return Text("\n").join(rows) if rows else Text("", style="dim")

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

    # The two labels that change with state are padded to their longest form.
    # A DataTable does not re-measure a column when a cell is updated, so a
    # label that grows would simply be cut off -- and a row that is built wide
    # (opening an already-pinned job, where "p Unpin" is two columns wider than
    # "p Pin") overflows the table and puts a horizontal scrollbar exactly
    # where its one row of actions was.
    _AUTO_WIDTH = len("a Auto: OFF")
    _PIN_WIDTH = len("p Unpin")

    def _auto_label(self) -> str:
        return f"a Auto: {'ON' if self.auto_update else 'OFF'}".ljust(self._AUTO_WIDTH)

    def compose(self) -> ComposeResult:
        # The compact block never scrolls; WkDir + Cmd share one horizontal
        # scroller so a single scrollbar covers just those two long lines.
        with Vertical(id="job-details-box"):
            yield Static(self._main_text(), id="job-details-body")
            yield Static(self._usage_text(), id="job-usage")
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
        actions.add_columns("", "", "", "", "", "", "", "", "", "")
        actions.add_row(
            "^X Cancel", "h Hold", "u Release", "^R Requeue",
            "f Refresh", self._auto_label(), self._pin_label(),
            "o Output", "y Copy", "Esc Close",
        )
        actions.cursor_background_priority = "css"
        actions.cursor_foreground_priority = "css"
        actions.move_cursor(row=0, column=self._CLOSE_COLUMN)
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
        self.query_one("#job-usage", Static).update(self._usage_text())
        self.query_one("#job-paths", Static).update(self._paths_text())

    def _alive(self) -> bool:
        """Whether this popup is still on screen and safe to draw into.

        `is_mounted` stays true for a moment after the popup is dismissed,
        while its widgets are already gone, so a refresh that was in flight
        when someone pressed Escape would query a widget that no longer
        exists. A failed worker takes the whole app down with it, so this is
        not a cosmetic check.
        """
        return self.is_mounted and self in self.app.screen_stack

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
        if not self._alive():
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

    def _pin_label(self) -> str:
        return ("p Unpin" if self.is_pinned else "p Pin").ljust(self._PIN_WIDTH)

    def action_toggle_pin(self) -> None:
        pinned, self.is_pinned = toggle_job_pin(self.job.job_id)
        actions = self.query_one("#job-actions", DataTable)
        actions.update_cell_at(Coordinate(0, self._PIN_COLUMN), self._pin_label())
        jobs_view = getattr(self.app, "jobs_view", None)
        if jobs_view is not None:
            jobs_view.pinned = frozenset(pinned)
        self._set_status(
            f"Status: job {self.job.job_id} {'pinned to the top' if self.is_pinned else 'unpinned'}"
        )

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
            self.action_toggle_pin()
            return
        if column == 7:
            self.action_open_output()
            return
        if column == 8:
            self.action_copy_job()
            return
        if column == self._CLOSE_COLUMN:
            self.dismiss()
            return

    async def _run_job_action(self, command: List[str], action_name: str) -> None:
        ok, output = await asyncio.to_thread(run_cmd_checked, command)
        if not self._alive():
            return
        status = f"Status: {action_name} {'OK' if ok else 'FAILED'} - {output}"
        self.query_one("#job-details-status", Static).update(status)
        if ok:
            await self.app.refresh_data()

    def action_open_output(self) -> None:
        """Tail this job's stdout/stderr in a popup of its own."""
        self.app.push_screen(JobOutputModal(self.job, self.detail))

    def action_copy_job(self) -> None:
        """Copy picker for the fields that are awkward to retype (workdir, command)."""
        j = self.job
        command = self._field("Command")
        if command == "-":
            command = self._field("SubmitLine")
        fields = [
            ("JOBID", j.job_id),
            ("USER", self._field("UserId", j.user)),
            ("NAME", self._field("JobName", j.name)),
            ("STATE", self._field("JobState", j.state)),
            ("PART", self._field("Partition", j.partition)),
            ("NODELIST", self._field("NodeList", j.node_list or "-")),
            ("CPUS", self._field("NumCPUs", j.ncpus)),
            ("MEM", self._field("MinMemoryNode", j.mem)),
            ("TIME", self._field("RunTime", j.time_used)),
            ("WORKDIR", self._field("WorkDir")),
            ("COMMAND", command),
            ("STDOUT", self._field("StdOut")),
            ("STDERR", self._field("StdErr")),
        ]
        self.app.push_screen(CopyModal(f"job {j.job_id}", fields))

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


class CopyModal(ModalScreen[None]):
    """Field picker for copying a table row to the clipboard.

    DataTable disables Textual's drag-to-select (it owns the mouse for its own
    cursor), so copying out of a panel needs an explicit route: this popup.
    Clipboard writes go out as OSC 52, which most terminals accept over SSH;
    where they do not, the echoed value below is a plain Static and can be
    mouse-selected and copied with ctrl+c.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("enter", "copy_selected", "Copy field"),
        ("a", "copy_all", "Copy whole row"),
    ]

    def __init__(self, subject: str, fields: "List[tuple[str, str]]") -> None:
        super().__init__()
        self.subject = subject
        self.fields = fields

    def compose(self) -> ComposeResult:
        yield Static(
            f"Copy from {self.subject} - Enter or click copies one field, a copies the whole row",
            id="copy-help",
        )
        yield DataTable(id="copy-table")
        yield Static("", id="copy-status")

    def on_mount(self) -> None:
        table = self.query_one("#copy-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("FIELD", "VALUE")
        for name, value in self.fields:
            table.add_row(name, value)
        if not self.fields:
            table.add_row("-", "nothing to copy")
        table.move_cursor(row=0)
        self.set_focus(table)
        self._set_status("Pick a field, then Enter. Text shown here can also be mouse-selected + ctrl+c.")

    def _set_status(self, text: str) -> None:
        self.query_one("#copy-status", Static).update(text)

    def _copy(self, text: str, label: str) -> None:
        self.app.copy_to_clipboard(text)
        self._set_status(f"Copied {label}:\n{text}")
        self.app.notify(f"Copied {label}")

    def action_copy_selected(self) -> None:
        table = self.query_one("#copy-table", DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self.fields):
            return
        name, value = self.fields[row]
        self._copy(value, name)

    def action_copy_all(self) -> None:
        if not self.fields:
            return
        self._copy("\t".join(value for _, value in self.fields), "whole row")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "copy-table":
            return
        self.action_copy_selected()


class JobOutputModal(ModalScreen[None]):
    """`tail -f` for the files a job's stdout and stderr are going to.

    Both at once by default, one above the other: a job that has gone wrong
    usually says so in stderr while stdout keeps printing, and having to toggle
    between them is exactly the moment you miss the line that mattered. Batch
    scripts that send both streams to one file get one pane, not two copies.

    Slurm stores the paths and nothing else, so this is plain file reading from
    the login node - which is also its one limitation: a job writing to scratch
    that is local to the compute node leaves a path nothing here can open.
    Only the last stretch of each file is read (`read_file_tail`), so following
    a job that has been printing for a week costs the same as following one
    that started a minute ago.
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("o", "cycle_view", "View"),
        ("1", "view_stdout", "stdout"),
        ("2", "view_stderr", "stderr"),
        ("3", "view_both", "both"),
        ("f", "manual_refresh", "Refresh"),
        ("r", "manual_refresh", "Refresh"),
        ("a", "toggle_follow", "Follow"),
        ("plus", "more_lines", "More"),
        ("minus", "fewer_lines", "Fewer"),
        ("y", "copy_output", "Copy"),
        ("home", "to_top", "Top"),
        ("end", "to_bottom", "Bottom"),
    ]

    FOLLOW_INTERVAL = 2.0
    # Both first: the view you want unless you know which stream to look in.
    MODES = ("both", "stdout", "stderr")
    STREAMS = ("stdout", "stderr")
    # How much scrollback one press of + or - moves through.
    LINE_STEPS = [50, 200, 1000, 5000]
    # The three views are three cells of the action row, in this order, so the
    # choice is visible rather than something you have to cycle to discover.
    VIEW_COLUMNS = ("stdout", "stderr", "both")
    _FOLLOW_COLUMN = 4

    def __init__(self, job: Job, detail: "Optional[Dict[str, str]]" = None,
                 mode: str = "both") -> None:
        super().__init__()
        self.job = job
        self.detail: Dict[str, str] = dict(detail or {})
        self.mode = mode if mode in self.MODES else "both"
        self.lines = OUTPUT_TAIL_LINES
        self.streams: Dict[str, Dict[str, object]] = {}
        self.tails: Dict[str, Dict[str, object]] = {}
        self.follow = bool(load_config().get("job_output_follow", True))
        self._follow_timer: Optional[Timer] = None

    # -- what is on screen -------------------------------------------------

    def _merged(self) -> bool:
        """True when the job sent both streams to the same file."""
        return bool(self.streams.get("stderr", {}).get("merged"))

    def _visible_streams(self) -> "List[str]":
        if self._merged():
            # One file, so one pane - a second copy of it would say nothing.
            return ["stdout"]
        if self.mode == "both":
            return list(self.STREAMS)
        return [self.mode]

    def _file(self, stream: str) -> Dict[str, object]:
        return self.streams.get(stream, {})

    def _path(self, stream: str) -> str:
        return str(self._file(stream).get("path", ""))

    # -- rendering ---------------------------------------------------------

    def _head_text(self) -> Text:
        dim = "dim"
        state = (self.detail.get("JobState", "") or self.job.state).upper()
        rows = [Text.assemble(
            (f"{self.job.name} ", "bold"),
            (state, self._head_state_style(state)),
            ("    last ", dim), f"{self.lines}", (" lines", dim),
            ("    view ", dim), "one file (merged)" if self._merged() else self.mode,
            ("    follow ", dim), ("ON" if self.follow else "OFF", "green" if self.follow else dim),
        )]
        for stream in self._visible_streams():
            info = self._file(stream)
            size = int(info.get("size", 0) or 0)
            modified = float(info.get("modified", 0.0) or 0.0)
            label = "both" if self._merged() else stream
            rows.append(Text.assemble(
                (f"{label:<7}", dim),
                (str(info.get("path", "")) or "-", "cyan"),
                ("  " + _human_bytes(size), dim) if info.get("exists") else ("", dim),
                ("  written " + time.strftime("%H:%M:%S", time.localtime(modified)), dim)
                if modified else ("", dim),
            ))
        return Text("\n").join(rows)

    @staticmethod
    def _head_state_style(state: str) -> str:
        if state.startswith("R"):
            return "bold green"
        if state.startswith("P"):
            return "bold yellow"
        return "bold"

    def _body_text(self, stream: str) -> Text:
        info = self._file(stream)
        tail = self.tails.get(stream, {})
        error = str(info.get("error", "")) or str(tail.get("error", ""))
        if error:
            return Text(f"Nothing to show: {error}", style="yellow")
        if not self.tails:
            return Text("Reading...", style="dim italic")
        text = str(tail.get("text", ""))
        if not text.strip():
            return Text("The file is there but still empty.", style="dim italic")
        # no_wrap so a wide log keeps its columns; the pane scrolls sideways
        # instead of reflowing the lines.
        return Text(text, no_wrap=True)

    def compose(self) -> ComposeResult:
        with Vertical(id="job-output-box"):
            yield Static(self._head_text(), id="job-output-head")
            for stream in self.STREAMS:
                with ScrollableContainer(id=f"pane-{stream}", classes="job-output-pane"):
                    yield Static("", id=f"body-{stream}")
        yield DataTable(id="job-output-actions")
        yield Static("Status: reading", id="job-output-status")

    def on_mount(self) -> None:
        for stream in self.STREAMS:
            pane = self.query_one(f"#pane-{stream}", ScrollableContainer)
            pane.border_title = stream
        actions = self.query_one("#job-output-actions", DataTable)
        actions.cursor_type = "cell"
        actions.zebra_stripes = False
        actions.show_header = False
        actions.add_columns("", "", "", "", "", "", "", "")
        actions.add_row(
            *[self._view_cell(mode) for mode in self.VIEW_COLUMNS],
            "f Refresh", self._follow_label(),
            "+/- Lines", "y Copy", "Esc Close",
        )
        actions.cursor_background_priority = "css"
        actions.cursor_foreground_priority = "css"
        actions.move_cursor(row=0, column=7)
        self.set_focus(actions)
        self.query_one("#job-output-box", Vertical).border_title = f"Output of job {self.job.job_id}"
        self._follow_timer = self.set_interval(
            self.FOLLOW_INTERVAL, self._follow_tick, pause=not self.follow
        )
        self._redraw(keep_position=False)
        self._request_reload("opened", resolve=not self.detail)

    def _view_cell(self, mode: str) -> Text:
        """One of the three view buttons, marked when it is the one in force.

        A job that sent both streams to one file has only one view; the cells
        stay on screen, dimmed, rather than vanishing and shifting the row.
        """
        hotkey = str(self.VIEW_COLUMNS.index(mode) + 1)
        if self._merged():
            return Text(f"{hotkey} {mode}", style="dim")
        # Bold and underlined for the one in force, dim for the others: an
        # attribute rather than a marker glyph, for the same reason the usage
        # bars are drawn in colour - and it keeps every cell the same width,
        # so switching views cannot reflow the row.
        chosen = mode == self.mode
        return Text(f"{hotkey} {mode}", style="bold underline" if chosen else "dim")

    _FOLLOW_WIDTH = len("a Follow: OFF")

    def _follow_label(self) -> str:
        return f"a Follow: {'ON' if self.follow else 'OFF'}".ljust(self._FOLLOW_WIDTH)

    def _set_status(self, text: str) -> None:
        self.query_one("#job-output-status", Static).update(text)

    def _at_bottom(self, pane: ScrollableContainer) -> bool:
        """Whether a pane is parked at the end, i.e. following the writer."""
        return pane.max_scroll_y <= 0 or pane.scroll_offset.y >= pane.max_scroll_y - 1

    def _redraw(self, keep_position: bool) -> None:
        visible = self._visible_streams()
        for stream in self.STREAMS:
            pane = self.query_one(f"#pane-{stream}", ScrollableContainer)
            pane.display = stream in visible
            if stream not in visible:
                continue
            # A new tail is only scrolled to the end when the reader was
            # already there; someone reading further up keeps their place.
            stick = not keep_position or self._at_bottom(pane)
            pane.border_title = "stdout + stderr" if self._merged() else stream
            self.query_one(f"#body-{stream}", Static).update(self._body_text(stream))
            if stick:
                self.call_after_refresh(pane.scroll_end, animate=False)
        self.query_one("#job-output-head", Static).update(self._head_text())
        actions = self.query_one("#job-output-actions", DataTable)
        for column, mode in enumerate(self.VIEW_COLUMNS):
            actions.update_cell_at(Coordinate(0, column), self._view_cell(mode))

    # -- loading -----------------------------------------------------------

    def _collect(self, resolve: bool) -> "tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]":
        detail = fetch_job_detail(self.job.job_id) if resolve else self.detail
        if resolve:
            self.detail = detail
        streams = job_output_paths(detail, self.job)
        merged = bool(streams.get("stderr", {}).get("merged"))
        wanted = ["stdout"] if merged else (list(self.STREAMS) if self.mode == "both" else [self.mode])
        tails = {
            name: read_file_tail(str(streams.get(name, {}).get("path", "")), lines=self.lines)
            for name in wanted
        }
        return streams, tails

    def _alive(self) -> bool:
        """Whether this popup is still on screen and safe to draw into.

        `is_mounted` stays true for a moment after the popup is dismissed,
        while its widgets are already gone, so a refresh that was in flight
        when someone pressed Escape would query a widget that no longer
        exists. A failed worker takes the whole app down with it, so this is
        not a cosmetic check.
        """
        return self.is_mounted and self in self.app.screen_stack

    def _request_reload(self, source: str, resolve: bool = False) -> None:
        self.run_worker(
            self._do_reload(source, resolve), group="job-output", exclusive=True
        )

    async def _do_reload(self, source: str, resolve: bool) -> None:
        streams, tails = await asyncio.to_thread(self._collect, resolve)
        if not self._alive():
            return
        first = not self.tails
        self.streams = streams
        self.tails = tails
        self._redraw(keep_position=not first)
        stamp = time.strftime("%H:%M:%S")
        counts = ", ".join(
            f"{name} {int(tail.get('line_count', 0) or 0)} line(s)"
            + ("+" if tail.get("truncated") else "")
            for name, tail in tails.items()
        )
        self._set_status(f"Status: {counts or 'nothing to read'} - updated ({source}) at {stamp}")

    def _follow_tick(self) -> None:
        self._request_reload("follow")

    # -- actions -----------------------------------------------------------

    def action_manual_refresh(self) -> None:
        self._request_reload("manual", resolve=True)

    def action_set_view(self, mode: str) -> None:
        """Show one stream, or both - the choice the action row offers."""
        if self._merged():
            self._set_status("Status: this job sends both streams to one file")
            return
        if mode == self.mode:
            return
        self.mode = mode
        self._redraw(keep_position=False)
        self._request_reload(f"view {self.mode}")

    def action_view_stdout(self) -> None:
        self.action_set_view("stdout")

    def action_view_stderr(self) -> None:
        self.action_set_view("stderr")

    def action_view_both(self) -> None:
        self.action_set_view("both")

    def action_cycle_view(self) -> None:
        """`o` again steps through the same three, for one-key switching."""
        if self._merged():
            self._set_status("Status: this job sends both streams to one file")
            return
        self.action_set_view(self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)])

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        config = load_config()
        config["job_output_follow"] = self.follow
        save_config(config)
        actions = self.query_one("#job-output-actions", DataTable)
        actions.update_cell_at(Coordinate(0, self._FOLLOW_COLUMN), self._follow_label())
        if self._follow_timer is not None:
            if self.follow:
                self._follow_timer.resume()
            else:
                self._follow_timer.pause()
        if self.follow:
            self._request_reload("follow")
        else:
            self._set_status("Status: follow OFF")

    def _set_lines(self, lines: int) -> None:
        if lines == self.lines:
            return
        self.lines = lines
        self._request_reload(f"{lines} lines")

    def action_more_lines(self) -> None:
        bigger = [n for n in self.LINE_STEPS if n > self.lines]
        self._set_lines(bigger[0] if bigger else self.lines)

    def action_fewer_lines(self) -> None:
        smaller = [n for n in self.LINE_STEPS if n < self.lines]
        self._set_lines(smaller[-1] if smaller else self.lines)

    def _focused_pane(self) -> ScrollableContainer:
        visible = self._visible_streams()
        for stream in visible:
            pane = self.query_one(f"#pane-{stream}", ScrollableContainer)
            if self.focused is pane:
                return pane
        return self.query_one(f"#pane-{visible[0]}", ScrollableContainer)

    def action_to_top(self) -> None:
        self._focused_pane().scroll_home(animate=False)

    def action_to_bottom(self) -> None:
        self._focused_pane().scroll_end(animate=False)

    def action_copy_output(self) -> None:
        fields: List[tuple[str, str]] = []
        for stream in self.STREAMS:
            path = self._path(stream)
            fields.append((f"{stream.upper()} PATH", path or "-"))
        first = self._visible_streams()[0]
        fields.append(("TAIL COMMAND", f"tail -f {self._path(first)}" if self._path(first) else "-"))
        for stream in self._visible_streams():
            fields.append((
                f"{stream.upper()} TEXT",
                str(self.tails.get(stream, {}).get("text", "")) or "-",
            ))
        self.app.push_screen(CopyModal(f"output of job {self.job.job_id}", fields))

    async def _run_action_by_column(self, column: int) -> None:
        if column < len(self.VIEW_COLUMNS):
            self.action_set_view(self.VIEW_COLUMNS[column])
        elif column == 3:
            self.action_manual_refresh()
        elif column == self._FOLLOW_COLUMN:
            self.action_toggle_follow()
        elif column == 5:
            self.action_more_lines()
        elif column == 6:
            self.action_copy_output()
        elif column == 7:
            self.dismiss()

    async def on_key(self, event: Key) -> None:
        if event.key != "enter":
            return
        actions = self.query_one("#job-output-actions", DataTable)
        if self.focused is not actions:
            return
        event.stop()
        if actions.cursor_column is not None:
            await self._run_action_by_column(actions.cursor_column)

    async def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        if event.data_table.id != "job-output-actions":
            return
        await self._run_action_by_column(event.coordinate.column)


class JobSearchModal(ModalScreen[None]):
    """Free-text filter over the jobs table, applied live as you type."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, current: str) -> None:
        super().__init__()
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="search-box"):
            yield Static("Filter jobs - every term must match somewhere in the row", id="search-help")
            yield Input(value=self.current, placeholder="e.g. gpu pending train", id="search-input")
            yield Static("Enter or Esc closes - an empty box shows all jobs again", id="search-hint")

    def on_mount(self) -> None:
        box = self.query_one("#search-box", Vertical)
        box.border_title = "Search jobs"
        field = self.query_one("#search-input", Input)
        field.cursor_position = len(field.value)
        self.set_focus(field)

    def _apply(self, value: str) -> None:
        app = self.app
        if isinstance(app, SlurmHtop):
            app.jobs_view.search_filter = value.strip()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._apply(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._apply(event.value)
        self.dismiss()


class NodeDetailsModal(ModalScreen[None]):
    """`scontrol show node` detail plus the jobs Slurm placed on that node."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("f", "manual_refresh", "Refresh"),
        ("p", "probe_cpu", "Read CPU"),
        ("c", "copy_node", "Copy"),
    ]

    _EMPTY_FIELDS = {"", "(null)", "N/A", "None", "Unknown"}

    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node
        self.detail: Dict[str, str] = {}
        self.node_jobs: List[Job] = []
        self.cpu: Dict[str, str] = load_cpu_info().get(node.name, {})
        self.cpu_error = ""
        self.probing = False

    @staticmethod
    def _state_style(state: str) -> str:
        s = state.lower()
        if s.startswith("idle"):
            return "bold green"
        if s.startswith(("alloc", "mix", "comp", "resv")):
            return "bold yellow"
        if s.startswith(("down", "drain", "drng", "fail", "err", "inval", "unk", "maint")):
            return "bold red"
        return "bold"

    def _field(self, key: str, default: str = "-") -> str:
        value = (self.detail.get(key) or "").strip()
        return default if value in self._EMPTY_FIELDS else value

    def _main_text(self) -> Text:
        n = self.node
        gd = self._field
        dim = "dim"
        state = gd("State", n.state)
        running = sum(1 for j in self.node_jobs if j.state.upper().startswith("R"))
        pending = len(self.node_jobs) - running

        rows = [
            Text.assemble(
                ("Node   ", dim), (n.name, "bold"),
                "    ", (state, self._state_style(state)),
                "    ", ("Reason ", dim), gd("Reason"),
            ),
            Text.assemble(
                ("CPUs   ", dim), f"{gd('CPUAlloc', n.cpus_alloc)} alloc / {gd('CPUTot', n.cpus_total)} total",
                "    ", ("idle ", dim), n.cpus_idle or "-",
                "    ", ("load ", dim), gd("CPULoad"),
            ),
            self._cpu_text(),
            Text.assemble(
                ("Memory ", dim), f"{_human_mem(gd('AllocMem'))} alloc / {_human_mem(gd('RealMemory', n.mem_total))} total",
                "    ", ("free ", dim), _human_mem(gd("FreeMem", n.mem_free)),
                "    ", ("tmp ", dim), _human_mem(gd("TmpDisk")),
            ),
            Text.assemble(
                ("GRES   ", dim), gd("Gres", n.gres or "-"),
                "    ", ("used ", dim), gd("GresUsed"),
            ),
            # Sockets and threads used to live here; the CPU row above spells
            # out the whole layout, so repeating two thirds of it only crowds
            # the line the partition list needs.
            Text.assemble(
                ("Part   ", dim), gd("Partitions"),
                "    ", ("weight ", dim), gd("Weight"),
                "    ", ("boards ", dim), gd("Boards"),
            ),
            Text.assemble(
                ("Feat   ", dim), gd("AvailableFeatures"),
                "    ", ("active ", dim), gd("ActiveFeatures"),
            ),
            Text.assemble(
                ("Uptime ", dim), ("boot ", dim), _short_time(self.detail.get("BootTime")),
                "    ", ("slurmd ", dim), _short_time(self.detail.get("SlurmdStartTime")),
                "    ", ("ver ", dim), gd("Version"),
            ),
            Text.assemble(
                ("Jobs   ", dim), f"{len(self.node_jobs)} here",
                "    ", ("running ", dim), str(running),
                "    ", ("pending ", dim), str(pending),
            ),
        ]
        alloc_tres = gd("AllocTRES")
        if alloc_tres != "-":
            rows.append(Text.assemble(("TRES   ", dim), alloc_tres))
        return Text("\n").join(rows)

    def _cpu_counts(self) -> Dict[str, int]:
        n = self.node
        # sinfo's %z is the fallback: the modal paints once before the
        # scontrol lookup it fires on open has come back.
        return cpu_counts(
            self._field("Sockets", n.sockets),
            self._field("CoresPerSocket", n.cores_per_socket),
            self._field("ThreadsPerCore", n.threads_per_core),
            self._field("CPUTot", n.cpus_total),
        )

    def _cpu_text(self) -> Text:
        """How many processors of what, at what speed.

        Slurm answers the first half on every refresh - sockets, cores, threads
        - but carries no model name and no clock at all. Those exist only on
        the node, so the second half stays an invitation to go and read them
        until someone presses p.
        """
        dim = "dim"
        gd = self._field
        counts = self._cpu_counts()
        layout = (
            f"{counts['processors']} sockets x {counts['cores_per_processor']} cores"
            f" x {counts['threads_per_core']} threads"
        )

        parts = [
            ("CPU    ", dim), (f"{counts['logical']} CPUs", "bold"),
            "    ", (f"{counts['cores']} cores" if counts["cores"] else "", dim),
            "    ", layout,
            "    ", ("arch ", dim), gd("Arch"),
        ]
        if self.probing:
            parts += ["\n", ("Model  ", dim), ("reading the node...", "italic")]
        elif self.cpu:
            source = self.cpu.get("source", "")
            parts += [
                "\n", ("Model  ", dim),
                (describe_cpu_count(counts, self.cpu) or "-", "bold"),
                "    ", (f"{counts['cores_per_processor']} cores each" if counts["cores_per_processor"] else "", dim),
                "    ", (f"(read over {source})" if source else "", dim),
            ]
            speeds = describe_cpu_speeds(self.cpu)
            if speeds:
                # Every clock we have, each labelled: a boost ceiling and an
                # idling governor both look like "the speed" on their own.
                parts.append("\n")
                parts.append(("Speed  ", dim))
                for index, (label, value) in enumerate(speeds):
                    if index:
                        parts.append("    ")
                    parts += [(f"{label} ", dim), (value, "bold" if label == "nominal" else None)]
        elif self.cpu_error:
            parts += ["\n", ("Model  ", dim), ("could not read it: ", dim), self.cpu_error]
        else:
            parts += ["\n", ("Model  ", dim), ("unknown - press p to read it from the node", "italic")]
        return Text.assemble(*parts)

    def compose(self) -> ComposeResult:
        with Vertical(id="node-details-box"):
            yield Static(self._main_text(), id="node-details-body")
        yield DataTable(id="node-jobs")
        yield Static("Status: loading...", id="node-details-status")

    def on_mount(self) -> None:
        table = self.query_one("#node-jobs", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("JOBID", "USER", "STATE", "PART", "NAME", "CPUS", "GPUS", "MEM", "TIME")
        table.border_title = "Jobs on this node (Enter: job details)"
        self.set_focus(table)
        self.query_one("#node-details-box", Vertical).border_title = f"Node {self.node.name}"
        self._request_refresh("opened")

    def _set_status(self, text: str) -> None:
        self.query_one("#node-details-status", Static).update(text)

    def _refresh_body(self) -> None:
        self.query_one("#node-details-body", Static).update(self._main_text())
        table = self.query_one("#node-jobs", DataTable)
        selected = self._selected_job()
        selected_id = selected.job_id if selected else None
        table.clear(columns=False)
        for j in self.node_jobs:
            style = None
            if j.state.upper().startswith("R"):
                style = "green"
            elif j.state.upper().startswith("P"):
                style = "yellow"
            table.add_row(
                j.job_id, j.user, Text(j.state, style=style), j.partition, j.name,
                j.ncpus, str(_parse_gpu_count(j.gpus)), j.mem, j.time_used,
            )
        if not self.node_jobs:
            table.add_row("-", "-", "-", "-", "no jobs on this node", "-", "-", "-", "-")
            return
        row = 0
        if selected_id is not None:
            for idx, job in enumerate(self.node_jobs):
                if job.job_id == selected_id:
                    row = idx
                    break
        table.move_cursor(row=row)

    def _selected_job(self) -> Optional[Job]:
        table = self.query_one("#node-jobs", DataTable)
        row = table.cursor_row
        if row is None or row < 0 or row >= len(self.node_jobs):
            return None
        return self.node_jobs[row]

    def _alive(self) -> bool:
        """Still on screen? See the note on JobDetailsModal._alive."""
        return self.is_mounted and self in self.app.screen_stack

    def _request_refresh(self, source: str) -> None:
        self.run_worker(
            self._do_refresh(source), group="node-detail-refresh", exclusive=True
        )

    async def _do_refresh(self, source: str) -> None:
        detail, jobs = await asyncio.to_thread(collect_node_info, self.node.name)
        if not self._alive():
            return
        stamp = time.strftime("%H:%M:%S")
        self.detail = detail
        self.node_jobs = jobs
        self._refresh_body()
        if not detail:
            self._set_status(f"Status: scontrol returned nothing for {self.node.name} - checked {stamp}")
            return
        self._set_status(
            f"Status: updated ({source}) at {stamp} - Enter opens a job, f refreshes, "
            "p reads the CPU, c copies"
        )

    def action_manual_refresh(self) -> None:
        self._request_refresh("manual")

    def action_probe_cpu(self) -> None:
        """Read the CPU model off the node itself, once, on request.

        Never done as part of the ordinary refresh: it either opens an ssh
        connection or submits a one-second job, and doing that for every node
        on every tick would be a poor neighbour on a shared cluster.
        """
        if self.probing:
            return
        self.probing = True
        self.cpu_error = ""
        self._refresh_body()
        self._set_status(f"Status: reading {self.node.name}'s CPU (ssh, then a one-second job)...")
        self.run_worker(self._do_probe_cpu(), group="node-cpu-probe", exclusive=True)

    async def _do_probe_cpu(self) -> None:
        info, error = await asyncio.to_thread(probe_node_cpu, self.node.name)
        if not self._alive():
            return
        self.probing = False
        self.cpu = info
        self.cpu_error = error
        self._refresh_body()
        nodes_view = getattr(self.app, "nodes_view", None)
        if nodes_view is not None:
            nodes_view.cpu_info = load_cpu_info()
        if info:
            self._set_status(f"Status: {self.node.name} is {describe_cpu(info)}")
        else:
            self._set_status(f"Status: could not read the CPU - {error}")

    def action_copy_node(self) -> None:
        n = self.node
        fields = [
            ("NODE", n.name),
            ("STATE", self._field("State", n.state)),
            ("CPUS", f"{self._field('CPUAlloc', n.cpus_alloc)}/{self._field('CPUTot', n.cpus_total)}"),
            ("MEM(total)", self._field("RealMemory", n.mem_total)),
            ("MEM(free)", self._field("FreeMem", n.mem_free)),
            ("GRES", self._field("Gres", n.gres or "-")),
            ("GRES(used)", self._field("GresUsed")),
            ("PARTITIONS", self._field("Partitions")),
            ("CPU(count)", f"{self._cpu_counts()['logical']} CPUs"),
            ("CPU(layout)", cpu_topology(n) or "-"),
            ("CPU(model)", self.cpu.get("model", "-") or "-"),
            ("CPU(speed)", "  ".join(f"{k} {v}" for k, v in describe_cpu_speeds(self.cpu)) or "-"),
        ]
        self.app.push_screen(CopyModal(f"node {n.name}", fields))

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "node-jobs":
            return
        job = self._selected_job()
        if job is None:
            return
        await self.app.push_screen(JobDetailsModal(job))


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
    #jobs, #nodes, #gpu-status, #disk-usage {
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
    /* NodesView/DiskUsageView are DataTables: height:auto would clip rows
       and collapse virtual_size (see the GPU note below), so they fill the
       viewport and scroll their own rows. */
    #nodes { height: 1fr; }
    #disk-usage { height: 1fr; }
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
    NodeDetailsModal {
        align: center middle;
    }
    CopyModal {
        align: center middle;
    }
    JobSearchModal {
        align: center middle;
    }
    JobOutputModal {
        align: center middle;
    }
    #job-details-box {
        width: 110;
        max-width: 100%;
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
    #job-usage {
        width: 1fr;
        height: auto;
        margin-top: 1;
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
        width: 110;
        max-width: 100%;
        border: round $boost;
        padding: 0 2;
        background: $surface;
    }
    #job-actions {
        width: 110;
        max-width: 100%;
        height: 3;
        border: round $boost;
        background: $surface;
        /* The row of actions is exactly one line tall, so a horizontal
           scrollbar does not sit under it - it replaces it. Clip a row too
           wide for the terminal rather than hide it. */
        overflow-x: hidden;
    }
    #job-actions:focus {
        border: round $accent;
    }
    #job-actions > .datatable--cursor {
        background: $accent 60%;
        color: $text;
        text-style: bold;
    }
    #job-output-box {
        width: 90%;
        height: 1fr;
        border: round $accent;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }
    #job-output-head {
        width: 1fr;
        height: auto;
    }
    /* Both panes share the box evenly. Textual gives a hidden widget no
       space, so the single-stream views need no separate rule. */
    .job-output-pane {
        width: 1fr;
        height: 1fr;
        overflow-x: auto;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        border: round $boost;
        border-title-color: $text-muted;
        margin-top: 1;
    }
    #pane-stdout {
        height: 2fr;
    }
    #pane-stderr {
        height: 1fr;
    }
    .job-output-pane:focus {
        border: round $accent;
        border-title-color: $accent;
    }
    .job-output-pane > Static {
        width: auto;
        height: auto;
    }
    #job-output-actions {
        width: 90%;
        height: 3;
        border: round $boost;
        background: $surface;
        /* Same reason as #job-actions: one line tall, so no scrollbar. */
        overflow-x: hidden;
    }
    #job-output-actions:focus {
        border: round $accent;
    }
    #job-output-actions > .datatable--cursor {
        background: $accent 60%;
        color: $text;
        text-style: bold;
    }
    #job-output-status {
        width: 90%;
        border: round $boost;
        padding: 0 2;
        background: $surface;
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
    #node-details-box {
        width: 104;
        height: auto;
        max-height: 60%;
        border: round $accent;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }
    #node-details-body {
        width: 1fr;
        height: auto;
    }
    #node-jobs {
        width: 104;
        height: 12;
        border: round $boost;
        background: $surface;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
    }
    #node-jobs:focus {
        border: round $accent;
    }
    #node-details-status {
        width: 104;
        border: round $boost;
        padding: 0 2;
        background: $surface;
    }
    #copy-help {
        width: 90;
        border: round $panel;
        padding: 0 1;
    }
    #copy-table {
        width: 90;
        height: 14;
        border: round $accent;
        background: $surface;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
    }
    #copy-status {
        width: 90;
        height: auto;
        border: round $boost;
        padding: 0 1;
        background: $surface;
    }
    #search-box {
        width: 72;
        height: auto;
        border: round $accent;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }
    #search-help, #search-hint {
        width: 1fr;
        height: auto;
        color: $text-muted;
    }
    #search-input {
        width: 1fr;
        margin: 1 0;
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
        ("slash", "open_search", "Find"),
        ("c", "copy_selection", "Copy"),
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
        # Jobs/Nodes/GPU/Disks write their own border titles (they carry live
        # counts and the active filters); only the static panel needs one here.
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
            # Cheap: re-read only when the file moved, so a CPU probe made in
            # the editor extension shows up here on the next tick.
            self.nodes_view.cpu_info = load_cpu_info()
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

    async def action_open_selected_job_output(self) -> None:
        selected_job = self.jobs_view.get_selected_job()
        if not selected_job:
            self.notify("No job selected")
            return
        await self.push_screen(JobOutputModal(selected_job))

    async def action_open_search(self) -> None:
        await self.push_screen(JobSearchModal(self.jobs_view.search_filter))

    async def action_open_selected_node(self) -> None:
        node = self.nodes_view.get_selected_node()
        if not node:
            self.notify("Select a node row first")
            return
        await self.push_screen(NodeDetailsModal(node))

    def _summary_fields(self) -> "List[tuple[str, str]]":
        fields: List[tuple[str, str]] = []
        summary = self.summary_bar.summary or {}
        for bucket in ("all", "me", "others"):
            for state in ("running", "pending"):
                data = summary.get(bucket, {}).get(state, {})
                fields.append((
                    f"{bucket} {state}",
                    f"{data.get('jobs', 0)} jobs / {data.get('gpus', 0)} GPUs / "
                    f"{data.get('cpus', 0)} CPUs / {_format_mb_human(data.get('mem_mb', 0))}",
                ))
        return fields

    def _copy_fields_for_panel(self) -> "tuple[str, List[tuple[str, str]]]":
        panel = self._focused_panel()
        if panel == "jobs":
            job = self.jobs_view.get_selected_job()
            fields = self.jobs_view.row_fields()
            if job is not None:
                # Columns the table has no room for, but that you often want.
                fields += [("NODELIST", job.node_list), ("TRES", job.gpus)]
            return (f"job {job.job_id}" if job else "jobs"), fields
        if panel == "nodes":
            node = self.nodes_view.get_selected_node()
            fields = self.nodes_view.row_fields()
            if node is not None:
                fields.append(("GRES", node.gres))
            return (f"node {node.name}" if node else "nodes"), fields
        if panel == "gpu":
            return "GPU status", self.gpu_status_view.row_fields()
        if panel == "disk":
            return "disks", self.disk_usage_view.row_fields()
        return "job statistics", self._summary_fields()

    def action_copy_selection(self) -> None:
        if len(self.screen_stack) > 1:
            self.notify("Copy works on the main panels - close this popup first")
            return
        subject, fields = self._copy_fields_for_panel()
        if not fields:
            self.notify("Nothing selected to copy")
            return
        self.push_screen(CopyModal(subject, fields))

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
