# Slurm htop-style TUI

A terminal dashboard for Slurm clusters, inspired by `htop`.

## Project Links

- GitHub: [hforoughmand/slurm-monitor-top](https://github.com/hforoughmand/slurm-monitor-top)
- Issues: [github.com/hforoughmand/slurm-monitor-top/issues](https://github.com/hforoughmand/slurm-monitor-top/issues)

## Screenshot

![slurm-top screenshot](https://github.com/hforoughmand/slurm-monitor-top/blob/main/assets/screenshot-01.png)

## Features

- Live jobs table with interactive sort, filters and free-text search
- Live node/server table (CPU and memory totals, reserved, and free), with a
  per-node details popup (`Enter`) listing the jobs on that node
- Copy popup (`c`) for the selected row of any panel, so values can be pasted elsewhere
- Live disk table (`df -h`) with usage, mount path, type, and size
- Job statistics split by:
  - all users
  - your jobs
  - other users
- GPU status under nodes:
  - number of GPU types
  - total GPUs
  - active GPUs (running jobs)
  - reserved GPUs (pending jobs)
- Auto refresh every 3 seconds that keeps your scroll position and selection
- JSON output (`slurm-top --json`) for other front ends
- A [VS Code extension](vscode-extension/) with the same panels, in the sidebar
  or a full editor tab

## Requirements

- Python 3.9+
- Slurm CLI commands available in PATH:
  - `squeue`
  - `sinfo`
- Python packages: see `requirements.txt` (main dependencies: `textual`, `rich`)

## Install

Install from PyPI:

```bash
pip install slurm-monitor-top
```

For local development (editable install), create and activate a [virtual environment](https://docs.python.org/3/library/venv.html), then:

```bash
pip install -e .
```

## Run

With that environment activated:

```bash
slurm-top
```

Alias command also works:

```bash
stop
```

You can still run it directly during development:

```bash
python stop.py
```

## JSON output

`--json` prints the same numbers the TUI renders, for scripts and other front
ends (the VS Code extension uses it). Parsing lives in `slurm_top.data`, which
imports nothing outside the standard library, so this mode works in an
interpreter without `textual` installed.

```bash
slurm-top --json                  # one snapshot, then exit
slurm-top --json --pretty         # ... indented
slurm-top --json --watch 3        # one snapshot per line, every 3 seconds
slurm-top --json --job 1234       # scontrol/sstat details for one job
slurm-top --json --node node01    # scontrol details plus the jobs placed there
```

A snapshot holds `jobs`, `nodes`, `disks`, the `gpu` and `summary` aggregates,
and the `user`, `host`, `timestamp` and `schema` it was taken with. Every row
carries derived fields so a consumer never has to re-parse a TRES, GRES or
size string:

- jobs — `gpu_count`, `gpu_types`, `cpu_count`, `mem_mb`
- nodes — `cpus_alloc_n`, `mem_free_mb` / `mem_free_human`, `gpu_total`,
  `gpu_used`, `gpu_free`, `gpu_inventory`, `cpu_load_n`, `cpu_load_ratio`,
  plus the `partition` and drain `reason` from `sinfo`
- disks — `usage_pct`, `size_mb`, `used_mb`, `avail_mb`

Allocated GPUs per node come from a second `sinfo -O GresUsed` call, because
`sinfo -o %G` only reports what a machine *has*, never what is handed out.

## VS Code extension

[`vscode-extension/`](vscode-extension/) ships the same dashboard for VS Code,
in two sizes: a compact activity-bar view, and a full editor tab laid out like
the terminal UI. It runs `slurm-top --json --watch` in the background, so both
share one collector and one parser with the TUI.

Job and machine details open as a dialog over the editor rather than as another
tab, and the extension carries its own copy of the collector, so installing the
Python package is optional for extension users.

```bash
cd vscode-extension
npm install && npm run package    # produces a .vsix
```

Install the `.vsix` with **Extensions: Install from VSIX…**, then reload the
window. Under **Remote - SSH** install it on the remote host, where `squeue`
lives. See [vscode-extension/README.md](vscode-extension/README.md) for
settings, publishing and development notes.

## Tests

```bash
python tests/smoke.py          # data layer, JSON export, CLI and TUI
cd vscode-extension && npm test  # extension host and webview
```

Both talk to the real cluster, so they need `squeue` and `sinfo` on `PATH`, and
they assert invariants rather than specific numbers.

## Build Package

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
```

## Keybindings

- `q` quit
- `r` refresh
- `s` toggle sort-pick mode
- `d` asc/desc
- `f` owner filter (`all`, `me`, `others`)
- `/` open the job search popup (free-text filter, applied while you type)
- `c` open the copy popup for the selected row of the focused panel
- `Enter` open selected job details popup
- `Alt+Left/Right` shrink/grow currently focused panel (tab/click to focus)
- `0` reset panel layout
- In the nodes panel, `Enter` opens the node details popup
- In GPU status panel, `Enter` opens jobs using/reserving selected GPU type
- In job details popup:
  - `c` cancel job
  - `h` hold job
  - `u` release job
  - `r` requeue job
  - `f` refresh now, `a` toggle auto-update
  - `y` copy popup for this job (work dir, command, stdout/stderr paths, ...)
  - `Esc` close popup
- In node details popup:
  - `Enter` open the details of the selected job on that node
  - `f` refresh now
  - `c` copy popup for this node
  - `Esc` close popup
- In the copy popup:
  - `Enter` or mouse click copies the highlighted field
  - `a` copies the whole row as one tab-separated line
  - `Esc` close popup

## Searching jobs

`/` opens a popup with a single input box. Filtering happens live as you type
and every whitespace-separated term has to match somewhere in the row (job id,
user, state, partition, name, node list, CPUs, memory, GRES, time). An empty
box shows all jobs again. The active search, the owner filter, the sort key and
the number of shown/total jobs are all written into the Jobs panel border, e.g.
`Jobs 4/50 | owner=me | sort=state asc | find="train"`.

## Copying values

Terminal DataTables own the mouse for their own row cursor, so dragging over
the jobs or nodes table does not select text. Instead press `c` (or `y` in the
job details popup) to open the copy popup, then `Enter` / click a field. The
value is sent to the system clipboard with an OSC 52 escape, which works over
SSH in most modern terminals; if yours does not support it, the copied value is
echoed at the bottom of the popup as plain text that can be mouse-selected and
copied with `ctrl+c`. The free-text blocks of the job details popup (work dir,
command) can also be mouse-selected directly.

## Panels

- **Left panel (Jobs)**
  - Interactive sorting and filtering (see keybindings)
  - Includes GPU count per job

- **Right panel (Nodes)**
  - Uses full right-column height
  - `Enter` opens node details: state, CPU/memory/GRES allocation, features,
    boot and slurmd start time, plus the jobs currently placed on that node
  - Node state
  - CPU total/allocated/idle
  - Memory:
    - total
    - reserved
    - free
  - Memory is displayed in human-readable units (`M`, `G`, `T`)
  - GPU total per node

- **GPU status (under Nodes)**
  - Cluster totals: total / active / reserved / free estimate
  - Per GPU type: total / active / reserved / free estimate

- **Bottom panel (Job statistics)**
- **Bottom row**
  - GPU status panel
  - Disk usage panel
  - Job statistics panel

- **Bottom panel (Job statistics)**
  - Per owner bucket: all / me / others
  - For running and pending:
    - jobs count
    - GPU sum
    - CPU sum
    - memory sum

## Notes

- Refresh interval is set in `stop.py` (`REFRESH_INTERVAL = 3.0`).
- Some Slurm deployments format memory fields differently; if numbers look off, adjust parsing in `stop.py`.
