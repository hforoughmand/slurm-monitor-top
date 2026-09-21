# Slurm htop-style TUI

A terminal dashboard for Slurm clusters, inspired by `htop`: every job, machine,
GPU and filesystem on one screen, refreshed in place, with the full `scontrol`
detail of any row one keypress away. The same data layer also feeds a JSON mode
and a VS Code extension.

## Project Links

- GitHub: [hforoughmand/slurm-monitor-top](https://github.com/hforoughmand/slurm-monitor-top)
- Issues: [github.com/hforoughmand/slurm-monitor-top/issues](https://github.com/hforoughmand/slurm-monitor-top/issues)

![The slurm-top dashboard](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-overview.png)

> **About the screenshots.** Every image on this page is captured against a
> synthetic cluster defined in
> [`tools/screenshots/fake_cluster.py`](https://github.com/hforoughmand/slurm-monitor-top/blob/main/tools/screenshots/fake_cluster.py): the
> users, job names, machines, partitions and paths are invented, so no real
> cluster or person appears anywhere.

## Features

- Live jobs table with interactive sort, owner filter and free-text search
- Live machine table: CPU and memory totals, reserved and free, GPUs per node
- Job details popup with the `scontrol` record, live `sstat` usage, and the
  cancel / hold / release / requeue actions
- Asked-for-versus-used bars on a running job: time against its limit, CPU time
  against the cores it holds, peak memory against the reservation
- The job's stdout and stderr, tailed side by side and following as they grow
  (`o`)
- Machine details popup listing the jobs Slurm placed on that node
- GPU accounting per model: total, active, reserved and a free estimate, plus
  the jobs behind those numbers
- Live disk table (`df -h`) with usage, mount path, type and size
- Job statistics split by all users / your jobs / other users
- Pin the jobs you are watching (`p`) so they stay on top of the table, in the
  terminal UI and the editor alike
- CPU model and layout per machine, read on demand (`p` in the machine popup)
- Copy popup (`c`) for the selected row of any panel, over SSH included
- Auto refresh every 3 seconds that keeps your scroll position and selection
- Resizable panels (`Alt+←` / `Alt+→`)
- JSON output (`slurm-top --json`) for other front ends
- A [VS Code extension](https://github.com/hforoughmand/slurm-monitor-top/tree/main/vscode-extension) with the same panels, in the sidebar
  or a full editor tab — and with several clusters merged into one view

## Feature tour

### One screen for the whole cluster

The default layout is four panels: jobs on the left, machines on the right, and
GPU status, disk usage and job statistics along the bottom. Every panel border
carries its own live counts and the filters in force, so the state of the view
is visible without opening a menu — `Jobs 26/26 | owner=all | sort=state asc`.
Everything refreshes every 3 seconds, in place: the scroll position, the
selected row and the sort order all survive the update.

### Job details, and acting on a job

`Enter` on a job shows the whole `scontrol show job -d` record, laid out rather
than dumped: state and pending reason, account and QOS, node list, time used
against the limit, submit and start times, the allocated TRES including GPUs,
and — for a running job you own — live `MaxRSS`, `MaxVMSize` and `AveCPU` from
`sstat`. The work directory and command line sit in their own scroller so long
paths do not push the rest of the block around.

![The job details popup](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-job-details.png)

The action row is not decoration: `^X` cancels and `^R` requeues the job, `h`
holds it and `u` releases it, `f` refreshes now, `a` toggles a 3-second
auto-update of the popup, and `y` (or `c`) opens the copy popup for this job's
paths and command.

The split is deliberate. **A bare letter never destroys work**: it either reads
something, or the letter beside it undoes it — `u` releases what `h` held. The
two actions that throw away a running job, cancel and requeue, need a modifier.
`c` in particular used to cancel, which was the worst place for it, since `c`
copies in every other panel; it copies here too now.

### How much of the reservation is actually working

Slurm hands out whole cores and whole gigabytes for the lifetime of a job, and
says nothing about what became of them. A job that reserved 16 cores and runs
one thread holds the other fifteen idle for days, and the only trace is the
arithmetic nobody does. The job popup does it:

![Asked for against used](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-job-usage.png)

Three bars, and each one is coloured by *which end of its scale is the bad one*.
Near the time limit or the memory ceiling is red because the job is about to be
killed; low CPU efficiency is red because the machine is being wasted. The
numbers come from `RunTime` against `TimeLimit`, `sstat`'s CPU time against
elapsed × allocated cores, and its `MaxRSS` against the memory reserved per
node.

`sstat` answers only for your own running jobs, so on someone else's job the
time bar stands alone and the popup says why rather than showing an empty one.
GPUs get a line without a bar: Slurm accounts for the reservation and never
measures the use.

### The job's output, while it runs

`o` — on a job in the table, or in its details popup — tails the files the job
is writing to. Both streams at once, one above the other, because a job that
has gone wrong usually says so in stderr while stdout keeps printing:

![stdout and stderr, tailed together](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-job-output.png)

Slurm stores the paths and nothing else, so this is ordinary file reading from
the login node: whatever `tail -f` could see, this sees. Only the last stretch
of each file is read, so following a job that has been printing for a week costs
what following a fresh one costs. The action row names the three views —
**stdout**, **stderr**, **both** — with the one in force marked, so switching
when one stream is the noisy one is a click or `1`/`2`/`3` rather than a guess
(`o` still cycles them). `a` toggles following, `+` and `-` change how much
scrollback is kept, and `y` copies the paths or the visible text. A script that
sent both streams to one file gets one pane, correctly labelled, not two copies.

The two things it cannot do: a job writing to scratch that is local to the
compute node leaves a path nothing on the login node can open, and `scontrol`
shows the paths only to the job's owner.

### Machine details

`Enter` on a machine gives `scontrol show node` the same treatment: state and
drain reason, CPU and memory allocation, GRES with the indices actually handed
out, features, boot and `slurmd` start times — followed by the jobs Slurm
currently places on that node. `Enter` on one of those walks straight into its
job details.

![The machine details popup](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-node-details.png)

### GPUs: what exists, what runs, what is merely reserved

The GPU panel keeps apart the three numbers that usually get conflated: how many
GPUs of each model the cluster has, how many are handed to running jobs, and how
many pending jobs have reserved. Allocated GPUs come from a second
`sinfo -O GresUsed` call, because `sinfo -o %G` only reports what a machine
*has*, never what is in use.

`Enter` on a GPU type lists the jobs behind those totals, tagged `USING` or
`RESERVING`.

![Jobs using and reserving one GPU type](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-gpu-jobs.png)

### Finding jobs

`/` opens a search box that filters as you type. Every whitespace-separated term
has to match somewhere in the row — job id, user, state, partition, name, node
list, CPUs, memory, GRES or time — so `gpu alice` means "alice's GPU jobs". An
empty box shows everything again.

![The job search popup](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-job-search.png)

The active search stays in the panel border after the box closes, next to the
owner filter, the sort key and the shown/total count:

![A jobs panel filtered by a search term](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-search-applied.png)

### Pinning the jobs you are watching

A long queue buries the two jobs you actually care about. `p` pins the selected
job: pinned rows float to the top of the table whatever the sort says, marked in
the `P` column, and the panel border counts them. Pins live in the config file,
so a job pinned in the terminal is already on top in the VS Code extension, and
they survive a restart.

![Two pinned jobs held at the top of the table](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-job-pin.png)

### What CPU is in that machine?

`sinfo` and `scontrol` report how many cores a node has, never *what* they are.
`p` in the machine popup goes and looks: first over `ssh`, and if the node
refuses (many clusters only let you in where you already have a job), through a
one-second `srun` job that prints `lscpu`. The answer — model, nominal or boost
clock, sockets by cores by threads — is cached, shown in the machine table's
`CPU` column, and shared with the extension.

![A machine's CPU model read over ssh](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-node-cpu.png)

The machine popup answers it as three lines — how many, of what, at what speed:

```
CPU    96 CPUs    48 cores    2 sockets x 24 cores x 2 threads    arch x86_64
Model  2 x EPYC 7763    24 cores each    (read over ssh)
Speed  max 3.53GHz    now 1.79GHz
```

`96 CPUs` is what Slurm hands out and what `nproc` reports; `48 cores` is the
silicon behind them. Every clock is labelled with what it actually is: a model
name carrying its own `@ 2.60GHz` is `nominal`, the speed the node runs at,
while `max` is a boost ceiling one core reaches and `now` is whatever the
governor happened to be doing when we looked.

### Owner filter and sorting

`f` cycles the owner filter through `all`, `me` and `others` — the fastest way
to answer "what am I actually running?".

![The jobs panel filtered to your own jobs](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-owner-filter.png)

`s` opens the sort picker (or press `1`..`8` directly), and `d` flips between
ascending and descending.

![The sort picker](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-sort-picker.png)

### Copying values out of a terminal table

Terminal tables own the mouse for their own row cursor, so dragging across the
jobs table does not select text. `c` opens a copy popup for the selected row
instead: `Enter` or a click copies one field, `a` copies the whole row as one
tab-separated line. Values go to the system clipboard through an OSC 52 escape,
which works over SSH in most modern terminals; where it does not, the value is
echoed as plain selectable text at the bottom of the popup.

![The copy popup](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-copy-popup.png)

### Panels that give way

`Alt+←` and `Alt+→` resize the focused panel and `0` restores the default
layout. Widening the machine panel brings out the memory and GPU columns a
default split has no room for:

![The machine panel widened with Alt+Right](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/assets/tui-panel-resize.png)

## VS Code extension

[`vscode-extension/`](https://github.com/hforoughmand/slurm-monitor-top/tree/main/vscode-extension) ships the same dashboard for VS Code.
It runs `slurm-top --json --watch` in the background, so the editor and the
terminal UI share one collector and one parser.

A compact activity-bar view, for keeping an eye on the queue while you work
(which panels it shows is a setting, `slurmTop.sidebarSections`):

![The extension's sidebar view](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/vscode-extension/assets/ext-sidebar.png)

And a full editor tab laid out like the terminal UI, with sortable columns and
allocation bars:

![The extension's dashboard tab](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/vscode-extension/assets/ext-dashboard.png)

Job and machine details open as a dialog over the editor rather than as another
tab — an overlay on the dashboard, a quick-pick popup, or a view of their own,
whichever `slurmTop.detailsIn` says:

![Machine details over the dashboard](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/vscode-extension/assets/ext-node-detail.png)

![Job details in a view of their own](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/vscode-extension/assets/ext-job-detail.png)

The editor view can watch **several clusters at once**, which the terminal UI
does not: `slurmTop.servers` is a name → where grid you fill in from VS Code's
settings editor — `here`, an ssh destination, or a command of your own — with
one collector process per cluster. A cluster reached over ssh needs nothing
installed on it: the extension sends its own copy of the collector, runs it in
memory there and reads the JSON back. Their jobs, machines, GPUs and
disks arrive in one set of tables, each row tagged with the cluster it came
from, with a server filter beside the owner and state ones — or, section by
section, one panel per cluster instead.

![Two clusters in one dashboard](https://raw.githubusercontent.com/hforoughmand/slurm-monitor-top/main/vscode-extension/assets/ext-two-servers.png)

The extension carries its own copy of the collector, so installing the Python
package is optional for extension users.

```bash
cd vscode-extension
npm install && npm run package    # produces a .vsix
```

Install the `.vsix` with **Extensions: Install from VSIX…**, then reload the
window. Under **Remote - SSH** install it on the remote host, where `squeue`
lives. See [vscode-extension/README.md](https://github.com/hforoughmand/slurm-monitor-top/blob/main/vscode-extension/README.md) for
settings, publishing and development notes.

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
slurm-top --json --job-output 1234           # the tail of that job's stdout
slurm-top --json --job-output 1234 --stream stderr --lines 500
slurm-top --json --node node01    # scontrol details plus the jobs placed there
slurm-top --json --node node01 --probe-cpu   # ... and go read its CPU model
slurm-top --json --pin 1234       # pin a job (--unpin, --toggle-pin)
```

A snapshot holds `jobs`, `nodes`, `disks`, the `gpu` and `summary` aggregates,
and the `user`, `host`, `timestamp` and `schema` it was taken with. Every row
carries derived fields so a consumer never has to re-parse a TRES, GRES or
size string:

- jobs — `gpu_count`, `gpu_types`, `cpu_count`, `mem_mb`
- nodes — `cpus_alloc_n`, `mem_free_mb` / `mem_free_human`, `gpu_total`,
  `gpu_used`, `gpu_free`, `gpu_inventory`, `cpu_load_n`, `cpu_load_ratio`,
  plus the `partition` and drain `reason` from `sinfo`, and a `cpu` block
- disks — `usage_pct`, `size_mb`, `used_mb`, `avail_mb`

A node's `cpu` block carries `sockets`, `cores_per_socket`, `threads_per_core`,
a rendered `topology`, and the counts they add up to — `cpus_total` (logical),
`cores` (physical) and `processors` (chips) — all of which Slurm always reports.
`model`, `count_summary` (`2 x EPYC 7763`), `speeds` (each labelled `nominal`,
`max` or `now`), `vendor`, `arch` and `source` are filled in only for machines
that have been probed, and `known` says which is which. The snapshot also lists
the `pinned` job ids.

A job lookup adds two things Slurm does not print. `metrics` is the
request-versus-use list the popups draw as bars: each entry carries a `label`,
a `percent` (null on the ones nothing bounds), the two figures as `value` and
`total`, a `note` saying what it means, and `risk` — `high` when a full bar is
the dangerous one (time, memory), `low` when an empty one is (CPU efficiency).
`output` names both streams with the path, size, mtime and, when there is
nothing to read, the `error` that says why: no path recorded, `/dev/null`, not
written yet, or another user's job. It never reads the files themselves, so a
three-second refresh stays cheap.

`--job-output` is the one that reads them. It returns both `streams` and an
`output` block with the `text` of the last `--lines` lines, `line_count`, and
`truncated` when the file is longer than what came back. Only the final chunk
of the file is read, so the cost does not grow with the log.

`--probe-cpu` is the only thing here that leaves the login node: it opens an
`ssh` connection to the machine, or failing that submits a one-second job.
Nothing probes on its own — a plain `--node` lookup reads the cache and stops
there. `--no-srun` restricts it to `ssh`.

Allocated GPUs per node come from a second `sinfo -O GresUsed` call, because
`sinfo -o %G` only reports what a machine *has*, never what is handed out.

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
- `p` pin / unpin the selected job (pinned jobs stay on top)
- `/` open the job search popup (free-text filter, applied while you type)
- `c` open the copy popup for the selected row of the focused panel
- `Enter` open selected job details popup
- `o` open the output viewer for the selected job (stdout and stderr, tailed)
- `Alt+Left/Right` shrink/grow currently focused panel (tab/click to focus)
- `0` reset panel layout
- In the nodes panel, `Enter` opens the node details popup
- In GPU status panel, `Enter` opens jobs using/reserving selected GPU type
- In job details popup (a bare letter never destroys work; the two actions that
  do need a modifier):
  - `Ctrl+X` cancel job (`scancel`)
  - `Ctrl+R` requeue job — kills the running job and re-queues it
  - `h` hold job, `u` release it again
  - `f` or `r` refresh now, `a` toggle auto-update
  - `p` pin / unpin this job
  - `o` open the output viewer (stdout and stderr, tailed together)
  - `y` or `c` copy popup for this job (work dir, command, stdout/stderr paths, ...)
  - `Esc` close popup
  - The action row along the bottom still runs any of these on click or `Enter`,
    including the two behind modifiers
- In node details popup:
  - `Enter` open the details of the selected job on that node
  - `p` read this node's CPU model (ssh, then a one-second `srun` job) — it
    submits a job but destroys nothing, so it stays a bare letter
  - `f` refresh now
  - `c` copy popup for this node
  - `Esc` close popup
- In the output viewer:
  - `1` stdout only, `2` stderr only, `3` both — the same three the action row
    offers by name, with the one in force marked; `o` cycles them
  - `a` follow (re-read every 2 seconds), `f` or `r` re-read now
  - `+` / `-` how many trailing lines to keep (50 / 200 / 1000 / 5000)
  - `Home` / `End` jump to the top or the bottom of the focused pane
  - `y` copy popup (both paths, a ready-made `tail -f`, the visible text)
  - `Esc` close popup
- In the copy popup:
  - `Enter` or mouse click copies the highlighted field
  - `a` copies the whole row as one tab-separated line
  - `Esc` close popup

## Panels

- **Left panel (Jobs)**
  - Interactive sorting and filtering (see keybindings)
  - Includes GPU count per job
  - `P` column marks pinned jobs, which sort above everything else

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
  - CPU column: the model once probed, the socket/core/thread layout until then

- **GPU status (under Nodes)**
  - Cluster totals: total / active / reserved / free estimate
  - Per GPU type: total / active / reserved / free estimate

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

## Screenshots

The images above are generated, not photographed, so they can be redone whenever
the UI changes and never carry real cluster data:

```bash
python tools/screenshots/capture_tui.py       # the terminal UI, scene by scene
python tools/screenshots/capture_webview.py   # the VS Code extension's webview
```

Both drive the real code against the fake cluster in
[`tools/screenshots/fake_cluster.py`](https://github.com/hforoughmand/slurm-monitor-top/blob/main/tools/screenshots/fake_cluster.py), which
stands in for `squeue`, `sinfo`, `scontrol`, `sstat` and `df`. See
[tools/screenshots/README.md](https://github.com/hforoughmand/slurm-monitor-top/blob/main/tools/screenshots/README.md) for the scene list
and what the scripts need installed.

## Notes

- Refresh interval is set in `stop.py` (`REFRESH_INTERVAL = 3.0`).
- Some Slurm deployments format memory fields differently; if numbers look off, adjust parsing in `stop.py`.
