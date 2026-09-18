# Changelog

## 0.6.0

### Added

- **What a job asked for against what it is using.** Job details open with a
  block of bars: elapsed time against the limit, CPU time against the cores the
  job holds, peak memory against the reservation. Each is coloured by which end
  of its scale is the dangerous one, so a job at 95% of its memory and a job at
  6% of its cores both read as red — one is about to be killed, the other is
  holding fifteen cores idle. The live figures come from `sstat`, which answers
  only for your own running jobs; on anyone else's the block says so rather than
  drawing an empty bar.
- **The job's stdout and stderr.** Each stream is listed with its path, size and
  last write, and **Open** opens the real file as an editor tab — so it searches
  and follows like any other file. When the file is not reachable from this
  machine (a remote `slurmTop.command`) or is too large to open whole, the last
  1000 lines are fetched through the collector instead.
- **Slurm: Open Job Output (stdout/stderr)…**, for going straight to a job's log
  from the command palette — offering **both**, stdout or stderr when the job
  wrote two files.

- **Pin the jobs you are watching.** Click the star at the left of a job row, or
  press <kbd>p</kbd>, and that job leads the table whatever the sort says. Pins
  are stored by the collector rather than by the editor, so they are shared with
  the `slurm-top` terminal dashboard and agree across the sidebar and the
  dashboard tab.
- **CPU model and speed per machine.** Slurm reports how many cores a node has
  and how they are arranged, but never what they are, so the new `CPU` column
  starts out showing the layout (`2 x 64C/2T`). Open a machine and press
  **Read it from the node** to fill in the model and clock: the collector goes
  over `ssh`, and if the node refuses falls back to a one-second Slurm job that
  runs `lscpu`. The answer is cached and shared with the terminal UI, so each
  machine is read once. Nothing is ever probed unless you ask.
- `slurmTop.cpuProbeUsesSrun` (default `true`) turns off that second route, for
  clusters where submitting a probe job is unwelcome.

### Changed

- A machine's details now open with a **processors** section above the raw
  `scontrol` fields: how many chips of which model (`2 x EPYC 7763`), logical
  CPUs against physical cores, the socket/core/thread breakdown, and the
  architecture.
- Nothing in the editor views cancels or requeues a job — those actions live in
  the terminal dashboard, where they now need `Ctrl+X` and `Ctrl+R` rather than
  a bare letter.
- Every clock is labelled with what it is — `nominal clock` for the speed the
  machine runs at, `max clock` for a boost ceiling, `clock right now` for
  whatever the governor was doing — rather than one unqualified number.

## 0.5.0

### Changed

- Job and machine details now open in a **floating window of their own**, like
  Settings moved out of the main window: resizable, movable, and usable on a
  second monitor. Needs VS Code 1.85 or later; older builds fall back to an
  editor tab.
- `slurmTop.detailsIn` now takes `window` (the new default), `popup`, `card`,
  `tab` or `overlay`. Values from earlier versions keep working.

## 0.4.2

### Changed

- Job and machine details now open as a **floating popup** rather than an editor
  tab: type to filter the fields, Enter to copy one, Escape to dismiss. VS Code
  hosts every webview in a tab, so this is the only detail shape that opens
  without one.
- `slurmTop.detailsIn` gained `popup` (the new default) and renamed `modal` to
  `card`. Existing `modal` and `window` values keep working.

## 0.4.1

First release of the VS Code extension.

### Added

- **Sidebar view** — all five panels (summary, jobs, machines, GPUs, disks) in
  the activity bar, trimmed to the columns that fit a narrow column. Click a
  panel header to fold it away.
- **Dashboard** — a full editor tab laid out like the terminal UI: jobs beside
  machines, GPUs and disks below.
- **Job and machine details** as a centred dialog over the editor, dismissed
  with Escape or a click outside. `slurmTop.detailsIn` switches it to an editor
  tab or to a panel inside the view.
- **Machines** show state with the drain reason inline, partition, allocated and
  idle CPUs, load against the core count, total and free memory, and allocated
  versus installed GPUs.
- **Disks** show usage, size, used and free space; size columns sort by real
  bytes rather than by their labels.
- Sorting on any column, filtering jobs by owner, state or free text, and row
  copying with <kbd>c</kbd>.

### Notes

- The extension carries its own copy of the collector, so
  `pip install slurm-monitor-top` is optional and only adds the `slurm-top`
  terminal dashboard.
- Marked as a workspace extension, so under Remote - SSH it runs on the cluster,
  where the Slurm commands live.
