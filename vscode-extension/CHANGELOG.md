# Changelog

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
