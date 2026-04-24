# Slurm htop-style TUI

A terminal dashboard for Slurm clusters, inspired by `htop`.

## Project Links

- GitHub: [hforoughmand/slurm-monitor-top](https://github.com/hforoughmand/slurm-monitor-top)
- Issues: [github.com/hforoughmand/slurm-monitor-top/issues](https://github.com/hforoughmand/slurm-monitor-top/issues)

## Screenshot

![slurm-top screenshot](https://github.com/hforoughmand/slurm-monitor-top/assets/screenshot-01.png)

## Features

- Live jobs table with interactive sort and filters
- Live node/server table (CPU and memory totals, reserved, and free)
- Job statistics split by:
  - all users
  - your jobs
  - other users
- GPU status under nodes:
  - number of GPU types
  - total GPUs
  - active GPUs (running jobs)
  - reserved GPUs (pending jobs)
- Auto refresh every 3 seconds

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

## Panels

- **Left panel (Jobs)**
  - Interactive sorting and filtering (see keybindings)
  - Includes GPU count per job

- **Right panel (Nodes)**
  - Uses full right-column height
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
  - Per owner bucket: all / me / others
  - For running and pending:
    - jobs count
    - GPU sum
    - CPU sum
    - memory sum

## Notes

- Refresh interval is set in `stop.py` (`REFRESH_INTERVAL = 3.0`).
- Some Slurm deployments format memory fields differently; if numbers look off, adjust parsing in `stop.py`.
