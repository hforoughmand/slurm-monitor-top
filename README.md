# Slurm htop-style TUI

A terminal dashboard for Slurm clusters, inspired by `htop`.

## Features

- Live jobs table (all jobs, with optional "my jobs" filter)
- Live node/server table (CPU and memory totals, reserved, and free)
- Job statistics split by:
  - all users
  - your jobs
  - other users
- Live search on jobs with `/`
- Auto refresh every 3 seconds

## Requirements

- Python 3.9+
- Slurm CLI commands available in PATH:
  - `squeue`
  - `sinfo`
- Python packages: see `requirements.txt` (main dependencies: `textual`, `rich`)

## Install

Create and activate a [virtual environment](https://docs.python.org/3/library/venv.html), then:

```bash
pip install -r requirements.txt
```

## Run

With that environment activated:

```bash
python stop.py
```

## Keybindings

- `q` quit
- `r` refresh now
- `f` toggle "only my jobs"
- `/` open search input and filter jobs
- `Esc` clear search and close search input
- `Enter` close search input (keeps current filter)

## Panels

- **Left panel (Jobs)**
  - Sorted by importance:
    1. running
    2. completing
    3. pending
    4. other states
  - Older jobs (smaller job ID) first inside each state group

- **Right panel (Nodes)**
  - Node state
  - CPU total/allocated/idle
  - Memory:
    - total
    - reserved
    - free
  - Memory is displayed in human-readable units (`M`, `G`, `T`)

- **Bottom panel (Job statistics)**
  - Per owner bucket: all / me / others
  - For running and pending:
    - jobs count
    - CPU sum
    - memory sum

## Notes

- Refresh interval is set in `stop.py` (`REFRESH_INTERVAL = 3.0`).
- Some Slurm deployments format memory fields differently; if numbers look off, adjust parsing in `stop.py`.
