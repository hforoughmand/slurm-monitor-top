#!/usr/bin/env python3
"""Screenshot the VS Code extension's webview against the fake cluster.

The webview is plain HTML/CSS/JS driven by messages from the extension host
(`vscode-extension/media/main.js`), so it can be rendered outside VS Code: this
script builds a page that loads the real `main.css` and `main.js`, stubs
`acquireVsCodeApi`, and posts the same `config` / `snapshot` / `detail`
messages the host would. The data comes from `slurm-top --json` run against
:mod:`fake_cluster`, so the numbers match the TUI screenshots exactly.

    python tools/screenshots/capture_webview.py             # all scenes
    python tools/screenshots/capture_webview.py ext-sidebar

What it cannot show is VS Code itself -- the activity bar, tabs and title bar
are not part of the extension and are not faked here.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# The extension's screenshots live with the extension, so its README can link
# to them relatively and vsce can rewrite those links to a URL that resolves.
ASSETS = ROOT / "vscode-extension" / "assets"
MEDIA = ROOT / "vscode-extension" / "media"

DEVICE_SCALE = 2

sys.path.insert(0, str(HERE))
from capture_tui import CHROME_CANDIDATES, find_chrome, finalize  # noqa: E402

# A stand-in for VS Code's Dark Modern theme. The webview inherits these from
# the editor at runtime; without them every `var(--vscode-…)` would fall back
# to nothing and the page would render as black text on a transparent page.
THEME = {
    "font-family": '"Segoe UI", system-ui, "Ubuntu", "Droid Sans", sans-serif',
    "font-size": "13px",
    "editor-font-family": '"Fira Code", "Droid Sans Mono", monospace',
    "foreground": "#cccccc",
    "descriptionForeground": "#9d9d9d",
    "errorForeground": "#f85149",
    "editor-background": "#1f1f1f",
    "editorWidget-background": "#202020",
    "editorGroupHeader-tabsBackground": "#181818",
    "panel-border": "#2b2b2b",
    "widget-border": "#313131",
    "list-activeSelectionBackground": "#04395e",
    "list-activeSelectionForeground": "#ffffff",
    "list-hoverBackground": "#2a2d2e",
    "button-secondaryBackground": "#313131",
    "button-secondaryForeground": "#cccccc",
    "button-secondaryHoverBackground": "#3c3c3c",
    "input-background": "#313131",
    "input-foreground": "#cccccc",
    "input-border": "#3c3c3c",
    "inputValidation-errorBackground": "#5a1d1d",
    "inputValidation-errorForeground": "#cccccc",
    "inputValidation-errorBorder": "#be1100",
    "progressBar-background": "#0078d4",
    "textLink-foreground": "#4daafc",
    "textLink-activeForeground": "#4daafc",
    "charts-blue": "#3794ff",
    "charts-green": "#89d185",
    "charts-red": "#f14c4c",
    "charts-yellow": "#cca700",
}

ALL_SECTIONS = ["summary", "jobs", "nodes", "gpus", "disks"]
# What `slurmTop.sidebarSections` is for: a narrow panel reads better with the
# wide machine table left to the dashboard.
SIDEBAR_SECTIONS = ["summary", "jobs", "gpus"]

# The webview restores its own state on load; seeding it is how a user who has
# already picked "all" and clicked a row would see the panel. A selection is
# keyed by `server/id`, because two clusters can both have a job 184220.
SAVED_STATE = {"ownerFilter": "all", "selected": {"jobs": "login01/184220", "nodes": "login01/gpu03"}}

# Pins live in the collector's config file, shared with the terminal UI, so the
# capture seeds a config rather than faking the state in the view.
PINNED_JOBS = ["184244", "184288"]

# The second cluster, for the scenes that show more than one. `FAKE_CLUSTER`
# picks its data out of fake_cluster.py; the name is what the panels show.
SITES = [("login01", "login01"), ("hpc2", "hpc2")]

# Merging as the settings default: one box per cluster, one table for all of
# them. The split scene is the other way round for the jobs table.
MERGED = {"summary": False, "jobs": True, "nodes": True, "gpus": True, "disks": True}
SPLIT = dict(MERGED, jobs=False, summary=True)

# name -> (variant, sections, detail target, viewport, servers)
SCENES = {
    "ext-sidebar": ("sidebar", SIDEBAR_SECTIONS, None, (520, 700), 1, MERGED),
    "ext-dashboard": ("dashboard", ALL_SECTIONS, None, (1500, 940), 1, MERGED),
    "ext-job-detail": ("detail", ALL_SECTIONS, ("job", "184220"), (1400, 820), 1, MERGED),
    "ext-node-detail": ("dashboard", ALL_SECTIONS, ("node", "gpu03"), (1500, 940), 1, MERGED),
    "ext-two-servers": ("dashboard", ALL_SECTIONS, None, (1500, 1000), 2, MERGED),
    "ext-two-servers-split": ("dashboard", ALL_SECTIONS, None, (1500, 1000), 2, SPLIT),
}


def collect(env, *args):
    """Run `slurm-top --json ...` against the fake cluster and parse stdout."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); "
         "from slurm_top.export import main; sys.exit(main(sys.argv[1:]))" % str(ROOT / "src"),
         *args],
        check=True, capture_output=True, text=True, env=env, timeout=120,
    )
    return json.loads(out.stdout)


def merge(parts):
    """Several snapshots as one, the way `src/merge.ts` does it for the host.

    Kept in step with that file by hand -- it is twenty lines of addition, and
    the alternative is running TypeScript to take a screenshot.
    """
    servers = []
    jobs, nodes, disks = [], [], []
    gpu = {"total": 0, "active": 0, "reserved": 0, "free_est": 0,
           "per_type": {}, "per_type_stats": {}, "types_count": 0}
    summary = {b: {p: {"jobs": 0, "cpus": 0, "mem_mb": 0, "gpus": 0}
                   for p in ("running", "pending")}
               for b in ("all", "me", "others")}
    for server_id, name, snapshot in parts:
        servers.append({
            "id": server_id, "name": name, "host": snapshot["host"], "user": snapshot["user"],
            "state": "running", "timestamp": snapshot["timestamp"], "gpu": snapshot["gpu"],
            "summary": snapshot["summary"], "pinned": snapshot.get("pinned", []),
            "counts": {k: len(snapshot[k]) for k in ("jobs", "nodes", "disks")},
        })
        tag = {"server": server_id, "server_name": name}
        jobs += [dict(row, **tag) for row in snapshot["jobs"]]
        nodes += [dict(row, **tag) for row in snapshot["nodes"]]
        disks += [dict(row, **tag) for row in snapshot["disks"]]
        for key in ("total", "active", "reserved", "free_est"):
            gpu[key] += snapshot["gpu"].get(key, 0)
        for model, count in snapshot["gpu"].get("per_type", {}).items():
            gpu["per_type"][model] = gpu["per_type"].get(model, 0) + count
        for model, stats in snapshot["gpu"].get("per_type_stats", {}).items():
            into = gpu["per_type_stats"].setdefault(
                model, {"total": 0, "active": 0, "reserved": 0, "free_est": 0})
            for key, value in stats.items():
                into[key] += value
        for bucket, phases in snapshot["summary"].items():
            for phase, cell in phases.items():
                for key, value in cell.items():
                    summary[bucket][phase][key] += value
    gpu["types_count"] = len(gpu["per_type_stats"])
    first = parts[0][2]
    return dict(first, servers=servers, jobs=jobs, nodes=nodes, disks=disks,
                gpu=gpu, summary=summary)


def page(variant, sections, snapshot, detail, merge_settings=None, servers=None):
    """The harness page: real stylesheet, real script, stubbed host."""
    messages = [
        {"type": "config", "sections": sections, "ownerFilter": "all",
         "interval": 3, "detailsIn": "overlay" if variant != "modal" else "window",
         "servers": servers or [], "merge": merge_settings or MERGED},
        {"type": "status", "state": "running"},
        {"type": "snapshot", "snapshot": snapshot},
    ]
    if detail is not None:
        messages.append({"type": "detail", "detail": detail})
    theme = "".join(f"  --vscode-{k}: {v};\n" for k, v in THEME.items())
    saved = json.dumps(SAVED_STATE)
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <style>
:root {{
{theme}}}
html, body {{ margin: 0; padding: 0; }}
    </style>
    <link rel="stylesheet" href="{(MEDIA / 'main.css').as_uri()}" />
  </head>
  <body data-variant="{variant}">
    <div id="status" class="status">Starting the Slurm collector…</div>
    <div id="root"></div>
    <div id="overlay" class="overlay hidden" role="dialog" aria-modal="true"></div>
    <script>
      var __posted = [];
      function acquireVsCodeApi() {{
        return {{
          postMessage: function (m) {{ __posted.push(m); }},
          getState: function () {{ return {saved}; }},
          setState: function () {{}},
        }};
      }}
    </script>
    <script src="{(MEDIA / 'main.js').as_uri()}"></script>
    <script>
      for (const message of {json.dumps(messages)}) {{
        window.dispatchEvent(new MessageEvent('message', {{ data: message }}));
      }}
    </script>
  </body>
</html>
"""


def shoot(chrome, html_path, png_path, viewport):
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [
                chrome, "--headless", "--disable-gpu", "--no-sandbox",
                "--hide-scrollbars", "--allow-file-access-from-files",
                f"--force-device-scale-factor={DEVICE_SCALE}",
                f"--window-size={viewport[0]},{viewport[1]}",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=2000",
                f"--screenshot={png_path}",
                html_path.as_uri(),
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
    finalize(png_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="scene names (default: all)")
    parser.add_argument("--out", default=str(ASSETS), help="output directory")
    args = parser.parse_args(argv)

    names = args.scenes or list(SCENES)
    unknown = [n for n in names if n not in SCENES]
    if unknown:
        parser.error(f"unknown scene(s): {', '.join(unknown)} (have: {', '.join(SCENES)})")

    chrome = find_chrome()
    if chrome is None:
        parser.error(f"no Chrome binary found (tried {', '.join(CHROME_CANDIDATES)})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import fake_cluster

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        bindir = tmpdir / "bin"
        fake_cluster.install(str(bindir))
        # A HOME of its own: the collector reads pins (and CPU readings) from
        # the config file under it, and a capture must never pick up - or
        # write - anything belonging to the person running it.
        home = tmpdir / "home"
        (home / ".config" / "slurm-monitor-top").mkdir(parents=True)
        (home / ".config" / "slurm-monitor-top" / "config.json").write_text(
            json.dumps({"pinned_jobs": {job: time.time() for job in PINNED_JOBS}})
        )
        env = dict(os.environ)
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
        env["USER"] = fake_cluster.ME
        env["HOME"] = str(home)

        collected = []
        for site, label in SITES:
            site_env = dict(env, FAKE_CLUSTER=site)
            snapshot = collect(site_env)
            snapshot["host"] = label
            collected.append((label, label, snapshot))
        details = {}
        for name in names:
            target = SCENES[name][2]
            if target and target not in details:
                flag = "--job" if target[0] == "job" else "--node"
                details[target] = collect(env, flag, target[1])

        for name in names:
            variant, sections, target, viewport, count, merge_settings = SCENES[name]
            parts = collected[:count]
            html = tmpdir / f"{name}.html"
            html.write_text(page(
                variant, sections, merge(parts), details.get(target),
                merge_settings, [{"id": s[0], "name": s[1]} for s in parts],
            ))
            png = out / f"{name}.png"
            shoot(chrome, html, png, viewport)
            print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
