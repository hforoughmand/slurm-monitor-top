#!/usr/bin/env python3
"""Drive the TUI against the fake cluster and write one PNG per feature.

Every scene runs a fresh :class:`SlurmHtop` headless (Textual's own test
driver), presses the keys a user would, exports the frame as SVG and hands it
to headless Chrome for the PNG. Nothing here talks to a real cluster: PATH is
pointed at :mod:`fake_cluster` first, so squeue/sinfo/scontrol/sstat/df are the
fixtures in that file.

    python tools/screenshots/capture_tui.py            # all scenes -> assets/
    python tools/screenshots/capture_tui.py overview   # just one

Needs `textual` (so run it in the project's virtualenv), a Chrome/Chromium
binary for the SVG -> PNG step and ImageMagick's `convert` to trim the result;
with --svg-only both of those are skipped.
"""

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = ROOT / "assets"

# Wide enough that no column is elided, short enough to stay readable on GitHub.
TERMINAL_SIZE = (170, 46)
DEVICE_SCALE = 2

CHROME_CANDIDATES = [
    "chromium-browser", "chromium", "google-chrome", "google-chrome-stable", "chrome",
]


def fake_path(tmpdir: Path) -> str:
    """Install the fake Slurm commands and return a PATH with them in front."""
    sys.path.insert(0, str(HERE))
    import fake_cluster

    bindir = tmpdir / "bin"
    fake_cluster.install(str(bindir))
    return f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"


# --------------------------------------------------------------------- scenes
# Each scene is an async function taking (app, pilot); it leaves the app in the
# state to be photographed. Order in this dict is the order they are captured.


async def scene_overview(app, pilot):
    """The default view: jobs, machines, GPUs, disks and the statistics panel."""
    await pilot.pause()


async def scene_job_search(app, pilot):
    """The `/` search popup, filtering as the terms are typed."""
    await pilot.press("slash")
    await pilot.pause()
    for key in "gpu alice":
        await pilot.press("space" if key == " " else key)
    await pilot.pause()


async def scene_search_applied(app, pilot):
    """The jobs panel after a search: the border carries the active filters."""
    await pilot.press("slash")
    await pilot.pause()
    for key in "train":
        await pilot.press(key)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


async def scene_job_details(app, pilot):
    """`Enter` on a job: scontrol detail, the usage bars and the job actions."""
    app.jobs_view.move_cursor(row=1)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.6)


async def scene_job_usage(app, pilot):
    """The request-versus-use bars on a job that is wasting its reservation."""
    # Row 3 is the 8-CPU job running at 6% of them; the bars are the point.
    app.jobs_view.move_cursor(row=3)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.8)


async def scene_job_output(app, pilot):
    """`o` on a job: its stdout and stderr, tailed side by side."""
    app.jobs_view.move_cursor(row=1)
    await pilot.pause()
    await pilot.press("o")
    # The tail runs in a worker; wait for the files rather than guess.
    for _ in range(60):
        await asyncio.sleep(0.05)
        if getattr(app.screen, "tails", None):
            break
    await pilot.pause(0.4)


async def scene_node_details(app, pilot):
    """`Enter` on a machine: its state, allocation and the jobs placed on it."""
    app.set_focus(app.nodes_view)
    app.nodes_view.move_cursor(row=2)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.6)


async def scene_gpu_jobs(app, pilot):
    """`Enter` on a GPU type: every job using or reserving that model."""
    app.set_focus(app.gpu_status_view)
    app.gpu_status_view.move_cursor(row=3)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.6)


async def scene_copy_popup(app, pilot):
    """`c`: every field of the selected row, one keypress from the clipboard."""
    app.jobs_view.move_cursor(row=1)
    await pilot.pause()
    await pilot.press("c")
    await pilot.pause(0.4)


async def scene_sort_picker(app, pilot):
    """`s`: pick the sort column by row or hotkey, `d` flips the direction."""
    await pilot.press("s")
    await pilot.pause(0.4)


async def scene_owner_filter(app, pilot):
    """`f`: cycle the owner filter through all / me / others."""
    await pilot.press("f")
    await pilot.pause(0.4)


async def scene_job_pin(app, pilot):
    """`p`: pin the jobs you are watching, and they stay on top of the table."""
    for row in (3, 12):
        app.jobs_view.move_cursor(row=row)
        await pilot.pause()
        await pilot.press("p")
        await pilot.pause(0.2)
    app.jobs_view.move_cursor(row=0)
    await pilot.pause(0.5)


async def scene_node_cpu(app, pilot):
    """`p` in the machine popup: read that node's CPU model over ssh or srun."""
    app.set_focus(app.nodes_view)
    app.nodes_view.move_cursor(row=2)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause(0.6)
    await pilot.press("p")
    # The probe runs in a worker; give the (fake) ssh time to answer.
    for _ in range(40):
        await asyncio.sleep(0.05)
        if not getattr(app.screen, "probing", True):
            break
    await pilot.pause(0.4)


async def scene_panel_resize(app, pilot):
    """`Alt+Right` on the machines panel: every memory and GPU column fits."""
    app.set_focus(app.nodes_view)
    await pilot.pause()
    for _ in range(3):
        await pilot.press("alt+right")
    await pilot.pause(0.4)


SCENES = {
    "overview": scene_overview,
    "job-search": scene_job_search,
    "search-applied": scene_search_applied,
    "job-details": scene_job_details,
    "job-usage": scene_job_usage,
    "job-output": scene_job_output,
    "node-details": scene_node_details,
    "gpu-jobs": scene_gpu_jobs,
    "copy-popup": scene_copy_popup,
    "sort-picker": scene_sort_picker,
    "owner-filter": scene_owner_filter,
    "job-pin": scene_job_pin,
    "node-cpu": scene_node_cpu,
    "panel-resize": scene_panel_resize,
}


async def shoot(name, scene, out_svg):
    from slurm_top.app import SlurmHtop

    app = SlurmHtop()
    async with app.run_test(size=TERMINAL_SIZE) as pilot:
        # The first refresh is a worker; wait for it before touching rows.
        await pilot.pause()
        for _ in range(40):
            if app.jobs_view.jobs:
                break
            await asyncio.sleep(0.05)
        await pilot.pause()
        await scene(app, pilot)
        out_svg.write_text(patch_svg(app.export_screenshot(title="slurm-top")))


# ------------------------------------------------------------------ svg -> png
def patch_svg(text):
    """Repair Rich's SVG export so the bottom row survives rendering.

    Rich emits one `<clipPath id="...-line-N">` per line but stops one line
    short, while still pointing the last row's `<text>` elements at it. A
    clip-path that resolves to nothing means "draw nothing" in every browser,
    so the footer (the keybinding bar) silently disappears. Synthesise the
    missing clips from the geometry of the ones that are there, and grow the
    viewBox to whatever the content actually needs.
    """
    defined = {int(n) for n in re.findall(r'id="terminal-\d+-line-(\d+)"', text)}
    used = {int(n) for n in re.findall(r'url\(#terminal-\d+-line-(\d+)\)', text)}
    missing = sorted(used - defined)
    if not missing:
        return text

    geometry = re.search(
        r'<clipPath id="terminal-(\d+)-line-0">\s*<rect x="([\d.]+)" y="([\d.]+)"'
        r' width="([\d.]+)" height="([\d.]+)"',
        text,
    )
    if geometry is None:
        return text
    uid, x0, y0, width, row_height = geometry.groups()
    y0, row_height = float(y0), float(row_height)
    # The step between rows is smaller than a row's height (Rich overlaps them
    # a little so backgrounds abut), so measure it instead of assuming.
    second = re.search(rf'id="terminal-{uid}-line-1">\s*<rect x="[\d.]+" y="([\d.]+)"', text)
    step = float(second.group(1)) - y0 if second else row_height

    clips = "".join(
        f'<clipPath id="terminal-{uid}-line-{n}">\n'
        f'      <rect x="{x0}" y="{y0 + n * step:.1f}" width="{width}"'
        f' height="{row_height}" />\n    </clipPath>\n'
        for n in missing
    )
    text = text.replace("</defs>", clips + "</defs>", 1)

    # The body group is pushed down by the window chrome; the viewBox Rich
    # computed does not leave room for the rows it forgot.
    top = sum(float(m) for m in re.findall(r'<g transform="translate\([\d.]+,\s*([\d.]+)\)"', text))
    needed = top + y0 + (max(used) + 1) * step + top / 3
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text)
    if float(box.group(2)) < needed:
        text = text.replace(box.group(0), f'viewBox="0 0 {box.group(1)} {needed:.1f}"', 1)
    return text


def find_chrome():
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def svg_to_png(chrome, svg_path, png_path):
    """Render an SVG with headless Chrome at roughly DEVICE_SCALE, then trim.

    The SVG is opened as the document rather than embedded in a wrapper page:
    an `<img>` drops the last rows of the terminal, the document does not.
    Chrome fits the document into the window with a margin of its own, so the
    window is oversized and the transparent border trimmed off afterwards.
    """
    text = svg_path.read_text()
    # Rich writes the terminal frame as a viewBox, not width/height attributes.
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text)
    width, height = float(box.group(1)), float(box.group(2))
    window = (int(width * DEVICE_SCALE * 1.1), int(height * DEVICE_SCALE * 1.1))
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [
                chrome, "--headless", "--disable-gpu", "--no-sandbox",
                "--hide-scrollbars", "--default-background-color=00000000",
                "--force-device-scale-factor=1",
                f"--window-size={window[0]},{window[1]}",
                f"--user-data-dir={profile}",
                f"--screenshot={png_path}",
                svg_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    finalize(png_path, crop=True)


def finalize(png_path, crop=False):
    """Trim Chrome's margin and shrink the file for a repository.

    A terminal frame uses a handful of colours, so a 256-entry palette is
    visually identical to the truecolour render at a third of the bytes.
    """
    if shutil.which("convert") is None:
        print(f"note: ImageMagick `convert` not found, {png_path.name} left as rendered")
        return
    subprocess.run(
        ["convert", str(png_path)]
        + (["-trim", "+repage"] if crop else [])
        + ["-strip", "-alpha", "off", "-colors", "256",
           "-define", "png:compression-level=9", str(png_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="scene names (default: all)")
    parser.add_argument("--out", default=str(ASSETS), help="output directory")
    parser.add_argument("--svg-only", action="store_true", help="skip the PNG step")
    args = parser.parse_args(argv)

    names = args.scenes or list(SCENES)
    unknown = [n for n in names if n not in SCENES]
    if unknown:
        parser.error(f"unknown scene(s): {', '.join(unknown)} (have: {', '.join(SCENES)})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    chrome = None if args.svg_only else find_chrome()
    if not args.svg_only and chrome is None:
        parser.error(f"no Chrome binary found (tried {', '.join(CHROME_CANDIDATES)})")

    sys.path.insert(0, str(ROOT / "src"))
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        os.environ["PATH"] = fake_path(tmpdir)
        os.environ["USER"] = "alice"
        os.environ["HOSTNAME"] = "login01"

        for name in names:
            # A fresh HOME per scene: the app persists pins and CPU readings
            # under it, so this keeps a stray real config out of the frame and
            # one scene's state out of the next one's.
            home = tmpdir / "home" / name
            home.mkdir(parents=True)
            os.environ["HOME"] = str(home)
            svg = (tmpdir if not args.svg_only else out) / f"tui-{name}.svg"
            asyncio.run(shoot(name, SCENES[name], svg))
            if args.svg_only:
                print(svg)
                continue
            png = out / f"tui-{name}.png"
            svg_to_png(chrome, svg, png)
            print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
