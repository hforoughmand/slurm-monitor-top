"""Smoke checks for both front ends. Run: python tests/smoke.py

Talks to the real cluster, so it needs `squeue`/`sinfo` on PATH. It asserts
shapes and invariants rather than specific numbers, which change every minute.

The TUI part needs `textual`; it is skipped when that is missing, because the
data layer and the JSON exporter are meant to work without it.
"""

import asyncio
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL {label}{f' ({detail})' if detail else ''}")


def check_data_layer_is_standalone():
    print("data layer:")
    import ast

    tree = ast.parse(open(os.path.join(os.path.dirname(__file__), "..", "src", "slurm_top", "data.py")).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = imported & {"rich", "textual"}
    check("data.py imports neither rich nor textual", not forbidden, ", ".join(sorted(forbidden)))

    # A subprocess with the package dir hidden from an installed copy proves the
    # module really has no third-party imports, not just none that are unused.
    code = "import slurm_top.data as d; print(len(d.parse_sinfo()))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")},
        capture_output=True,
        text=True,
    )
    check("data.py imports in a bare interpreter", result.returncode == 0, result.stderr.strip()[:200])


def check_export():
    print("\nJSON export:")
    from slurm_top.export import SCHEMA_VERSION, job_detail, node_detail, snapshot

    snap = snapshot()
    check("schema is current", snap["schema"] == SCHEMA_VERSION)
    for key in ("timestamp", "user", "host", "jobs", "nodes", "disks", "gpu", "summary"):
        check(f"snapshot has {key}", key in snap)
    check("snapshot is JSON-serializable", isinstance(json.dumps(snap), str))
    check("nodes were found", len(snap["nodes"]) > 0, "no nodes: is sinfo working?")

    for job in snap["jobs"]:
        check_job_fields(job)
        break
    else:
        print("  skip job field checks (no jobs queued)")

    if snap["nodes"]:
        node = snap["nodes"][0]
        for key in (
            "cpus_total_n", "cpus_alloc_n", "mem_total_mb", "mem_free_human", "gpu_inventory",
            "partition", "cpu_load_n", "cpu_load_ratio", "reason",
            "gpu_total", "gpu_used", "gpu_free", "gpu_types",
        ):
            check(f"node has derived {key}", key in node)
        check("node cpu counts are integers", isinstance(node["cpus_total_n"], int))
        check("node load is a number", isinstance(node["cpu_load_n"], float))
        check("machines report a partition", any(n["partition"] for n in snap["nodes"]))
        check(
            "allocated GPUs never exceed installed ones",
            all(n["gpu_used"] <= n["gpu_total"] for n in snap["nodes"]),
        )
        check(
            "per-node GPU totals match the cluster total",
            sum(n["gpu_total"] for n in snap["nodes"]) == snap["gpu"]["total"],
            f'{sum(n["gpu_total"] for n in snap["nodes"])} vs {snap["gpu"]["total"]}',
        )

    if snap["disks"]:
        disk = snap["disks"][0]
        for key in ("used", "avail", "usage_pct", "size_mb", "used_mb", "avail_mb"):
            check(f"disk has {key}", key in disk)
        check("disk usage is an int percentage", isinstance(disk["usage_pct"], int) and 0 <= disk["usage_pct"] <= 100)
        check("disk sizes parsed to numbers", all(isinstance(d["size_mb"], int) for d in snap["disks"]))
        # df rounds each column independently, so used + avail only has to land
        # near size, not on it.
        sized = [d for d in snap["disks"] if d["size_mb"] > 1024]
        check(
            "used + avail is close to size",
            all(abs(d["used_mb"] + d["avail_mb"] - d["size_mb"]) <= 0.15 * d["size_mb"] for d in sized),
        )
        check(
            "mount paths look like paths",
            all(d["mount"].startswith("/") for d in snap["disks"]),
            ", ".join(d["mount"] for d in snap["disks"] if not d["mount"].startswith("/"))[:120],
        )

    gpu = snap["gpu"]
    check("gpu totals agree with per-type inventory", gpu["total"] == sum(gpu["per_type"].values()))
    check("free GPUs never exceed the total", gpu["free_est"] <= gpu["total"])

    summary = snap["summary"]
    check("summary buckets present", set(summary) == {"all", "me", "others"})
    check(
        "all == me + others for running jobs",
        summary["all"]["running"]["jobs"] == summary["me"]["running"]["jobs"] + summary["others"]["running"]["jobs"],
    )

    if snap["jobs"]:
        detail = job_detail(snap["jobs"][0]["job_id"])
        check("job detail is tagged", detail["kind"] == "job")
        check("job detail is JSON-serializable", isinstance(json.dumps(detail), str))
    if snap["nodes"]:
        detail = node_detail(snap["nodes"][0]["name"])
        check("node detail is tagged", detail["kind"] == "node")
        check("node detail lists jobs", isinstance(detail["jobs"], list))


def check_job_fields(job):
    for key in ("job_id", "user", "state", "gpu_count", "gpu_types", "cpu_count", "mem_mb"):
        check(f"job has {key}", key in job)
    check("job gpu_count is an int", isinstance(job["gpu_count"], int))
    check("job gpu_count matches gpu_types", job["gpu_count"] == sum(job["gpu_types"].values()) or not job["gpu_types"])


def check_cli():
    print("\nCLI:")
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")}
    result = subprocess.run([sys.executable, "-m", "slurm_top.export", "--help"], env=env, capture_output=True, text=True)
    check("--help exits 0", result.returncode == 0)
    check("--help mentions --watch", "--watch" in result.stdout, "the extension probes for this string")

    result = subprocess.run([sys.executable, "-m", "slurm_top.export"], env=env, capture_output=True, text=True)
    check("one-shot prints one JSON line", result.returncode == 0 and len(result.stdout.strip().splitlines()) == 1)

    watch = subprocess.Popen(
        [sys.executable, "-m", "slurm_top.export", "--watch", "1"],
        env=env, stdout=subprocess.PIPE, text=True,
    )
    try:
        lines = [json.loads(watch.stdout.readline()) for _ in range(2)]
        check("--watch streams successive snapshots", lines[1]["timestamp"] > lines[0]["timestamp"])
    finally:
        watch.kill()
        watch.wait()


def check_tui():
    print("\nTUI:")
    try:
        from slurm_top.app import SlurmHtop
    except ImportError as exc:
        print(f"  skip (textual not installed: {exc})")
        return

    async def drive():
        app = SlurmHtop()
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            await app.refresh_data()
            await pilot.pause()
            check("jobs table populated", app.jobs_view.row_count >= 0)
            check("nodes table populated", app.nodes_view.row_count > 0)
            check("disks table populated", app.disk_usage_view.row_count > 0)
            check("summary computed", "all" in (app.summary_bar.summary or {}))
            # The df/sinfo parsers grew columns for the extension; the TUI's
            # own columns must still line up with the values it reads.
            fields = dict(app.disk_usage_view.row_fields())
            check("disk row keeps its usage column", fields.get("USAGE", "").endswith("%"), str(fields))
            check("disk row keeps its mount path", str(fields.get("PATH", "")).startswith("/"), str(fields))

            app.jobs_view.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            check("job details popup opens", type(app.screen).__name__ == "JobDetailsModal", type(app.screen).__name__)
            await pilot.press("escape")
            await pilot.pause()

            app.nodes_view.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            check("node details popup opens", type(app.screen).__name__ == "NodeDetailsModal", type(app.screen).__name__)
            await pilot.press("escape")
            await pilot.pause()

    asyncio.run(drive())


if __name__ == "__main__":
    check_data_layer_is_standalone()
    check_export()
    check_cli()
    check_tui()
    print(f"\n{len(FAILURES)} check(s) failed" if FAILURES else "\nall checks passed")
    sys.exit(1 if FAILURES else 0)
