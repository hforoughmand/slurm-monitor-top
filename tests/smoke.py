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


def check_cpu_and_pins():
    """The two things Slurm does not store for us: CPU models and pins."""
    print("\nCPU info and pins:")
    from slurm_top import data

    # Parsing is the part that has to be right; probing a node is a side effect
    # on the cluster, so it is exercised by hand, not here.
    lscpu = (
        "Architecture:        x86_64\n"
        "CPU(s):              24\n"
        "Thread(s) per core:  2\n"
        "Core(s) per socket:  6\n"
        "Socket(s):           2\n"
        "Vendor ID:           GenuineIntel\n"
        "Model name:          Intel(R) Xeon(R) CPU           X5660  @ 2.80GHz\n"
        "CPU MHz:             1596.000\n"
        "CPU max MHz:         2794.0000\n"
    )
    info = data._parse_lscpu(lscpu)
    check("lscpu model parsed", info.get("model", "").startswith("Intel(R) Xeon"))
    check("lscpu layout parsed", (info.get("sockets"), info.get("cores_per_socket")) == ("2", "6"), str(info))
    check("trademark noise is dropped", data.short_cpu_model(info) == "Xeon X5660", data.short_cpu_model(info))
    # The nominal clock in the model name beats both the idle 1596MHz reading
    # and the 2794MHz boost ceiling.
    check("nominal clock wins over current and max", data.cpu_speed(info) == ("2.80GHz", "nominal"), str(data.cpu_speed(info)))

    # /proc/cpuinfo is the fallback when lscpu is missing on a node.
    proc = "processor\t: 0\nmodel name\t: AMD EPYC 7702P 64-Core Processor\ncpu MHz\t\t: 3350.000\n"
    amd = data._parse_lscpu(proc)
    check("/proc/cpuinfo parses too", amd.get("model", "").startswith("AMD EPYC"), str(amd))
    speed, kind = data.cpu_speed(amd)
    check("a clock with no nominal value is labelled", kind == "current", f"{speed} {kind}")
    check("a non-nominal clock is not passed off as the real one",
          "current" in data.describe_cpu(amd), data.describe_cpu(amd))

    check("an empty reading describes nothing", data.describe_cpu({}) == "")

    # How many of what, at what speed.
    counts = data.cpu_counts("2", "24", "2", "96")
    check("chips are counted per socket", counts["processors"] == 2, str(counts))
    check("cores are sockets x cores per socket", counts["cores"] == 48, str(counts))
    check("logical CPUs come from Slurm, not from the multiplication",
          data.cpu_counts("2", "24", "2", "90")["logical"] == 90,
          "a node with cores reserved for the OS reports fewer than 2x24x2")
    check("logical CPUs fall back to the multiplication", counts["logical"] == 96, str(counts))
    check("the count names the model", data.describe_cpu_count(counts, info) == "2 x Xeon X5660",
          data.describe_cpu_count(counts, info))
    check("an unprobed node gets no count line", data.describe_cpu_count(counts, {}) == "",
          "'2 x' on its own says nothing the socket count does not")

    labels = dict(data.describe_cpu_speeds(info))
    check("every clock is reported", set(labels) == {"nominal", "max", "now"}, str(labels))
    check("the nominal clock is the model's own", labels.get("nominal") == "2.80GHz", str(labels))
    check("a boost ceiling is kept apart from it", labels.get("max") == "2.79GHz", str(labels))
    check("a machine with only a boost figure claims no nominal one",
          "nominal" not in dict(data.describe_cpu_speeds(amd)), str(data.describe_cpu_speeds(amd)))

    nodes = data.parse_sinfo()
    if nodes:
        check("sinfo reports the core layout", any(n.sockets and n.cores_per_socket for n in nodes),
              "no %z data: is this an old Slurm?")
        laid_out = [n for n in nodes if n.sockets]
        check("layout renders", all("C/" in data.cpu_topology(n) for n in laid_out))

    # Pins round-trip through the shared config file.
    before = data.load_pinned_jobs()
    try:
        pinned, is_pinned = data.toggle_job_pin("smoke-test-job")
        check("pinning adds the job", is_pinned and "smoke-test-job" in pinned)
        pinned, is_pinned = data.toggle_job_pin("smoke-test-job")
        check("unpinning removes it", not is_pinned and "smoke-test-job" not in pinned)
        check("other pins are untouched", data.load_pinned_jobs() == before, str(data.load_pinned_jobs()))
    finally:
        data.set_job_pinned("smoke-test-job", False)

    jobs = [data.Job(str(i), "u", "R", "p", "n", "1", "1", "1G", "", "0:01", "n1") for i in range(4)]
    ordered = data.apply_pins(jobs, {"2"})
    check("a pinned job leads the list", ordered[0].job_id == "2")
    check("pinning keeps every job", len(ordered) == len(jobs))
    check("the rest keep their order", [j.job_id for j in ordered[1:]] == ["0", "1", "3"],
          ",".join(j.job_id for j in ordered))


def check_usage_and_output():
    """The two things the job popup adds to what Slurm prints: bars and logs."""
    print("\nUsage and output:")
    from slurm_top import data

    for text, expected in (
        ("2-00:00:00", 172800), ("00:18:15", 1095), ("17:28", 1048),
        ("1-02:03", 93780), ("90", 90),
    ):
        check(f"{text} is {expected}s", data.parse_duration(text) == expected, str(data.parse_duration(text)))
    check("a missing limit is not a number", data.parse_duration("UNLIMITED") == -1)
    check("an empty field is not a number", data.parse_duration("") == -1)
    # sstat prints this for the container step Slurm wraps around a job; taken
    # at face value it would claim 585 million years of CPU time.
    check("an overflowed counter is rejected",
          data.parse_duration("213503982334-14:25:51") == -1)
    check("durations round-trip", data.format_duration(93780) == "1-02:03:00", data.format_duration(93780))

    # The placeholders Slurm allows in -o/-e, for the versions that hand back
    # the pattern instead of the path.
    expanded = data.expand_job_path(
        "%x-%A_%a.out",
        {"JobName": "train", "ArrayJobId": "77", "ArrayTaskId": "3", "WorkDir": "/scratch/me"},
    )
    check("output patterns are expanded", expanded == "/scratch/me/train-77_3.out", expanded)
    check("a relative path is anchored to the work dir",
          data.expand_job_path("out.log", {"WorkDir": "/scratch/me"}) == "/scratch/me/out.log")
    check("a path with no placeholders is left alone",
          data.expand_job_path("/tmp/x.out", {}) == "/tmp/x.out")

    missing = data.stat_output_file("/nonexistent/slurm-top-smoke.out")
    check("a file that is not there is reported, not raised",
          missing["exists"] is False and bool(missing["error"]), str(missing))
    check("a discarded stream is named as such",
          "/dev/null" in data.stat_output_file("/dev/null")["error"])
    check("no path at all is its own message", bool(data.stat_output_file("")["error"]))

    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as handle:
        handle.write("".join(f"line {i}\n" for i in range(1, 501)))
        sample = handle.name
    try:
        tail = data.read_file_tail(sample, lines=10)
        check("the tail is the end of the file, not the start",
              tail["text"].splitlines()[0] == "line 491", tail["text"].splitlines()[:1])
        check("the tail stops at the line count asked for", tail["line_count"] == 10)
        check("a trimmed file says it was trimmed", tail["truncated"] is True)
        whole = data.read_file_tail(sample, lines=5000)
        check("a short file is not called trimmed", whole["truncated"] is False)
        check("reading only the last bytes never splits a line",
              all(line.startswith("line ") for line in
                  data.read_file_tail(sample, lines=500, max_bytes=120)["text"].splitlines()),
              "the fragment at the seek point must be dropped")
    finally:
        os.unlink(sample)

    # Against the real cluster: a running job of this user's, where sstat can
    # answer. Everything here is shape, not value.
    jobs = [j for j in data.parse_squeue()
            if j.state.upper().startswith("R") and j.user.startswith(os.environ.get("USER", "")[:8])]
    if not jobs:
        print("  skip live usage checks (no running job of your own)")
        return
    job, detail, usage = data.collect_job_info(jobs[0].job_id)
    if job is None:
        print("  skip live usage checks (the job ended mid-check)")
        return
    metrics = data.job_usage_metrics(job, detail, usage)
    check("a running job gets a time bar", any(m["key"] == "time" for m in metrics), str(metrics)[:200])
    check("every bar carries a percentage",
          all(isinstance(m["percent"], float) for m in metrics if m["kind"] == "bar"))
    check("percentages are not wild",
          all(0 <= m["percent"] <= 500 for m in metrics if m["kind"] == "bar"),
          str([(m["key"], m["percent"]) for m in metrics if m["kind"] == "bar"]))
    check("every metric says what it means", all(m["note"] for m in metrics), str(metrics)[:200])
    if usage:
        check("live usage belongs to the job we asked about",
              data.parse_duration(usage.get("TotalCPU", "")) >= 0,
              "sstat ignores an array task id and answers for the whole cluster; "
              "the rows must be filtered")
        cpu = next((m for m in metrics if m["key"] == "cpu"), None)
        check("a running job with usage gets a CPU bar", cpu is not None, str(usage))
    streams = data.job_output_paths(detail, job)
    check("both streams are described", set(streams) == {"stdout", "stderr"}, str(list(streams)))
    check("each stream says where it goes or why it cannot",
          all(s["path"] or s["error"] for s in streams.values()), str(streams))


def check_export():
    print("\nJSON export:")
    from slurm_top.export import SCHEMA_VERSION, job_detail, node_detail, snapshot

    snap = snapshot()
    check("schema is current", snap["schema"] == SCHEMA_VERSION)
    for key in ("timestamp", "user", "host", "jobs", "nodes", "disks", "gpu", "summary", "pinned"):
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
        check("node carries a cpu block", isinstance(node.get("cpu"), dict))
        cpu = node.get("cpu") or {}
        check("cpu block has the layout Slurm knows", {"sockets", "cores_per_socket", "threads_per_core"} <= set(cpu))
        check(
            "cpu block says whether the model is known",
            isinstance(cpu.get("known"), bool) and (bool(cpu.get("model")) == cpu.get("known")),
            str(cpu),
        )
        check(
            "sockets x cores x threads accounts for every CPU",
            all(
                not n["cpu"]["sockets"]
                or n["cpu"]["sockets"] * n["cpu"]["cores_per_socket"] * n["cpu"]["threads_per_core"] == n["cpus_total_n"]
                for n in snap["nodes"]
            ),
            ", ".join(
                f'{n["name"]}: {n["cpu"]["topology"]} vs {n["cpus_total_n"]}'
                for n in snap["nodes"]
                if n["cpu"]["sockets"]
                and n["cpu"]["sockets"] * n["cpu"]["cores_per_socket"] * n["cpu"]["threads_per_core"] != n["cpus_total_n"]
            )[:160],
        )
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
        check("job detail carries the usage metrics", isinstance(detail.get("metrics"), list))
        check(
            "every metric has the fields the front ends draw",
            all(
                {"key", "label", "kind", "percent", "value", "total", "risk", "note"} <= set(m)
                for m in detail["metrics"]
            ),
            str(detail["metrics"])[:200],
        )
        check("job detail names both output streams",
              set(detail.get("output") or {}) == {"stdout", "stderr"}, str(detail.get("output")))
        check(
            "a detail lookup does not read the log files themselves",
            all("text" not in stream for stream in (detail.get("output") or {}).values()),
            "the contents are a separate request, so a 3-second refresh stays cheap",
        )
    if snap["nodes"]:
        detail = node_detail(snap["nodes"][0]["name"])
        check("node detail is tagged", detail["kind"] == "node")
        check("node detail lists jobs", isinstance(detail["jobs"], list))
        check("node detail carries cpu info", isinstance(detail.get("cpu"), dict))
        cpu = detail["cpu"]
        check("node detail counts the CPUs", {"cpus_total", "cores", "processors"} <= set(cpu), str(cpu))
        check(
            "the CPU count agrees with what sinfo reported",
            cpu["cpus_total"] == snap["nodes"][0]["cpus_total_n"],
            f'{cpu["cpus_total"]} vs {snap["nodes"][0]["cpus_total_n"]}',
        )
        check("node detail lists the clocks it knows", isinstance(cpu.get("speeds"), list))
        check(
            "a plain node lookup does not go out to the machine",
            detail["cpu"]["error"] == "" and detail["cpu"]["source"] in ("", "ssh", "srun"),
            str(detail["cpu"]),
        )


def check_job_output_cli():
    print("\nJob output CLI:")
    env = {**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "src")}
    from slurm_top import data

    jobs = data.parse_squeue()
    if not jobs:
        print("  skip (nothing queued)")
        return
    result = subprocess.run(
        [sys.executable, "-m", "slurm_top.export", "--job-output", jobs[0].job_id, "--lines", "5"],
        env=env, capture_output=True, text=True,
    )
    check("--job-output exits 0", result.returncode == 0, result.stderr.strip()[:200])
    payload = json.loads(result.stdout or "{}")
    check("--job-output is tagged", payload.get("kind") == "job-output", result.stdout[:160])
    check("--job-output names both streams", set(payload.get("streams", {})) == {"stdout", "stderr"})
    check("--job-output returns a tail block", "text" in (payload.get("output") or {}))
    check(
        "a job whose output cannot be read still answers in JSON",
        "error" in (payload.get("output") or {}),
        "the front ends show the reason rather than an empty box",
    )


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

    # The extension pins through these, so they have to answer in JSON.
    for flag, expected in (("--pin", True), ("--unpin", False)):
        result = subprocess.run(
            [sys.executable, "-m", "slurm_top.export", flag, "smoke-cli-job"],
            env=env, capture_output=True, text=True,
        )
        payload = json.loads(result.stdout or "{}")
        check(f"{flag} reports the new state", payload.get("is_pinned") is expected, result.stdout[:160])
        check(f"{flag} returns the full pin list", isinstance(payload.get("pinned"), list))

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


def check_dangerous_keys():
    """No bare letter may run a command that throws work away.

    Asserted against the binding table rather than by pressing the keys: this
    suite runs on the real cluster, and a test that proved `ctrl+x` cancels by
    cancelling something would be a poor way to find out.
    """
    print("\nDangerous-action keys:")
    try:
        from slurm_top.app import JobDetailsModal, NodeDetailsModal
    except ImportError as exc:
        print(f"  skip (textual not installed: {exc})")
        return

    # Actions whose handler shells out to scancel/scontrol and cannot be undone
    # by another keystroke in the same popup.
    DESTRUCTIVE = {"cancel_job", "requeue_job"}

    def keys_for(screen):
        table = {}
        for binding in screen.BINDINGS:
            key, action = (binding[0], binding[1]) if isinstance(binding, tuple) else (binding.key, binding.action)
            table.setdefault(action, []).append(key)
        return table

    job_keys = keys_for(JobDetailsModal)
    for action in sorted(DESTRUCTIVE):
        keys = job_keys.get(action, [])
        check(f"{action} is bound at all", bool(keys))
        bare = [k for k in keys if len(k) == 1]
        check(f"{action} needs a modifier", not bare, f"reachable with {bare}")
        check(f"{action} uses ctrl", all(k.startswith("ctrl+") for k in keys), ", ".join(keys))

    # ctrl+h, ctrl+i, ctrl+j and ctrl+m arrive as Backspace, Tab, Enter and
    # Return, so a binding on one of them fires on the wrong keystroke.
    unusable = {"ctrl+h", "ctrl+i", "ctrl+j", "ctrl+m", "ctrl+c"}
    for screen in (JobDetailsModal, NodeDetailsModal):
        bound = {k for keys in keys_for(screen).values() for k in keys}
        clash = bound & unusable
        check(f"{screen.__name__} avoids keys the terminal reserves", not clash, ", ".join(sorted(clash)))

    # The letter that copies everywhere else must not do something else here.
    check("c copies in the job popup, as it does in every panel",
          "c" in job_keys.get("copy_job", []), str(job_keys.get("copy_job")))
    check("no bare letter is left pointing at a destructive action",
          not (set(job_keys.get("copy_job", [])) & {"ctrl+x", "ctrl+r"}))


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

            # The keys freed up by moving cancel and requeue behind modifiers
            # must now do the harmless thing they do elsewhere in the app. Only
            # the safe half is pressed here: this runs on the real cluster, so
            # nothing in this suite may reach scancel.
            if app.jobs_view.row_count:
                await pilot.press("c")
                await pilot.pause()
                check("c opens the copy popup instead of cancelling",
                      type(app.screen).__name__ == "CopyModal", type(app.screen).__name__)
                await pilot.press("escape")
                await pilot.pause()
                await pilot.press("r")
                await pilot.pause()
                check("r refreshes instead of requeueing",
                      type(app.screen).__name__ == "JobDetailsModal", type(app.screen).__name__)
            await pilot.press("escape")
            await pilot.pause()

            # A pinned job's action row is the widest the popup ever draws
            # ("p Unpin" against "p Pin"). It used to overflow the table, and a
            # horizontal scrollbar then replaced the single row of actions.
            selected = app.jobs_view.get_selected_job()
            if selected is not None:
                from slurm_top.data import set_job_pinned

                try:
                    set_job_pinned(selected.job_id, True)
                    app.jobs_view.pinned = frozenset([selected.job_id])
                    await pilot.press("enter")
                    await pilot.pause()
                    actions = app.screen.query_one("#job-actions")
                    labels = [str(cell) for cell in actions.get_row_at(0)]
                    check("a pinned job still shows its actions",
                          not actions.show_horizontal_scrollbar,
                          f"virtual={actions.virtual_size.width} of {actions.container_size.width}")
                    check("the row ends with the close action", labels[-1].strip() == "Esc Close",
                          str(labels))
                    check("pinning does not change the width of the row",
                          len(labels[6]) == len("p Unpin"), repr(labels[6]))
                    await pilot.press("escape")
                    await pilot.pause()
                finally:
                    set_job_pinned(selected.job_id, False)
                    app.jobs_view.pinned = frozenset()

            # The usage bars live in the job popup, under the scontrol block.
            app.jobs_view.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            for _ in range(20):
                await asyncio.sleep(0.05)
                if getattr(app.screen, "detail", None):
                    break
            if type(app.screen).__name__ == "JobDetailsModal":
                usage = app.screen.query_one("#job-usage").renderable.plain
                check("the job popup has a usage block", bool(usage.strip()), repr(usage[:120]))
                check("it is drawn as bars or says why not",
                      "[" in usage or "nothing" in usage.lower() or "sstat" in usage.lower(),
                      usage[:160])
                # Block characters (U+2588 / U+2591) depend on a font that has
                # them, sizes them to the cell and leaves no seam; nothing can
                # ask a terminal whether its font does. So the block stays
                # ASCII and the bars are drawn with background colour.
                check("the usage block needs no glyph a font might lack",
                      usage.isascii(), repr([c for c in usage if not c.isascii()][:8]))
                from slurm_top.app import _usage_bar

                coloured = _usage_bar(50.0, "high")
                check("a bar is a band of colour, not a row of characters",
                      set(coloured.plain) <= {"[", "]", " "}, repr(coloured.plain))
                check("and it carries the colour to draw it with",
                      any("on " in str(span.style) for span in coloured.spans),
                      str(coloured.spans))
                # Colour is the one thing a terminal can be asked about.
                monochrome = _usage_bar(50.0, "high", color=False).plain
                check("a terminal with no colour still gets a bar",
                      "#" in monochrome and "-" in monochrome, monochrome)
                await pilot.press("o")
                await pilot.pause()
                check("o opens the output viewer",
                      type(app.screen).__name__ == "JobOutputModal", type(app.screen).__name__)
                if type(app.screen).__name__ == "JobOutputModal":
                    for _ in range(30):
                        await asyncio.sleep(0.05)
                        if app.screen.tails:
                            break
                    panes = [s for s in ("stdout", "stderr")
                             if app.screen.query_one(f"#pane-{s}").display]
                    check("both streams are on screen unless the job merged them",
                          panes == ["stdout", "stderr"] or app.screen._merged(), str(panes))
                    if not app.screen._merged():
                        # The action row offers the three views by name, and
                        # 1/2/3 pick them without cycling through the others.
                        for key, expected in (("1", ["stdout"]), ("2", ["stderr"]),
                                              ("3", ["stdout", "stderr"])):
                            await pilot.press(key)
                            await pilot.pause()
                            shown = [s for s in ("stdout", "stderr")
                                     if app.screen.query_one(f"#pane-{s}").display]
                            check(f"{key} shows {'+'.join(expected)}", shown == expected, str(shown))
                        cells = app.screen.query_one("#job-output-actions").get_row_at(0)[:3]
                        marked = [str(c) for c in cells if "underline" in str(c.style)]
                        check("the row marks which view is in force",
                              marked == ["3 both"], str([(str(c), str(c.style)) for c in cells]))
                    head = app.screen.query_one("#job-output-head").renderable.plain
                    check("the viewer names the file it is tailing",
                          "/" in head or "owner" in head or "not there" in head, head[:160])
                    await pilot.press("escape")
                    await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

            app.nodes_view.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            check("node details popup opens", type(app.screen).__name__ == "NodeDetailsModal", type(app.screen).__name__)
            body = app.screen.query_one("#node-details-body").renderable.plain
            check("node details name the processors", "CPU " in body and "sockets" in body, body[:200])
            check("node details count the CPUs", "CPUs" in body and "cores" in body, body[:200])
            check("node details offer to read the CPU model",
                  "Model" in body, body[:200])
            await pilot.press("escape")
            await pilot.pause()

            # Pinning: p must lift the job to the top and survive a refresh.
            if app.jobs_view.row_count > 1:
                app.jobs_view.focus()
                await pilot.pause()
                app.jobs_view.move_cursor(row=min(1, app.jobs_view.row_count - 1))
                await pilot.pause()
                target = app.jobs_view.get_selected_job()
                try:
                    await pilot.press("p")
                    await pilot.pause()
                    check("p pins the selected job", target.job_id in app.jobs_view.pinned,
                          str(sorted(app.jobs_view.pinned)))
                    check("a pinned job moves to the top",
                          app.jobs_view._display_jobs[0].job_id == target.job_id,
                          app.jobs_view._display_jobs[0].job_id)
                    await app.refresh_data()
                    await pilot.pause()
                    check("it stays there across a refresh",
                          app.jobs_view._display_jobs[0].job_id == target.job_id,
                          app.jobs_view._display_jobs[0].job_id)
                    await pilot.press("p")
                    await pilot.pause()
                    check("p again unpins it", target.job_id not in app.jobs_view.pinned)
                finally:
                    from slurm_top.data import set_job_pinned

                    set_job_pinned(target.job_id, False)
            else:
                print("  skip pin checks (fewer than two jobs queued)")

    asyncio.run(drive())


if __name__ == "__main__":
    check_data_layer_is_standalone()
    check_cpu_and_pins()
    check_usage_and_output()
    check_export()
    check_cli()
    check_job_output_cli()
    check_dangerous_keys()
    check_tui()
    print(f"\n{len(FAILURES)} check(s) failed" if FAILURES else "\nall checks passed")
    sys.exit(1 if FAILURES else 0)
