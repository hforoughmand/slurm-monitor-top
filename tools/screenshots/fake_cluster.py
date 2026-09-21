#!/usr/bin/env python3
"""A synthetic Slurm cluster, so screenshots never carry real cluster data.

The file doubles as five fake Slurm/coreutils commands. ``--install DIR`` drops
``squeue``, ``sinfo``, ``scontrol``, ``sstat`` and ``df`` into DIR as symlinks
back to this script; putting DIR first on ``PATH`` makes ``slurm-top`` collect
the cluster defined below instead of the real one. Dispatch is by ``argv[0]``,
exactly like busybox.

Only the invocations :mod:`slurm_top.data` actually makes are emulated, and
each one prints in the format that module parses:

    squeue -a -h --Format=jobid:|,...            all jobs, one per line
    squeue -a -h -j ID --Format=...              one job
    squeue -a -h -w NODE --Format=...            the jobs on one node
    sinfo -o '%n|%t|%c|%C|%m|%e|%G|%O|%P|%E'     node table, with a header line
    sinfo -h -N -O NodeHost:64,GresUsed:128      allocated GRES per node
    scontrol show job -d ID                      job detail block
    scontrol show node NAME                      node detail block
    sstat -a -P -n --format=JobID,... -j ID       live usage of a running job
    df -h --output=pcent,fstype,size,used,avail,target
    ssh NODE 'lscpu ...'                        CPU probe, route one
    srun --nodelist NODE ... sh -c 'lscpu ...'  CPU probe, route two

``ssh`` and ``srun`` are stubbed for the same reason as the rest: without them a
capture would open a real connection, or submit a real job, to whatever host
happens to answer to the name of a fake node.

Names, users and paths are invented. Any resemblance to a real cluster is the
point; any resemblance to a real *user* is not.
"""

import os
import sys

# The fake "me": screenshots are taken with USER set to this, so the owner
# filter and the Job statistics panel have something of their own to show.
ME = "alice"

# How much of each running job's reservation is actually doing something:
# the fraction of its cores that are busy, and of its memory that is resident.
# Deliberately uneven - a job at 6% of its cores is the one worth spotting.
CPU_BUSY = {
    "184213": 0.94, "184220": 0.71, "184231": 0.38, "184244": 0.06,
    "184247": 0.83, "184251": 0.22, "184258": 0.55, "184262": 0.97,
    "184266": 0.48, "184271": 0.12,
}
MEM_USED = {
    "184213": 0.62, "184220": 0.88, "184231": 0.41, "184244": 0.09,
    "184247": 0.74, "184251": 0.96, "184258": 0.33, "184262": 0.58,
    "184266": 0.21, "184271": 0.47,
}

# --------------------------------------------------------------------- jobs
# Fields mirror the squeue --Format order in slurm_top.data._SQUEUE_FORMAT:
# jobid, user, state, partition, name, nodes, cpus, mem, tres-alloc, time, nodelist.
JOBS = [
    ("184213", "alice", "RUNNING", "gpu", "train-resnet", "1", "16", "64G",
     "cpu=16,mem=64G,node=1,billing=16,gres/gpu:a100=2", "6:42:11", "gpu01"),
    ("184220", "bob", "RUNNING", "gpu", "finetune-llm", "1", "32", "256G",
     "cpu=32,mem=256G,node=1,billing=32,gres/gpu:h100=4", "1-03:18:55", "gpu03"),
    ("184231", "carol", "RUNNING", "cpu", "assemble-genome", "2", "128", "512G",
     "cpu=128,mem=512G,node=2,billing=128", "8:05:47", "cpu[01-02]"),
    ("184244", "alice", "RUNNING", "gpu", "hyperopt", "1", "8", "32G",
     "cpu=8,mem=32G,node=1,billing=8,gres/gpu:a100=1", "2:27:03", "gpu02"),
    ("184247", "dave", "RUNNING", "cpu", "align-reads", "1", "64", "192G",
     "cpu=64,mem=192G,node=1,billing=64", "4:51:22", "cpu03"),
    ("184251", "erin", "RUNNING", "bigmem", "variant-call", "1", "48", "1500G",
     "cpu=48,mem=1500G,node=1,billing=48", "12:14:09", "bigmem01"),
    ("184258", "bob", "RUNNING", "gpu", "embed-corpus", "1", "12", "48G",
     "cpu=12,mem=48G,node=1,billing=12,gres/gpu:v100=2", "3:33:41", "gpu05"),
    ("184262", "frank", "RUNNING", "cpu", "nextflow-main", "1", "4", "16G",
     "cpu=4,mem=16G,node=1,billing=4", "1-18:02:37", "cpu04"),
    ("184266", "alice", "RUNNING", "gpu", "eval-sweep", "1", "8", "64G",
     "cpu=8,mem=64G,node=1,billing=8,gres/gpu:h100=1", "0:47:12", "gpu03"),
    ("184270", "carol", "COMPLETING", "cpu", "postprocess", "1", "8", "32G",
     "cpu=8,mem=32G,node=1,billing=8", "0:22:58", "cpu03"),
    ("184274", "alice", "PENDING", "gpu", "train-resnet-b", "1", "16", "64G",
     "cpu=16,mem=64G,node=1,billing=16,gres/gpu:a100=2", "0:00", ""),
    ("184275", "dave", "PENDING", "gpu", "diffusion-ft", "2", "64", "512G",
     "cpu=64,mem=512G,node=2,billing=64,gres/gpu:h100=8", "0:00", ""),
    ("184276", "erin", "PENDING", "cpu", "blast-sweep", "1", "32", "96G",
     "cpu=32,mem=96G,node=1,billing=32", "0:00", ""),
    ("184277", "frank", "PENDING", "bigmem", "index-build", "1", "16", "900G",
     "cpu=16,mem=900G,node=1,billing=16", "0:00", ""),
    ("184278", "bob", "PENDING", "gpu", "ablation-3", "1", "8", "32G",
     "cpu=8,mem=32G,node=1,billing=8,gres/gpu:v100=1", "0:00", ""),
    ("184279", "alice", "PENDING", "cpu", "qc-report", "1", "2", "8G",
     "cpu=2,mem=8G,node=1,billing=2", "0:00", ""),
    ("184281", "carol", "RUNNING", "cpu", "bwa-index", "1", "16", "64G",
     "cpu=16,mem=64G,node=1,billing=16", "5:09:33", "cpu03"),
    ("184283", "dave", "RUNNING", "cpu", "salmon-quant", "1", "24", "72G",
     "cpu=24,mem=72G,node=1,billing=24", "2:02:18", "cpu03"),
    ("184285", "alice", "RUNNING", "cpu", "fastqc", "1", "8", "16G",
     "cpu=8,mem=16G,node=1,billing=8", "0:19:44", "cpu04"),
    ("184288", "erin", "RUNNING", "gpu", "segment-3d", "1", "16", "96G",
     "cpu=16,mem=96G,node=1,billing=16,gres/gpu:a100=2", "9:58:02", "gpu01"),
    ("184290", "frank", "RUNNING", "gpu", "protein-fold", "1", "8", "64G",
     "cpu=8,mem=64G,node=1,billing=8,gres/gpu:a100=1", "1-06:41:27", "gpu02"),
    ("184292", "bob", "RUNNING", "cpu", "merge-bams", "1", "8", "24G",
     "cpu=8,mem=24G,node=1,billing=8", "0:08:51", "cpu04"),
    ("184294", "carol", "PENDING", "gpu", "train-seg-v2", "1", "16", "128G",
     "cpu=16,mem=128G,node=1,billing=16,gres/gpu:h100=2", "0:00", ""),
    ("184296", "dave", "PENDING", "cpu", "kraken-classify", "1", "48", "256G",
     "cpu=48,mem=256G,node=1,billing=48", "0:00", ""),
    ("184298", "alice", "PENDING", "gpu", "sweep-lr-001", "1", "8", "32G",
     "cpu=8,mem=32G,node=1,billing=8,gres/gpu:v100=1", "0:00", ""),
    ("184299", "erin", "PENDING", "cpu", "annotate", "1", "4", "12G",
     "cpu=4,mem=12G,node=1,billing=4", "0:00", ""),
]

# Why each pending job is waiting, shown in the job details popup.
PENDING_REASON = {
    "184274": "Resources",
    "184294": "Resources",
    "184296": "Priority",
    "184298": "Priority",
    "184299": "Priority",
    "184275": "Resources",
    "184276": "Priority",
    "184277": "QOSMaxMemoryPerUser",
    "184278": "Priority",
    "184279": "Priority",
}

# -------------------------------------------------------------------- nodes
# name, state, cpus_total, "alloc/idle/other/total", mem_total_mb, mem_free_mb,
# gres, gres_used, load, partition, drain reason, sockets:cores:threads.
NODES = [
    ("gpu01", "mix", "64", "16/48/0/64", "515000", "449464", "gpu:a100:4(S:0-1)",
     "gpu:a100:2(IDX:0-1)", "15.62", "gpu*", "none", "2:16:2"),
    ("gpu02", "mix", "64", "8/56/0/64", "515000", "482232", "gpu:a100:4(S:0-1)",
     "gpu:a100:1(IDX:0)", "7.91", "gpu*", "none", "2:16:2"),
    ("gpu03", "mix", "96", "40/56/0/96", "1031000", "703512", "gpu:h100:8(S:0-1)",
     "gpu:h100:5(IDX:0-4)", "38.44", "gpu*", "none", "2:24:2"),
    ("gpu04", "idle", "96", "0/96/0/96", "1031000", "1031000", "gpu:h100:8(S:0-1)",
     "gpu:h100:0(IDX:N/A)", "0.03", "gpu*", "none", "2:24:2"),
    ("gpu05", "mix", "32", "12/20/0/32", "257000", "207848", "gpu:v100:4(S:0-1)",
     "gpu:v100:2(IDX:0-1)", "11.28", "gpu*", "none", "2:8:2"),
    ("cpu01", "alloc", "128", "128/0/0/128", "1018896", "756752", "(null)",
     "(null)", "127.42", "cpu", "none", "2:32:2"),
    ("cpu02", "alloc", "128", "128/0/0/128", "1018896", "756752", "(null)",
     "(null)", "126.88", "cpu", "none", "2:32:2"),
    ("cpu03", "mix", "128", "72/56/0/128", "1018896", "789488", "(null)",
     "(null)", "70.15", "cpu", "none", "2:32:2"),
    ("cpu04", "mix", "128", "4/124/0/128", "1018896", "1002512", "(null)",
     "(null)", "4.07", "cpu", "none", "2:32:2"),
    ("cpu05", "drain", "128", "0/0/128/128", "1018896", "1018896", "(null)",
     "(null)", "0.01", "cpu", "failed disk replacement", "2:32:2"),
    ("bigmem01", "mix", "96", "48/48/0/96", "4128000", "2592000", "(null)",
     "(null)", "47.33", "bigmem", "none", "4:12:2"),
]

# --------------------------------------------------------------------- CPUs
# What the CPU probe (`ssh`, else a one-second `srun`) finds on each node.
# Two shapes on purpose: an Intel model carries its nominal clock in the name,
# an AMD one does not, so the UI has to fall back to the boost ceiling and say
# so. Keyed by node prefix.
CPU_MODELS = {
    "gpu": ("AMD EPYC 7763 64-Core Processor", "3529.0520", "1500.0000"),
    "cpu": ("Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz", "3400.0000", "800.0000"),
    "bigmem": ("Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", "3200.0000", "800.0000"),
}

# -------------------------------------------------------------------- disks
# pcent, fstype, size, used, avail, target
DISKS = [
    ("6%", "xfs", "220G", "13G", "207G", "/"),
    ("91%", "nfs4", "20T", "18T", "2.0T", "/home"),
    ("68%", "lustre", "480T", "326T", "154T", "/scratch"),
    ("44%", "nfs4", "150T", "66T", "84T", "/projects"),
    ("12%", "xfs", "3.6T", "432G", "3.2T", "/local"),
    ("77%", "nfs4", "90T", "70T", "20T", "/archive"),
]

# ------------------------------------------------------------- second site
# A second, smaller cluster, so the multi-server view has two real ones to
# merge rather than the same one twice. Selected with `FAKE_CLUSTER=hpc2`;
# everything above describes `login01`.
#
# It shares the a100 with login01 on purpose -- that is what makes the merged
# GPU table add a model up across clusters -- and shares `alice`, so the owner
# filter has something of its own on both.
HPC2_JOBS = [
    ("90118", "alice", "RUNNING", "gpu", "segment-cells", "1", "12", "48G",
     "cpu=12,mem=48G,node=1,billing=12,gres/gpu:a100=1", "5:12:44", "n01"),
    ("90121", "greta", "RUNNING", "gpu", "render-volumes", "1", "48", "256G",
     "cpu=48,mem=256G,node=1,billing=48,gres/gpu:rtx6000=4", "19:38:02", "n02"),
    ("90124", "hugo", "RUNNING", "gpu", "track-particles", "1", "12", "32G",
     "cpu=12,mem=32G,node=1,billing=12", "2:04:19", "n01"),
    ("90131", "alice", "PENDING", "short", "qc-batch", "1", "16", "64G",
     "cpu=16,mem=64G,node=1,billing=16", "0:00", ""),
    ("90133", "greta", "PENDING", "gpu", "render-volumes-b", "1", "48", "256G",
     "cpu=48,mem=256G,node=1,billing=48,gres/gpu:rtx6000=4", "0:00", ""),
]

HPC2_NODES = [
    ("n01", "mix", "48", "24/24/0/48", "386000", "201324", "gpu:a100:2(S:0-1)",
     "gpu:a100:1(IDX:0)", "23.18", "gpu*", "none", "2:12:2"),
    ("n02", "alloc", "48", "48/0/0/48", "386000", "132880", "gpu:rtx6000:4(S:0-1)",
     "gpu:rtx6000:4(IDX:0-3)", "47.91", "gpu*", "none", "2:12:2"),
    ("n03", "idle", "64", "0/64/0/64", "515000", "515000", "(null)",
     "(null)", "0.02", "short", "none", "2:16:2"),
    ("n04", "drain", "64", "0/0/64/64", "515000", "515000", "(null)",
     "(null)", "0.00", "long", "awaiting a firmware update", "2:16:2"),
]

HPC2_CPU_MODELS = {
    "n": ("Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz", "4000.0000", "1000.0000"),
}

HPC2_DISKS = [
    ("31%", "xfs", "400G", "118G", "282G", "/"),
    ("58%", "nfs4", "8.0T", "4.5T", "3.5T", "/work"),
    ("83%", "ceph", "120T", "98T", "22T", "/data"),
]

HPC2_CPU_BUSY = {"90118": 0.81, "90121": 0.35, "90124": 0.09}
HPC2_MEM_USED = {"90118": 0.66, "90121": 0.52, "90124": 0.14}

if os.environ.get("FAKE_CLUSTER") == "hpc2":
    JOBS, NODES, DISKS = HPC2_JOBS, HPC2_NODES, HPC2_DISKS
    CPU_MODELS = HPC2_CPU_MODELS
    CPU_BUSY, MEM_USED = HPC2_CPU_BUSY, HPC2_MEM_USED


JOB_FIELDS = (
    "job_id user state partition name nodes ncpus mem tres time nodelist".split()
)


def job_map(row):
    return dict(zip(JOB_FIELDS, row))


def job_line(row):
    j = job_map(row)
    return "|".join(
        [j["job_id"], j["user"], j["state"], j["partition"], j["name"], j["nodes"],
         j["ncpus"], j["mem"], j["tres"], j["time"], j["nodelist"]]
    )


def node_names_of(nodelist):
    """Expand the small subset of Slurm hostlist syntax this fixture uses."""
    if not nodelist:
        return []
    if "[" not in nodelist:
        return [nodelist]
    prefix, _, rest = nodelist.partition("[")
    span = rest.rstrip("]")
    names = []
    for part in span.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            width = len(lo)
            names += [f"{prefix}{n:0{width}d}" for n in range(int(lo), int(hi) + 1)]
        else:
            names.append(f"{prefix}{part}")
    return names


def find_job(job_id):
    for row in JOBS:
        if row[0] == job_id:
            return row
    return None


def find_node(name):
    for row in NODES:
        if row[0] == name:
            return row
    return None


def arg_after(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag) and len(a) > len(flag) and not a[len(flag)].isalpha():
            return a[len(flag):]
    return None


# ------------------------------------------------------------------ squeue
def cmd_squeue(argv):
    job_id = arg_after(argv, "-j")
    node = arg_after(argv, "-w")
    rows = JOBS
    if job_id:
        rows = [r for r in JOBS if r[0] == job_id]
    elif node:
        rows = [r for r in JOBS if node in node_names_of(r[10])]
    for row in rows:
        print(job_line(row))
    return 0


# -------------------------------------------------------------------- sinfo
def cmd_sinfo(argv):
    joined = " ".join(argv)
    if "GresUsed" in joined:
        for row in NODES:
            print(f"{row[0]:<64}{row[7]}")
        return 0
    print("HOSTNAMES|STATE|CPUS|CPUS(A/I/O/T)|MEMORY|FREE_MEM|GRES|CPU_LOAD"
          "|PARTITION|S:C:T|REASON")
    for name, state, cpus, cstate, mem, free, gres, _used, load, part, reason, sct in NODES:
        print("|".join([name, state, cpus, cstate, mem, free, gres, load, part, sct, reason]))
    return 0


# ----------------------------------------------------------------- scontrol
def _scontrol_job(job_id):
    row = find_job(job_id)
    if row is None:
        print(f"slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
        return 1
    j = job_map(row)
    running = j["state"] == "RUNNING"
    nodes = node_names_of(j["nodelist"])
    stdout_path, stderr_path = log_paths(job_id)
    reason = PENDING_REASON.get(job_id, "None") if not running else "None"
    day = 17 + (int(job_id) % 2)
    submit = f"2026-09-{day:02d}T08:{int(job_id) % 60:02d}:11"
    start = f"2026-09-{day:02d}T09:{int(job_id) % 60:02d}:02" if running else "Unknown"
    print(
        f"JobId={job_id} JobName={j['name']}\n"
        f"   UserId={j['user']}(50{job_id[-2:]}) GroupId={j['user']}(50{job_id[-2:]}) MCS_label=N/A\n"
        f"   Priority={4000 + int(job_id) % 900} Nice=0 Account=research QOS=normal\n"
        f"   JobState={j['state']} Reason={reason} Dependency=(null)\n"
        f"   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0\n"
        f"   RunTime={j['time']} TimeLimit=2-00:00:00 TimeMin=N/A\n"
        f"   SubmitTime={submit} EligibleTime={submit}\n"
        f"   StartTime={start} EndTime=Unknown Deadline=N/A\n"
        f"   Partition={j['partition']} AllocNode:Sid=login01:24{job_id[-3:]}\n"
        f"   ReqNodeList=(null) ExcNodeList=(null)\n"
        f"   NodeList={j['nodelist'] or '(null)'} BatchHost={nodes[0] if nodes else '(null)'}\n"
        f"   NumNodes={j['nodes']} NumCPUs={j['ncpus']} NumTasks={j['nodes']} CPUs/Task=1\n"
        f"   TRES={j['tres']}\n"
        f"   {'AllocTRES' if running else 'ReqTRES'}={j['tres']}\n"
        f"   MinCPUsNode=1 MinMemoryNode={j['mem']} MinTmpDiskNode=0\n"
        f"   Features=(null) Gres=(null) Reservation=(null)\n"
        f"   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)\n"
        f"   Command=/home/{j['user']}/projects/pipeline/scripts/{j['name']}.sbatch\n"
        f"   WorkDir=/scratch/{j['user']}/runs/{j['name']}\n"
        f"   StdErr={stderr_path}\n"
        f"   StdIn=/dev/null\n"
        f"   StdOut={stdout_path}\n"
        f"   Power=\n"
    )
    return 0


def _scontrol_node(name):
    row = find_node(name)
    if row is None:
        print(f"Node {name} not found", file=sys.stderr)
        return 1
    (nm, state, cpus, cstate, mem, free, gres, used, load, part, reason, sct) = row
    alloc = cstate.split("/")[0]
    alloc_mem = int(mem) - int(free)
    sockets, cores_per_socket, threads_per_core = (int(v) for v in sct.split(":"))
    gres_line = gres if gres != "(null)" else "(null)"
    reason_line = (
        f"   Reason={reason} [slurm@2026-09-16T11:04:55]\n" if reason != "none" else ""
    )
    alloc_tres = f"cpu={alloc},mem={alloc_mem}M,billing={alloc}"
    if used != "(null)" and not used.endswith("0(IDX:N/A)"):
        alloc_tres += f",gres/gpu={used.split(':')[2].split('(')[0]}"
    print(
        f"NodeName={nm} Arch=x86_64 CoresPerSocket={cores_per_socket}\n"
        f"   CPUAlloc={alloc} CPUEfficiency=0.00% CPUTot={cpus} CPULoad={load}\n"
        f"   AvailableFeatures={'gpu,avx512' if gres_line != '(null)' else 'avx512'}\n"
        f"   ActiveFeatures={'gpu,avx512' if gres_line != '(null)' else 'avx512'}\n"
        f"   Gres={gres_line}\n"
        f"   GresUsed={used}\n"
        f"   NodeAddr={nm} NodeHostName={nm} Version=23.11.6\n"
        f"   OS=Linux 5.14.0-427.el9.x86_64\n"
        f"   RealMemory={mem} AllocMem={alloc_mem} FreeMem={free} Sockets={sockets} Boards=1\n"
        f"   State={state.upper()} ThreadsPerCore={threads_per_core} TmpDisk=3600000"
        f" Weight=1 Owner=N/A MCS_label=N/A\n"
        f"   Partitions={part.rstrip('*')}\n"
        f"   BootTime=2026-08-30T03:12:40 SlurmdStartTime=2026-08-30T03:16:02\n"
        f"   LastBusyTime=2026-09-18T07:41:19 ResumeAfterTime=None\n"
        f"   CfgTRES=cpu={cpus},mem={mem}M,billing={cpus}\n"
        f"   AllocTRES={alloc_tres}\n"
        f"   CapWatts=n/a\n"
        f"   CurrentWatts=0 AveWatts=0\n"
        f"   ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a\n"
        f"{reason_line}"
    )
    return 0


def cmd_scontrol(argv):
    args = [a for a in argv if not a.startswith("-")]
    if len(args) >= 3 and args[0] == "show" and args[1] == "job":
        return _scontrol_job(args[2])
    if len(args) >= 3 and args[0] == "show" and args[1] == "node":
        return _scontrol_node(args[2])
    return 1


# -------------------------------------------------------------------- sstat
def _seconds(elapsed):
    """`1-03:18:55` -> seconds, for working out how much CPU time to invent."""
    days, _, rest = elapsed.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _hms(seconds):
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    head = f"{days}-" if days else ""
    return f"{head}{hours:02d}:{minutes:02d}:{secs:02d}"


def cmd_sstat(argv):
    """JobID|MaxRSS|MaxVMSize|AveCPU|NTasks|TRESUsageInTot, one row per step.

    The `.extern` row is the container step Slurm wraps around every job. It
    does no work and its AveCPU is the famous overflowed counter; it is here
    because `fetch_job_usage` has to go on dropping it.
    """
    job_id = arg_after(argv, "-j")
    row = find_job(job_id or "")
    if row is None or row[2] != "RUNNING":
        return 1
    j = job_map(row)
    cpus = int(j["ncpus"])
    req_gb = int(j["mem"].rstrip("G"))
    # Each job gets its own efficiency, so the bars are not all the same
    # length: some of these reservations are being wasted, which is the point.
    busy = CPU_BUSY.get(job_id, 0.5)
    cpu_time = _hms(_seconds(j["time"]) * cpus * busy)
    rss = f"{req_gb * MEM_USED.get(job_id, 0.5):.2f}G"
    vmem = f"{req_gb * MEM_USED.get(job_id, 0.5) * 1.2:.2f}G"
    print(f"{job_id}.extern|||213503982334-14:25:51|1|energy=0")
    print(f"{job_id}.batch|{rss}|{vmem}|{cpu_time}|1|cpu={cpu_time},energy=0,mem={rss},vmem={vmem}")
    return 0


# ------------------------------------------------------------------ ssh/srun
def _lscpu(node_name):
    """`lscpu` output for one node, in the shape :func:`_parse_lscpu` reads."""
    row = find_node(node_name)
    if row is None:
        return None
    sockets, cores, threads = (int(v) for v in row[11].split(":"))
    prefix = "".join(ch for ch in node_name if not ch.isdigit())
    model, max_mhz, min_mhz = CPU_MODELS[prefix]
    return (
        f"Architecture:        x86_64\n"
        f"CPU op-mode(s):      32-bit, 64-bit\n"
        f"Byte Order:          Little Endian\n"
        f"CPU(s):              {row[2]}\n"
        f"Thread(s) per core:  {threads}\n"
        f"Core(s) per socket:  {cores}\n"
        f"Socket(s):           {sockets}\n"
        f"NUMA node(s):        {sockets}\n"
        f"Vendor ID:           {'AuthenticAMD' if 'AMD' in model else 'GenuineIntel'}\n"
        f"Model name:          {model}\n"
        f"CPU MHz:             1795.4210\n"
        f"CPU max MHz:         {max_mhz}\n"
        f"CPU min MHz:         {min_mhz}\n"
        f"L3 cache:            32768K\n"
    )


def cmd_ssh(argv):
    """The probe's first route. Options come in `-o key=value` pairs."""
    rest = list(argv)
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag == "-o" and rest:
            rest.pop(0)
    if not rest:
        return 255
    out = _lscpu(rest[0])
    if out is None:
        print(f"ssh: Could not resolve hostname {rest[0]}", file=sys.stderr)
        return 255
    sys.stdout.write(out)
    return 0


def cmd_srun(argv):
    """The probe's fallback route, used when ssh is refused."""
    node = arg_after(argv, "--nodelist")
    out = _lscpu(node or "")
    if out is None:
        print("srun: error: Unable to allocate resources: Invalid node name specified",
              file=sys.stderr)
        return 1
    sys.stdout.write(out)
    return 0


# ----------------------------------------------------------------------- df
def cmd_df(argv):
    print("Use% Type   Size  Used Avail Mounted on")
    for pcent, fstype, size, used, avail, target in DISKS:
        print(f"{pcent:>4} {fstype:<6} {size:>5} {used:>5} {avail:>5} {target}")
    return 0


# ------------------------------------------------------------------- output
# A job's stdout and stderr are ordinary files, so the fake cluster writes
# real ones next to its fake commands and points scontrol at them. Without
# this the output viewer would have nothing to show but "not there yet".

LOG_STDOUT = """\
[{t0}] {name}: starting on {node}, {ncpus} cpus, {mem} requested
[{t0}] {name}: loading dataset shard 0/8
[{t1}] {name}: epoch 1/40  loss 2.9134  lr 3.0e-04  {ncpus} workers
[{t1}] {name}: epoch 2/40  loss 2.1077  lr 3.0e-04
[{t2}] {name}: epoch 3/40  loss 1.8420  lr 3.0e-04
[{t2}] {name}: checkpoint written to checkpoints/epoch-003.pt
[{t3}] {name}: epoch 4/40  loss 1.6688  lr 2.7e-04
[{t3}] {name}: epoch 5/40  loss 1.5502  lr 2.7e-04
[{t4}] {name}: validation  acc 0.7412  macro-f1 0.7188
[{t4}] {name}: epoch 6/40  loss 1.4731  lr 2.7e-04
"""

LOG_STDERR = """\
[{t1}] warning: NCCL falling back to the socket transport (no IB device found)
[{t2}] warning: dataloader worker 3 was slow for 12.4s; check the shard layout
[{t4}] warning: 2 batches skipped after an out-of-range label
"""


def log_dir():
    """Where the fake logs live: `logs/` beside the directory of fake commands."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), "logs")


def log_paths(job_id):
    base = os.path.join(log_dir(), job_id)
    return f"{base}.out", f"{base}.err"


def write_logs(directory):
    """One .out and one .err per running job, with plausible contents."""
    os.makedirs(directory, exist_ok=True)
    for row in JOBS:
        j = job_map(row)
        if j["state"] != "RUNNING":
            continue
        stamps = {f"t{i}": f"09:{(int(row[0]) + i * 7) % 60:02d}:1{i}" for i in range(5)}
        node = node_names_of(j["nodelist"])
        fields = dict(
            stamps, name=j["name"], node=node[0] if node else "?",
            ncpus=j["ncpus"], mem=j["mem"],
        )
        out, err = os.path.join(directory, f"{row[0]}.out"), os.path.join(directory, f"{row[0]}.err")
        with open(out, "w") as handle:
            handle.write(LOG_STDOUT.format(**fields))
        with open(err, "w") as handle:
            handle.write(LOG_STDERR.format(**fields))
    return directory


COMMANDS = {
    "squeue": cmd_squeue,
    "sinfo": cmd_sinfo,
    "scontrol": cmd_scontrol,
    "sstat": cmd_sstat,
    "ssh": cmd_ssh,
    "srun": cmd_srun,
    "df": cmd_df,
}


def install(directory):
    """Expose this file as the commands slurm_top.data shells out to.

    Also writes the fake jobs' output files one level up, where `log_paths`
    (and so `scontrol`) looks for them.
    """
    os.makedirs(directory, exist_ok=True)
    write_logs(os.path.join(os.path.dirname(os.path.abspath(directory)), "logs"))
    target = os.path.abspath(__file__)
    for name in COMMANDS:
        link = os.path.join(directory, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
        os.chmod(target, 0o755)
    return directory


def main(argv):
    name = os.path.basename(argv[0])
    if name in COMMANDS:
        return COMMANDS[name](argv[1:])
    if len(argv) > 2 and argv[1] == "--install":
        print(install(argv[2]))
        return 0
    sys.stderr.write(__doc__ + "\nusage: fake_cluster.py --install DIR\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
