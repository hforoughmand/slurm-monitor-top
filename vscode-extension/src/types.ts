/** Shapes emitted by `slurm-top --json` (see `src/slurm_top/export.py`). */

export const SUPPORTED_SCHEMA = 1;

/** Where job and node details are shown. */
export type DetailPresentation = 'window' | 'popup' | 'card' | 'tab' | 'overlay';

export interface Job {
  job_id: string;
  user: string;
  state: string;
  partition: string;
  name: string;
  nodes: string;
  ncpus: string;
  mem: string;
  gpus: string;
  time_used: string;
  node_list: string;
}

/**
 * What a node's processors are.
 *
 * The layout fields come from Slurm on every refresh. The model name and clock
 * do not exist anywhere in Slurm, so they stay empty (`known: false`) until
 * that node has been probed once; see `probe_node_cpu` in `data.py`.
 */
export interface CpuInfo {
  sockets: number;
  cores_per_socket: number;
  threads_per_core: number;
  /** `2 x 64C/2T`, or empty when Slurm did not report the layout. */
  topology?: string;
  /** Logical CPUs — what Slurm allocates, and what `nproc` reports. */
  cpus_total?: number;
  /** Physical cores: sockets × cores per socket. */
  cores?: number;
  /** Chips in the machine — one per socket. */
  processors?: number;
  /** `4 x Xeon E5-4650 v4`; empty until the node has been probed. */
  count_summary?: string;
  /** Every clock we have, each labelled `nominal`, `max` or `now`. */
  speeds?: { label: string; value: string }[];
  model: string;
  model_short: string;
  speed: string;
  /** `nominal` (from the model name), `max` (a boost ceiling) or `current`. */
  speed_kind: string;
  vendor: string;
  arch: string;
  /** `Xeon E5-4650 v4 @ 2.20GHz`, ready for a table cell. */
  summary: string;
  /** `ssh` or `srun` — how the reading was taken. */
  source: string;
  probed_at: number;
  known: boolean;
  /** Why the probe failed, on a detail that asked for one. */
  error?: string;
}

export interface Node {
  name: string;
  state: string;
  cpus_total: string;
  cpus_alloc: string;
  cpus_idle: string;
  mem_total: string;
  mem_reserved: string;
  mem_free: string;
  gres: string;
  partition: string;
  cpu_load: string;
  /** Why Slurm drained or downed the node; empty when it is healthy. */
  reason: string;
  gres_used: string;
  cpus_total_n: number;
  cpus_alloc_n: number;
  cpus_idle_n: number;
  mem_total_mb: number;
  mem_total_human: string;
  mem_reserved_mb: number;
  mem_reserved_human: string;
  mem_free_mb: number;
  mem_free_human: string;
  gpu_inventory: Record<string, number>;
  gpu_allocated: Record<string, number>;
  gpu_total: number;
  gpu_used: number;
  gpu_free: number;
  gpu_types: string[];
  cpu_load_n: number;
  /** Load divided by core count; may exceed 1 on an oversubscribed node. */
  cpu_load_ratio: number;
  sockets: string;
  cores_per_socket: string;
  threads_per_core: string;
  /** Missing when the collector predates CPU support. */
  cpu?: CpuInfo;
}

export interface DiskUsage {
  usage_percent: string;
  mount: string;
  fs_type: string;
  size: string;
  used: string;
  avail: string;
  usage_pct: number;
  size_mb: number;
  used_mb: number;
  avail_mb: number;
}

export interface SummaryCell {
  jobs: number;
  cpus: number;
  mem_mb: number;
  gpus: number;
}

export type SummaryBucket = Record<'running' | 'pending', SummaryCell>;

export interface GpuTypeStats {
  total: number;
  active: number;
  reserved: number;
  free_est: number;
}

export interface GpuStats {
  total: number;
  types_count: number;
  per_type: Record<string, number>;
  per_type_stats: Record<string, GpuTypeStats>;
  active: number;
  reserved: number;
  free_est: number;
}

export interface Snapshot {
  schema: number;
  timestamp: number;
  user: string;
  host: string;
  jobs: Job[];
  nodes: Node[];
  disks: DiskUsage[];
  /** Job ids the user pinned, shared with the terminal UI through its config. */
  pinned?: string[];
  gpu: GpuStats;
  summary: Record<'all' | 'me' | 'others', SummaryBucket>;
  /** Present instead of the data fields when the collector hit a transient error. */
  error?: string;
}

/**
 * One "asked for this much, using that much" figure.
 *
 * `bar` entries have a percentage worth drawing; `fact` entries are a number
 * nothing bounds (GPUs held, time queued); `note` entries explain why a bar is
 * missing — sstat answers only for your own running jobs.
 */
export interface UsageMetric {
  key: string;
  label: string;
  kind: 'bar' | 'fact' | 'note';
  percent: number | null;
  value: string;
  total: string;
  /** Which end of the scale is the bad one: `high` for limits, `low` for waste. */
  risk: 'high' | 'low' | '';
  note: string;
}

/** Where one of a job's two output streams goes, and what is there. */
export interface JobOutputFile {
  path: string;
  exists: boolean;
  size: number;
  /** Epoch seconds of the last write; 0 when the file is not there. */
  modified: number;
  /** Why there is nothing to read: no path, /dev/null, not created yet, ... */
  error: string;
  stream: 'stdout' | 'stderr';
  /** stderr was not given its own file, so it lands in the stdout one. */
  merged: boolean;
}

/** The tail itself, from `slurm-top --json --job-output`. */
export interface JobOutputTail extends JobOutputFile {
  text: string;
  line_count: number;
  /** The file is longer than what came back. */
  truncated: boolean;
}

export interface JobOutput {
  kind: 'job-output';
  job_id: string;
  state: string;
  streams: Record<'stdout' | 'stderr', JobOutputFile>;
  output: JobOutputTail;
}

export interface JobDetail {
  kind: 'job';
  job_id: string;
  job: Job | null;
  detail: Record<string, string>;
  usage: Record<string, string>;
  /** Missing when the collector predates request-versus-use reporting. */
  metrics?: UsageMetric[];
  /** Missing on that same older collector. */
  output?: Record<'stdout' | 'stderr', JobOutputFile>;
}

export interface NodeDetail {
  kind: 'node';
  node: string;
  detail: Record<string, string>;
  jobs: Job[];
  cpu?: CpuInfo;
}

/** Extension host -> webview. */
export type HostMessage =
  | { type: 'snapshot'; snapshot: Snapshot }
  | { type: 'status'; state: 'starting' | 'running' | 'paused' | 'error'; message?: string }
  | {
      type: 'config';
      sections: string[];
      ownerFilter: string;
      interval: number;
      detailsIn: DetailPresentation;
    }
  | { type: 'pinned'; pinned: string[] }
  | { type: 'cpuProbe'; node: string; state: 'started' | 'done'; message?: string }
  | { type: 'detailTarget'; kind: 'job' | 'node'; id: string }
  | { type: 'detail'; detail: JobDetail | NodeDetail }
  | { type: 'detailError'; message: string };

/** Webview -> extension host. */
export type ViewMessage =
  | { type: 'ready' }
  | { type: 'refresh' }
  | { type: 'openDashboard' }
  | { type: 'openDetail'; kind: 'job' | 'node'; id: string }
  | { type: 'refreshDetail' }
  | { type: 'closeDetail' }
  | { type: 'copy'; text: string; label?: string }
  | { type: 'togglePin'; jobId: string }
  | { type: 'openOutput'; jobId: string; stream: 'stdout' | 'stderr'; path: string; size: number }
  | { type: 'probeCpu'; node: string }
  | { type: 'showLog' };
