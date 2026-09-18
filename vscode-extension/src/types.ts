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
  gpu: GpuStats;
  summary: Record<'all' | 'me' | 'others', SummaryBucket>;
  /** Present instead of the data fields when the collector hit a transient error. */
  error?: string;
}

export interface JobDetail {
  kind: 'job';
  job_id: string;
  job: Job | null;
  detail: Record<string, string>;
  usage: Record<string, string>;
}

export interface NodeDetail {
  kind: 'node';
  node: string;
  detail: Record<string, string>;
  jobs: Job[];
}

/** Extension host -> webview. */
export type HostMessage =
  | { type: 'snapshot'; snapshot: Snapshot }
  | { type: 'status'; state: 'starting' | 'running' | 'paused' | 'error'; message?: string }
  | { type: 'config'; sections: string[]; ownerFilter: string; interval: number; detailsIn: DetailPresentation }
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
  | { type: 'showLog' };
