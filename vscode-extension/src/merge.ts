import {
  GpuStats,
  GpuTypeStats,
  MergedSnapshot,
  ServerState,
  ServerView,
  Snapshot,
  SummaryBucket,
  SummaryCell,
  SUPPORTED_SCHEMA,
} from './types';

/**
 * One cluster's latest answer, as the cluster client holds it.
 */
export interface ServerEntry {
  id: string;
  /** Configured label; empty means fall back to the reported hostname. */
  name: string;
  state: ServerState;
  message?: string;
  snapshot?: Snapshot;
}

const BUCKETS = ['all', 'me', 'others'] as const;
const PHASES = ['running', 'pending'] as const;

function emptyCell(): SummaryCell {
  return { jobs: 0, cpus: 0, mem_mb: 0, gpus: 0 };
}

export function emptySummary(): Record<'all' | 'me' | 'others', SummaryBucket> {
  const out = {} as Record<'all' | 'me' | 'others', SummaryBucket>;
  for (const bucket of BUCKETS) {
    out[bucket] = { running: emptyCell(), pending: emptyCell() };
  }
  return out;
}

export function emptyGpuStats(): GpuStats {
  return { total: 0, types_count: 0, per_type: {}, per_type_stats: {}, active: 0, reserved: 0, free_est: 0 };
}

function addCell(into: SummaryCell, from?: Partial<SummaryCell>): void {
  into.jobs += Number(from?.jobs) || 0;
  into.cpus += Number(from?.cpus) || 0;
  into.mem_mb += Number(from?.mem_mb) || 0;
  into.gpus += Number(from?.gpus) || 0;
}

/** Sum the per-bucket job statistics of several clusters. */
export function mergeSummaries(
  parts: (Record<string, SummaryBucket> | undefined)[]
): Record<'all' | 'me' | 'others', SummaryBucket> {
  const total = emptySummary();
  for (const part of parts) {
    for (const bucket of BUCKETS) {
      for (const phase of PHASES) {
        addCell(total[bucket][phase], (part?.[bucket] as SummaryBucket | undefined)?.[phase]);
      }
    }
  }
  return total;
}

/**
 * Sum GPU inventories across clusters.
 *
 * Per type rather than per (server, type): two clusters with A100s have, between
 * them, that many A100s, and "how many of these can I get right now" is the
 * question the panel exists to answer.
 */
export function mergeGpuStats(parts: (GpuStats | undefined)[]): GpuStats {
  const total = emptyGpuStats();
  for (const part of parts) {
    if (!part) {
      continue;
    }
    total.total += Number(part.total) || 0;
    total.active += Number(part.active) || 0;
    total.reserved += Number(part.reserved) || 0;
    total.free_est += Number(part.free_est) || 0;
    for (const [type, count] of Object.entries(part.per_type || {})) {
      total.per_type[type] = (total.per_type[type] || 0) + (Number(count) || 0);
    }
    for (const [type, stats] of Object.entries(part.per_type_stats || {})) {
      const into: GpuTypeStats = total.per_type_stats[type] || { total: 0, active: 0, reserved: 0, free_est: 0 };
      into.total += Number(stats.total) || 0;
      into.active += Number(stats.active) || 0;
      into.reserved += Number(stats.reserved) || 0;
      into.free_est += Number(stats.free_est) || 0;
      total.per_type_stats[type] = into;
    }
  }
  total.types_count = Object.keys(total.per_type_stats).length;
  return total;
}

/** What to show for a server: its label, else the hostname it reported. */
export function displayName(entry: ServerEntry): string {
  return entry.name || entry.snapshot?.host || entry.id;
}

function serverView(entry: ServerEntry): ServerView {
  const snapshot = entry.snapshot;
  return {
    id: entry.id,
    name: displayName(entry),
    host: snapshot?.host ?? '',
    user: snapshot?.user ?? '',
    state: entry.state,
    message: entry.message,
    timestamp: snapshot?.timestamp ?? 0,
    gpu: snapshot?.gpu ?? emptyGpuStats(),
    summary: (snapshot?.summary as Record<'all' | 'me' | 'others', SummaryBucket>) ?? emptySummary(),
    pinned: snapshot?.pinned ?? [],
    counts: {
      jobs: snapshot?.jobs?.length ?? 0,
      nodes: snapshot?.nodes?.length ?? 0,
      disks: snapshot?.disks?.length ?? 0,
    },
  };
}

/**
 * Every watched cluster in one snapshot.
 *
 * Rows are tagged rather than renamed: a job keeps its own id, and `server`
 * says which cluster it belongs to, so a click can be routed back to the
 * collector that knows about it. Job ids and node names are only unique within
 * one cluster, so nothing downstream may key on them alone.
 *
 * The top-level fields describe the first server, which makes a merged snapshot
 * a drop-in for the single-server one every part of the view used to be handed.
 */
export function mergeSnapshots(entries: ServerEntry[]): MergedSnapshot {
  const withData = entries.filter((entry) => entry.snapshot);
  const first = withData[0]?.snapshot;
  const merged: MergedSnapshot = {
    schema: first?.schema ?? SUPPORTED_SCHEMA,
    timestamp: Math.max(0, ...withData.map((entry) => entry.snapshot?.timestamp ?? 0)),
    user: first?.user ?? '',
    host: first?.host ?? '',
    servers: entries.map(serverView),
    jobs: [],
    nodes: [],
    disks: [],
    pinned: first?.pinned ?? [],
    gpu: mergeGpuStats(withData.map((entry) => entry.snapshot?.gpu)),
    summary: mergeSummaries(withData.map((entry) => entry.snapshot?.summary)),
  };

  for (const entry of entries) {
    const snapshot = entry.snapshot;
    if (!snapshot) {
      continue;
    }
    const tag = { server: entry.id, server_name: displayName(entry) };
    for (const job of snapshot.jobs || []) {
      merged.jobs.push({ ...job, ...tag });
    }
    for (const node of snapshot.nodes || []) {
      merged.nodes.push({ ...node, ...tag });
    }
    for (const disk of snapshot.disks || []) {
      merged.disks.push({ ...disk, ...tag });
    }
  }
  return merged;
}

/**
 * One state for the whole cluster set.
 *
 * A single sick server must be visible without drowning out the others, so it
 * only makes the aggregate an error when nothing is working; otherwise the
 * banner names it and the running servers keep updating.
 */
export function aggregateState(entries: ServerEntry[]): { state: ServerState; message?: string } {
  if (!entries.length) {
    return { state: 'paused', message: 'No Slurm servers are configured.' };
  }
  const broken = entries.filter((entry) => entry.state === 'error');
  if (broken.length === entries.length) {
    const first = broken[0];
    const label = entries.length > 1 ? `${broken.length} servers are failing. ${displayName(first)}: ` : '';
    return { state: 'error', message: `${label}${first.message ?? 'collector error'}` };
  }
  if (broken.length) {
    return {
      state: 'running',
      message: `${broken.map((entry) => displayName(entry)).join(', ')}: ${broken[0].message ?? 'collector error'}`,
    };
  }
  if (entries.some((entry) => entry.state === 'running')) {
    return { state: 'running' };
  }
  if (entries.some((entry) => entry.state === 'starting')) {
    return { state: 'starting' };
  }
  return { state: 'paused' };
}
