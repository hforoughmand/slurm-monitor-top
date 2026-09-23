import * as vscode from 'vscode';

/**
 * Which column each table shows.
 *
 * Every field the collector reports has a column; most tables show a useful
 * subset of them and the rest wait here until someone ticks them on the
 * settings page. The keys must match the column keys in `media/main.js` --
 * `test/columns.smoke.js` fails the build when the two drift apart, which is
 * the only thing keeping three separate lists (here, the webview, and the
 * schema in package.json) honest.
 */
export type ColumnSettings = Record<string, Record<string, boolean>>;

/** The sections with a configurable column set, and their default answer. */
export const COLUMN_DEFAULTS: ColumnSettings = {
  jobs: {
    job_id: true,
    user: true,
    state: true,
    partition: true,
    name: true,
    nodes: true,
    cpus: true,
    gpus: true,
    mem: true,
    time: true,
    node_list: true,
  },
  nodes: {
    name: true,
    state: true,
    partition: true,
    cpus: true,
    cpus_idle: true,
    cpus_total: false,
    load: true,
    mem_total: true,
    mem_free: true,
    mem_used: false,
    gpus: true,
    gpu_free: true,
    gres: true,
    reason: false,
    topology: false,
    cpu: true,
  },
  gpus: {
    node: true,
    type: true,
    used: true,
    free: true,
    mem_free: true,
    cpus_idle: true,
    state: true,
    partition: false,
    load: false,
  },
  disks: {
    usage: true,
    mount: true,
    size: true,
    used: true,
    avail: true,
    type: true,
  },
};

/** The settings key holding one section's columns: `nodes` -> `nodeColumns`. */
export function settingKey(section: string): string {
  const singular = section.replace(/s$/, '');
  return `${singular}Columns`;
}

export interface ColumnConfig {
  columns: ColumnSettings;
  /**
   * Sections the user has actually configured.
   *
   * The sidebar drops wide columns to fit, which is right until someone has
   * gone and ticked the ones they want: from then on their choice is the whole
   * answer and the narrow view stops editing it.
   */
  chosen: Record<string, boolean>;
}

export function readColumnSettings(): ColumnConfig {
  const settings = vscode.workspace.getConfiguration('slurmTop');
  const columns: ColumnSettings = {};
  const chosen: Record<string, boolean> = {};

  for (const section of Object.keys(COLUMN_DEFAULTS)) {
    const key = settingKey(section);
    const defaults = COLUMN_DEFAULTS[section];
    const configured = settings.get<Record<string, boolean>>(key, {}) ?? {};
    const merged: Record<string, boolean> = { ...defaults };
    for (const column of Object.keys(defaults)) {
      if (typeof configured[column] === 'boolean') {
        merged[column] = configured[column];
      }
    }
    columns[section] = merged;

    const seen = settings.inspect<Record<string, boolean>>(key);
    chosen[section] = Boolean(
      seen &&
        (seen.globalValue !== undefined ||
          seen.workspaceValue !== undefined ||
          seen.workspaceFolderValue !== undefined)
    );
  }

  return { columns, chosen };
}
