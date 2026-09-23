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
}

/**
 * What the user actually said, not what they said plus every default.
 *
 * Only the columns a person has set are sent, because the webview treats a
 * setting as an instruction that outranks the narrow view's own trimming: tick
 * `user` and it appears in the sidebar, where width would otherwise have
 * dropped it. Sending the merged defaults too would make every default-on
 * column an instruction as well, and asking for one column would drag the
 * whole table into a 300px panel.
 */
export function readColumnSettings(): ColumnConfig {
  const settings = vscode.workspace.getConfiguration('slurmTop');
  const columns: ColumnSettings = {};

  for (const section of Object.keys(COLUMN_DEFAULTS)) {
    const defaults = COLUMN_DEFAULTS[section];
    const seen = settings.inspect<Record<string, boolean>>(settingKey(section));
    // Folder over workspace over user, the order VS Code resolves them in.
    const configured = {
      ...(seen?.globalValue ?? {}),
      ...(seen?.workspaceValue ?? {}),
      ...(seen?.workspaceFolderValue ?? {}),
    };
    const chosen: Record<string, boolean> = {};
    for (const column of Object.keys(defaults)) {
      if (typeof configured[column] === 'boolean') {
        chosen[column] = configured[column];
      }
    }
    columns[section] = chosen;
  }

  return { columns };
}
