import * as vscode from 'vscode';

import { SlurmClient } from './client';
import { Job, JobDetail, NodeDetail } from './types';

/**
 * Job and node details as a floating quick pick.
 *
 * A webview always lives in an editor tab, so neither the dashboard panel nor
 * the "modal" card can avoid one. The quick pick is the only VS Code surface
 * that both floats free of the editor and scrolls, which a job's ~50 scontrol
 * fields need; the native modal dialog is a fixed-size block of text.
 *
 * What it gives up against the tab: no bars, no tables, no live refresh. In
 * exchange you can type to filter the fields and hit Enter to copy one, which
 * is what the terminal UI's copy popup does.
 */

interface Row {
  label: string;
  value: string;
  section: string;
  /** Set on a row that opens another job or node instead of copying. */
  jump?: { kind: 'job' | 'node'; id: string };
}

const REFRESH_BUTTON: vscode.QuickInputButton = {
  iconPath: new vscode.ThemeIcon('refresh'),
  tooltip: 'Refresh',
};

const OPEN_TAB_BUTTON: vscode.QuickInputButton = {
  iconPath: new vscode.ThemeIcon('link-external'),
  tooltip: 'Open in an editor tab',
};

const COPY_ALL_BUTTON: vscode.QuickInputButton = {
  iconPath: new vscode.ThemeIcon('copy'),
  tooltip: 'Copy every field',
};

function jobRows(payload: JobDetail): Row[] {
  const rows: Row[] = [];
  if (payload.job) {
    for (const [key, value] of Object.entries(payload.job)) {
      if (value !== null && typeof value !== 'object') {
        rows.push({ label: key, value: String(value), section: 'squeue' });
      }
    }
  }
  for (const [key, value] of Object.entries(payload.detail ?? {})) {
    rows.push({ label: key, value: String(value), section: 'scontrol' });
  }
  for (const [key, value] of Object.entries(payload.usage ?? {})) {
    rows.push({ label: key, value: String(value), section: 'usage' });
  }
  return rows;
}

function nodeRows(payload: NodeDetail): Row[] {
  const rows: Row[] = [];
  for (const [key, value] of Object.entries(payload.detail ?? {})) {
    rows.push({ label: key, value: String(value), section: 'scontrol' });
  }
  for (const job of payload.jobs ?? []) {
    rows.push({
      label: job.job_id,
      value: `${job.user}  ${job.state}  ${job.ncpus} cpu  ${job.mem}  ${job.name}  ${job.time_used}`,
      section: `jobs on ${payload.node}`,
      jump: { kind: 'job', id: job.job_id },
    });
  }
  return rows;
}

function toItems(rows: Row[]): (vscode.QuickPickItem & { row?: Row })[] {
  const items: (vscode.QuickPickItem & { row?: Row })[] = [];
  let section: string | undefined;
  for (const row of rows) {
    if (row.section !== section) {
      section = row.section;
      items.push({ label: section, kind: vscode.QuickPickItemKind.Separator });
    }
    items.push({
      label: row.label,
      description: row.value,
      detail: row.jump ? 'Enter to open this job' : undefined,
      row,
    });
  }
  return items;
}

function headline(payload: JobDetail | NodeDetail): string {
  if (payload.kind === 'job') {
    const job: Job | null = payload.job;
    return job ? `Job ${payload.job_id} — ${job.state} — ${job.name}` : `Job ${payload.job_id}`;
  }
  return `Node ${payload.node} — ${(payload.jobs ?? []).length} job(s)`;
}

/**
 * Open the details of one job or node as a quick pick.
 *
 * Refresh is a button rather than a timer: re-filling the list under the
 * cursor while you are typing a filter would move the selection out from under
 * you every few seconds.
 */
export async function showDetailQuickPick(
  client: SlurmClient,
  log: vscode.OutputChannel,
  target: { kind: 'job' | 'node'; id: string },
  openInTab: (target: { kind: 'job' | 'node'; id: string }) => void
): Promise<void> {
  const pick = vscode.window.createQuickPick<vscode.QuickPickItem & { row?: Row }>();
  pick.matchOnDescription = true;
  pick.placeholder = 'Type to filter fields · Enter to copy the selected value';
  pick.buttons = [REFRESH_BUTTON, COPY_ALL_BUTTON, OPEN_TAB_BUTTON];
  pick.ignoreFocusOut = true;

  let current = target;
  let rows: Row[] = [];

  const load = async () => {
    pick.busy = true;
    pick.title = current.kind === 'job' ? `Job ${current.id}` : `Node ${current.id}`;
    try {
      const payload = await client.fetchDetail(current.kind, current.id);
      rows = payload.kind === 'job' ? jobRows(payload) : nodeRows(payload);
      pick.title = headline(payload);
      pick.items = toItems(rows);
      if (rows.length === 0) {
        pick.items = [{ label: 'Slurm returned no fields', description: 'the job may have finished' }];
      }
    } catch (err) {
      log.appendLine(`quick pick: ${current.kind} ${current.id} failed: ${String(err)}`);
      pick.items = [{ label: 'Could not read details', description: String(err) }];
    } finally {
      pick.busy = false;
    }
  };

  pick.onDidTriggerButton(async (button) => {
    if (button === REFRESH_BUTTON) {
      await load();
    } else if (button === OPEN_TAB_BUTTON) {
      pick.hide();
      openInTab(current);
    } else if (button === COPY_ALL_BUTTON) {
      await vscode.env.clipboard.writeText(rows.map((row) => `${row.label}: ${row.value}`).join('\n'));
      vscode.window.setStatusBarMessage(`Copied ${rows.length} fields`, 2000);
    }
  });

  pick.onDidAccept(async () => {
    const selected = pick.selectedItems[0];
    if (!selected?.row) {
      return;
    }
    if (selected.row.jump) {
      // Follow a job listed on a node without closing and reopening the pick.
      current = selected.row.jump;
      pick.value = '';
      await load();
      return;
    }
    await vscode.env.clipboard.writeText(selected.row.value);
    vscode.window.setStatusBarMessage(`Copied ${selected.row.label}`, 2000);
  });

  pick.onDidHide(() => pick.dispose());
  pick.show();
  await load();
}
