import * as vscode from 'vscode';

import { commandFor, describeCommand, readServerSettings } from './servers';

/**
 * Adding and managing servers.
 *
 * `slurmTop.servers` is a `name -> where` map, which VS Code's settings editor
 * renders as an editable grid, so the real answer to "how do I add a cluster"
 * is **Settings**. What is here is the shortcut: two questions, and the row is
 * written for you.
 */

/** The placeholder that explains the three forms a `where` can take. */
const WHERE_HINT = 'me@login01  ·  ssh -J jump login02  ·  here';

const WHERE_PROMPT =
  'An ssh destination as you would type it after `ssh`, a whole command line if it needs options, ' +
  'or `here` for the machine this is already running on.';

/** The scope to write to: wherever the setting is already kept. */
function target(): vscode.ConfigurationTarget {
  const inspected = vscode.workspace
    .getConfiguration('slurmTop')
    .inspect<Record<string, string>>('servers');
  if (inspected?.workspaceFolderValue !== undefined) {
    return vscode.ConfigurationTarget.WorkspaceFolder;
  }
  if (inspected?.workspaceValue !== undefined) {
    return vscode.ConfigurationTarget.Workspace;
  }
  // User settings by default, so a cluster you added once is there in every
  // window and every folder afterwards.
  return vscode.ConfigurationTarget.Global;
}

async function write(servers: Record<string, string>): Promise<void> {
  await vscode.workspace.getConfiguration('slurmTop').update('servers', servers, target());
}

/** Open the settings editor on the list itself. */
export async function openServerSettings(): Promise<void> {
  await vscode.commands.executeCommand('workbench.action.openSettings', 'slurmTop.servers');
}



/** Ask for a name that is not taken, and where to find it. */
async function ask(
  servers: Record<string, string>,
  was?: { name: string; where: string }
): Promise<{ name: string; where: string } | undefined> {
  const name = await vscode.window.showInputBox({
    title: was ? `Rename ${was.name}` : 'Add a Slurm server',
    prompt: 'What to call this cluster in the tables',
    placeHolder: 'alpha',
    value: was?.name,
    validateInput: (value) => {
      const trimmed = value.trim();
      if (!trimmed) {
        return 'A name is needed — it is what the tables label this cluster with.';
      }
      if (trimmed !== was?.name && trimmed in servers) {
        return `There is already a server called ${trimmed}.`;
      }
      return undefined;
    },
  });
  if (name === undefined) {
    return undefined;
  }
  const where = await vscode.window.showInputBox({
    title: `Where is ${name.trim()}?`,
    prompt: WHERE_PROMPT,
    placeHolder: WHERE_HINT,
    value: was?.where,
  });
  if (where === undefined) {
    return undefined;
  }
  return { name: name.trim(), where: where.trim() };
}

/**
 * Ask for one more cluster and add it to the map.
 *
 * The map it adds to is the one the settings grid shows, default row included,
 * so adding the second server writes the first one out explicitly rather than
 * dropping it.
 */
export async function addServer(): Promise<boolean> {
  const servers = { ...readServerSettings() };
  const answer = await ask(servers);
  if (!answer) {
    return false;
  }
  servers[answer.name] = answer.where;
  await write(servers);

  const ssh = commandFor(answer.where)[0] === 'ssh';
  const listed = Object.keys(servers).join(', ');
  vscode.window.showInformationMessage(
    `Watching ${listed}.` +
      (ssh
        ? ' Over ssh this needs slurm-monitor-top installed there, on the PATH of a non-interactive' +
          ' login, and a login that does not ask for a password.'
        : '')
  );
  return true;
}

/** The list, with what can be done to each row. */
export async function manageServers(): Promise<void> {
  const servers = { ...readServerSettings() };
  const names = Object.keys(servers);

  const items = [
    ...names.map((name) => ({
      label: `$(server) ${name}`,
      description: describeCommand(commandFor(servers[name])),
      action: 'edit' as const,
      name,
    })),
    { label: '$(add) Add a server…', description: '', action: 'add' as const, name: '' },
    {
      label: '$(gear) Edit the list in Settings',
      description: '',
      action: 'settings' as const,
      name: '',
    },
  ];

  const picked = await vscode.window.showQuickPick(items, { title: 'Slurm servers' });
  if (!picked) {
    return;
  }
  if (picked.action === 'add') {
    await addServer();
    return;
  }
  if (picked.action === 'settings') {
    await openServerSettings();
    return;
  }

  const name = picked.name;
  const action = await vscode.window.showQuickPick(
    [
      { label: 'Change name or address', action: 'edit' as const },
      { label: 'Remove', action: 'remove' as const },
    ],
    { title: name, placeHolder: describeCommand(commandFor(servers[name])) }
  );
  if (!action) {
    return;
  }
  if (action.action === 'remove') {
    if (names.length === 1) {
      vscode.window.showInformationMessage(
        `${name} is the only server there is, so it stays. Add another one first.`
      );
      return;
    }
    const remove = 'Remove';
    const confirm = await vscode.window.showWarningMessage(
      `Stop watching ${name}?`,
      { modal: true },
      remove
    );
    if (confirm !== remove) {
      return;
    }
    delete servers[name];
    await write(servers);
    return;
  }

  const answer = await ask(servers, { name, where: servers[name] });
  if (!answer) {
    return;
  }
  // Rebuilt rather than patched, so a renamed server keeps its place in the
  // list -- and so the panels keep their order.
  const next: Record<string, string> = {};
  for (const key of names) {
    if (key === name) {
      next[answer.name] = answer.where;
    } else {
      next[key] = servers[key];
    }
  }
  await write(next);
}
