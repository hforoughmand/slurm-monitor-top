import { execFile } from 'child_process';
import * as vscode from 'vscode';

import { ClusterClient } from './cluster';
import { detailPresentation, openDetailWindow, openJobOutput } from './detail';
import { showDetailQuickPick } from './quickpick';
import { addServer, manageServers } from './serverui';
import { DetailTarget } from './types';
import { bindWebview, renderHtml } from './view';

const DASHBOARD_VIEW_TYPE = 'slurmTop.dashboard';

/**
 * Whether the `slurm-top` terminal UI is installed.
 *
 * Looked up rather than executed: running it to find out would start a
 * full-screen TUI in whatever terminal we opened, and an older build ignores
 * `--version` and does exactly that.
 */
function terminalCliInstalled(): Promise<boolean> {
  return new Promise((resolve) => {
    execFile('/usr/bin/env', ['which', 'slurm-top'], { timeout: 5000 }, (err) => resolve(!err));
  });
}

/**
 * Tracks which Slurm views are on screen so the collector can be paused when
 * none of them are. Polling squeue every few seconds for a view nobody is
 * looking at is rude on a shared login node.
 */
class VisibilityTracker {
  private readonly visible = new Set<string>();

  constructor(private readonly client: ClusterClient) {}

  set(key: string, isVisible: boolean): void {
    if (isVisible) {
      this.visible.add(key);
    } else {
      this.visible.delete(key);
    }
    this.apply();
  }

  forget(key: string): void {
    this.visible.delete(key);
    this.apply();
  }

  apply(): void {
    const pauseWhenHidden = vscode.workspace
      .getConfiguration('slurmTop')
      .get<boolean>('pauseWhenHidden', true);
    if (this.visible.size > 0 || !pauseWhenHidden) {
      void this.client.start();
    } else {
      this.client.stop();
    }
  }
}

class SidebarProvider implements vscode.WebviewViewProvider {
  static readonly viewType = 'slurmTop.sidebar';

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly client: ClusterClient,
    private readonly log: vscode.OutputChannel,
    private readonly visibility: VisibilityTracker
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, 'media')],
    };
    view.webview.html = renderHtml(view.webview, this.context.extensionUri, 'sidebar');

    const binding = bindWebview(this.context, view.webview, 'sidebar', this.client, this.log, () => view.visible);
    this.visibility.set('sidebar', view.visible);
    view.onDidChangeVisibility(() => this.visibility.set('sidebar', view.visible));
    view.onDidDispose(() => {
      binding.dispose();
      this.visibility.forget('sidebar');
    });
  }
}

let dashboard: vscode.WebviewPanel | undefined;

function openDashboard(
  context: vscode.ExtensionContext,
  client: ClusterClient,
  log: vscode.OutputChannel,
  visibility: VisibilityTracker
): void {
  if (dashboard) {
    dashboard.reveal(dashboard.viewColumn ?? vscode.ViewColumn.Active);
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    DASHBOARD_VIEW_TYPE,
    'Slurm Dashboard',
    { viewColumn: vscode.ViewColumn.Active, preserveFocus: false },
    {
      enableScripts: true,
      // Without this the webview is torn down when the tab goes to the
      // background, and a tab you come back to would flash empty.
      retainContextWhenHidden: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')],
    }
  );
  panel.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'slurm.svg');
  panel.webview.html = renderHtml(panel.webview, context.extensionUri, 'dashboard');

  const binding = bindWebview(context, panel.webview, 'dashboard', client, log, () => panel.visible);
  dashboard = panel;
  visibility.set('dashboard', panel.visible);
  panel.onDidChangeViewState(() => visibility.set('dashboard', panel.visible));
  panel.onDidDispose(() => {
    binding.dispose();
    dashboard = undefined;
    visibility.forget('dashboard');
  });
}

/**
 * Which server a typed id belongs to.
 *
 * Only asked when there is more than one: job ids and node names are unique
 * within a cluster and nowhere else, so with two clusters watching, "1234"
 * alone is genuinely ambiguous.
 */
async function pickServer(client: ClusterClient, what: string): Promise<string | undefined> {
  const servers = client.servers;
  if (servers.length <= 1) {
    return servers[0]?.id ?? '';
  }
  const picked = await vscode.window.showQuickPick(
    servers.map((spec) => ({ label: client.nameOf(spec.id), id: spec.id })),
    { title: `Which server has this ${what}?` }
  );
  return picked?.id;
}

/** Open details wherever `slurmTop.detailsIn` says they belong. */
async function showDetails(
  context: vscode.ExtensionContext,
  client: ClusterClient,
  log: vscode.OutputChannel,
  target: DetailTarget
): Promise<void> {
  if (detailPresentation() === 'popup') {
    await showDetailQuickPick(client, log, target, (fallback) => {
      void openDetailWindow(context, client, log, fallback, false, true);
    });
    return;
  }
  // 'overlay' has no view to draw in when the command is run from the palette,
  // so it falls back to a panel like the other non-popup shapes.
  await openDetailWindow(context, client, log, target);
}

/**
 * Ask which of a job's two output files to open, then open it.
 *
 * The pick is skipped when there is only one file worth offering: most batch
 * scripts send both streams to the same place, and a menu of one is noise.
 */
async function pickJobOutput(
  client: ClusterClient,
  log: vscode.OutputChannel,
  server: string,
  jobId: string
): Promise<void> {
  let streams;
  try {
    const detail = await client.fetchDetail({ server, kind: 'job', id: jobId });
    streams = detail.kind === 'job' ? detail.output : undefined;
  } catch (err) {
    log.appendLine(`output: could not look up job ${jobId}: ${String(err)}`);
    vscode.window.showWarningMessage(`Could not look up job ${jobId}: ${String(err)}`);
    return;
  }
  if (!streams) {
    vscode.window.showWarningMessage(
      `Slurm reported no output paths for job ${jobId}. Only the job's owner can see them.`
    );
    return;
  }
  const separate = (['stdout', 'stderr'] as const).filter(
    (name) => !(name === 'stderr' && streams[name].merged)
  );
  const choices: (vscode.QuickPickItem & { streams: ('stdout' | 'stderr')[] })[] = separate.map((name) => ({
    label: String(name),
    description: streams[name].path || streams[name].error,
    detail: streams[name].exists ? undefined : streams[name].error,
    streams: [name],
  }));
  if (separate.length > 1) {
    // Both is what you want when you do not yet know which stream said the
    // useful thing, so it leads — as it does in the terminal UI's viewer.
    choices.unshift({
      label: 'both',
      description: 'open stdout and stderr side by side',
      detail: undefined,
      streams: [...separate],
    });
  }
  const chosen =
    choices.length === 1
      ? choices[0]
      : await vscode.window.showQuickPick(choices, { placeHolder: `Output of job ${jobId}` });
  if (!chosen) {
    return;
  }
  for (const stream of chosen.streams) {
    const file = streams[stream];
    await openJobOutput(client, log, server, jobId, stream, file.path, file.size);
  }
}

export function activate(context: vscode.ExtensionContext): void {
  const log = vscode.window.createOutputChannel('Slurm Monitor');
  const client = new ClusterClient(context.extensionPath, log);
  const visibility = new VisibilityTracker(client);
  context.subscriptions.push(log, client);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      SidebarProvider.viewType,
      new SidebarProvider(context, client, log, visibility),
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('slurmTop.openDashboard', () =>
      openDashboard(context, client, log, visibility)
    ),
    vscode.commands.registerCommand('slurmTop.refresh', () => client.restart()),
    vscode.commands.registerCommand('slurmTop.restart', () => client.restart()),
    vscode.commands.registerCommand('slurmTop.showOutput', () => log.show(true)),
    vscode.commands.registerCommand('slurmTop.addServer', () => addServer()),
    vscode.commands.registerCommand('slurmTop.manageServers', () => manageServers()),
    vscode.commands.registerCommand('slurmTop.openJobDetails', async (jobId?: string) => {
      const id = jobId ?? (await vscode.window.showInputBox({ prompt: 'Slurm job id', placeHolder: '1234567' }));
      if (!id) {
        return;
      }
      const server = await pickServer(client, 'job');
      if (server !== undefined) {
        await showDetails(context, client, log, { server, kind: 'job', id: id.trim() });
      }
    }),
    vscode.commands.registerCommand('slurmTop.openNodeDetails', async (nodeName?: string) => {
      const id = nodeName ?? (await vscode.window.showInputBox({ prompt: 'Node name', placeHolder: 'node01' }));
      if (!id) {
        return;
      }
      const server = await pickServer(client, 'machine');
      if (server !== undefined) {
        await showDetails(context, client, log, { server, kind: 'node', id: id.trim() });
      }
    }),
    vscode.commands.registerCommand('slurmTop.openJobOutput', async (jobId?: string) => {
      const id =
        jobId ?? (await vscode.window.showInputBox({ prompt: 'Slurm job id', placeHolder: '1234567' }));
      if (!id) {
        return;
      }
      const server = await pickServer(client, 'job');
      if (server !== undefined) {
        await pickJobOutput(client, log, server, id.trim());
      }
    }),
    vscode.commands.registerCommand('slurmTop.openTerminalTui', async () => {
      // The extension itself does not need the Python package -- it falls back
      // to its bundled collector -- so this command is the one place a missing
      // install shows up. Say so instead of printing "command not found".
      if (!(await terminalCliInstalled())) {
        const install = 'Copy install command';
        const choice = await vscode.window.showWarningMessage(
          'The slurm-top terminal dashboard is not installed. The Slurm views in the editor work without it.',
          install
        );
        if (choice === install) {
          await vscode.env.clipboard.writeText('pip install slurm-monitor-top');
          vscode.window.setStatusBarMessage('Copied: pip install slurm-monitor-top', 3000);
        }
        return;
      }
      const terminal = vscode.window.createTerminal('slurm-top');
      terminal.show();
      terminal.sendText('slurm-top');
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (event) => {
      if (event.affectsConfiguration('slurmTop.servers')) {
        // Only the servers that actually changed are respawned; the rest keep
        // streaming, so adding a fourth cluster does not blank the other three.
        await client.reload();
      } else if (
        event.affectsConfiguration('slurmTop.pythonPath') ||
        event.affectsConfiguration('slurmTop.command') ||
        event.affectsConfiguration('slurmTop.refreshInterval')
      ) {
        await client.restart();
      }
      if (event.affectsConfiguration('slurmTop.pauseWhenHidden')) {
        visibility.apply();
      }
    })
  );
}

export function deactivate(): void {
  // Disposables registered on the context handle teardown; the collector child
  // is killed by SlurmClient.dispose().
}
