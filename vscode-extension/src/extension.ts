import { execFile } from 'child_process';
import * as vscode from 'vscode';

import { SlurmClient } from './client';
import { detailPresentation, openDetailWindow } from './detail';
import { showDetailQuickPick } from './quickpick';
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

  constructor(private readonly client: SlurmClient) {}

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
    private readonly client: SlurmClient,
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
  client: SlurmClient,
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

/** Open details wherever `slurmTop.detailsIn` says they belong. */
async function showDetails(
  context: vscode.ExtensionContext,
  client: SlurmClient,
  log: vscode.OutputChannel,
  target: { kind: 'job' | 'node'; id: string }
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

export function activate(context: vscode.ExtensionContext): void {
  const log = vscode.window.createOutputChannel('Slurm Monitor');
  const client = new SlurmClient(context.extensionPath, log);
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
    vscode.commands.registerCommand('slurmTop.openJobDetails', async (jobId?: string) => {
      const id = jobId ?? (await vscode.window.showInputBox({ prompt: 'Slurm job id', placeHolder: '1234567' }));
      if (id) {
        await showDetails(context, client, log, { kind: 'job', id: id.trim() });
      }
    }),
    vscode.commands.registerCommand('slurmTop.openNodeDetails', async (nodeName?: string) => {
      const id = nodeName ?? (await vscode.window.showInputBox({ prompt: 'Node name', placeHolder: 'node01' }));
      if (id) {
        await showDetails(context, client, log, { kind: 'node', id: id.trim() });
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
      if (
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
