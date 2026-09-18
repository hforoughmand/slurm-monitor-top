import * as vscode from 'vscode';

import { SlurmClient } from './client';
import { DetailPresentation, JobDetail, NodeDetail } from './types';

/** Past this, opening the file whole is a bad idea; the tail is offered first. */
const LARGE_OUTPUT_BYTES = 25 * 1024 * 1024;
/** How much of a log the fallback fetches when the file itself is out of reach. */
const TAIL_LINES = 1000;

function humanBytes(size: number): string {
  const units = ['B', 'K', 'M', 'G', 'T'];
  let value = size;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${value}B` : `${value.toFixed(1)}${units[unit]}`;
}

/**
 * Show a job's stdout or stderr.
 *
 * The real file in an editor is the better answer whenever it can be had: it
 * follows, it searches, it is a normal tab. Two cases it cannot cover, and
 * both fall back to fetching the tail through the collector:
 *
 * - the collector runs somewhere else (a remote `slurmTop.command`), so the
 *   path exists on the cluster and not on this machine;
 * - the file is enormous, and a job that has been printing for a week should
 *   not be pulled through in full just to see the last few lines.
 */
export async function openJobOutput(
  client: SlurmClient,
  log: vscode.OutputChannel,
  jobId: string,
  stream: 'stdout' | 'stderr',
  path: string,
  size = 0
): Promise<void> {
  if (path && size > LARGE_OUTPUT_BYTES) {
    const whole = 'Open the whole file';
    const choice = await vscode.window.showWarningMessage(
      `${path} is ${humanBytes(size)}.`,
      { modal: false },
      whole,
      `Last ${TAIL_LINES} lines`
    );
    if (!choice) {
      return;
    }
    if (choice === whole) {
      await openFile(path, log);
      return;
    }
  } else if (path && (await openFile(path, log))) {
    return;
  }

  try {
    const payload = await client.fetchJobOutput(jobId, stream, TAIL_LINES);
    const tail = payload.output;
    if (!tail.text) {
      vscode.window.showWarningMessage(
        `No ${stream} to show for job ${jobId}: ${tail.error || 'the file is empty'}`
      );
      return;
    }
    const header =
      `${stream} of job ${jobId} — last ${tail.line_count} line(s) of ${tail.path}` +
      `${tail.truncated ? ' (the file is longer)' : ''} — read at ${new Date().toLocaleTimeString()}\n\n`;
    const document = await vscode.workspace.openTextDocument({ content: header + tail.text });
    await vscode.window.showTextDocument(document, { preview: false });
  } catch (err) {
    log.appendLine(`output: could not read ${stream} of ${jobId}: ${String(err)}`);
    vscode.window.showWarningMessage(`Could not read the ${stream} of job ${jobId}: ${String(err)}`);
  }
}

/** Open a path as an ordinary editor tab; false when it is not reachable here. */
async function openFile(path: string, log: vscode.OutputChannel): Promise<boolean> {
  const uri = vscode.Uri.file(path);
  try {
    await vscode.workspace.fs.stat(uri);
    const document = await vscode.workspace.openTextDocument(uri);
    await vscode.window.showTextDocument(document, { preview: false });
    return true;
  } catch (err) {
    log.appendLine(`output: ${path} is not readable from this machine (${String(err)})`);
    return false;
  }
}

/** Moves the active editor into a floating window; VS Code 1.85 and later. */
const MOVE_TO_NEW_WINDOW = 'workbench.action.moveEditorToNewWindow';

let floatingSupported: boolean | undefined;

/**
 * Whether this VS Code can put an editor in its own window.
 *
 * Checked once by asking for the command rather than by comparing
 * `vscode.version`, so a build that ships the feature under a different
 * version string still gets it, and an older one degrades to a tab instead of
 * throwing.
 */
async function canFloat(): Promise<boolean> {
  if (floatingSupported === undefined) {
    const commands = await vscode.commands.getCommands(true);
    floatingSupported = commands.includes(MOVE_TO_NEW_WINDOW);
  }
  return floatingSupported;
}
import { renderHtml } from './view';

interface Target {
  kind: 'job' | 'node';
  id: string;
}

/**
 * Job and node details, away from the sidebar.
 *
 * `scontrol show job` prints dozens of fields; squeezed into the sidebar's
 * ~300px they are unreadable. Both presentations here use the editor area
 * instead:
 *
 * - `modal` draws a centred card over a dimmed backdrop and disposes itself on
 *   Escape or a click outside, so it feels transient and leaves nothing behind.
 * - `tab` fills the editor like any other tab -- movable, splittable, and it
 *   stays put while you work next to it.
 *
 * VS Code gives extensions no way to paint outside their own frame, so the
 * modal dims the editor area it occupies rather than the whole window; the
 * sidebar and activity bar stay lit.
 *
 * One panel is reused and retitled as you click through rows: clicking twenty
 * jobs should not leave twenty tabs behind.
 */
class DetailWindow {
  private readonly disposables: vscode.Disposable[] = [];
  private target?: Target;
  private timer?: NodeJS.Timeout;
  private busy = false;

  constructor(
    private readonly panel: vscode.WebviewPanel,
    private readonly client: SlurmClient,
    private readonly log: vscode.OutputChannel,
    private readonly onDisposed: (window: DetailWindow) => void
  ) {
    this.disposables.push(
      panel.webview.onDidReceiveMessage(async (message) => {
        if (message.type === 'ready' || message.type === 'refreshDetail') {
          // A detail window opens with no pin list of its own, so seed it from
          // the collector's latest snapshot before the first paint.
          void this.panel.webview.postMessage({
            type: 'pinned',
            pinned: this.client.latest?.pinned ?? [],
          });
          await this.load();
        } else if (message.type === 'probeCpu') {
          await this.probeCpu(String(message.node));
        } else if (message.type === 'togglePin') {
          await this.togglePin(String(message.jobId));
        } else if (message.type === 'copy') {
          await vscode.env.clipboard.writeText(message.text);
          vscode.window.setStatusBarMessage(`Copied ${message.label ?? 'value'}`, 2000);
        } else if (message.type === 'openDetail') {
          // A job row inside a node's detail view, or vice versa.
          this.show({ kind: message.kind, id: String(message.id) });
        } else if (message.type === 'closeDetail') {
          this.panel.dispose();
        } else if (message.type === 'openOutput') {
          await openJobOutput(
            this.client,
            this.log,
            String(message.jobId),
            message.stream === 'stderr' ? 'stderr' : 'stdout',
            String(message.path ?? ''),
            Number(message.size ?? 0)
          );
        } else if (message.type === 'showLog') {
          this.log.show(true);
        }
      })
    );
    panel.onDidChangeViewState(() => this.reschedule());
    panel.onDidDispose(() => this.dispose());
  }

  get viewColumn(): vscode.ViewColumn | undefined {
    return this.panel.viewColumn;
  }

  /** Point this window at a job or node and start refreshing it. */
  show(target: Target): void {
    this.target = target;
    this.panel.title = target.kind === 'job' ? `Job ${target.id}` : `Node ${target.id}`;
    void this.panel.webview.postMessage({ type: 'detailTarget', kind: target.kind, id: target.id });
    void this.load();
    this.reschedule();
  }

  reveal(): void {
    // Once the panel lives in a floating window this raises that window; the
    // column is whatever VS Code assigned it there.
    this.panel.reveal(this.panel.viewColumn, true);
  }

  /**
   * Read this node's CPU off the machine, then redraw with the answer.
   *
   * Bypasses the `busy` guard's purpose deliberately: a probe is something the
   * user asked for by clicking, and it must not be dropped because a periodic
   * refresh happens to be in flight.
   */
  private async probeCpu(node: string): Promise<void> {
    void this.panel.webview.postMessage({ type: 'cpuProbe', node, state: 'started' });
    try {
      const detail = await this.client.fetchDetail('node', node, { probeCpu: true });
      if (this.target?.kind === 'node' && this.target.id === node) {
        void this.panel.webview.postMessage({ type: 'detail', detail });
      }
      const error = detail.kind === 'node' ? detail.cpu?.error : undefined;
      void this.panel.webview.postMessage({ type: 'cpuProbe', node, state: 'done', message: error });
    } catch (err) {
      this.log.appendLine(`detail window: cpu probe for ${node} failed: ${String(err)}`);
      void this.panel.webview.postMessage({
        type: 'cpuProbe',
        node,
        state: 'done',
        message: String(err),
      });
    }
  }

  private async togglePin(jobId: string): Promise<void> {
    try {
      const pinned = await this.client.togglePin(jobId);
      void this.panel.webview.postMessage({ type: 'pinned', pinned });
    } catch (err) {
      this.log.appendLine(`detail window: pin toggle for ${jobId} failed: ${String(err)}`);
      vscode.window.showWarningMessage(`Could not pin job ${jobId}: ${String(err)}`);
    }
  }

  private async load(): Promise<void> {
    const target = this.target;
    if (!target || this.busy) {
      return;
    }
    // `scontrol` on a busy controller can outlast the refresh interval; without
    // this guard the requests would stack up and hammer slurmctld.
    this.busy = true;
    try {
      const detail: JobDetail | NodeDetail = await this.client.fetchDetail(target.kind, target.id);
      if (this.target === target) {
        void this.panel.webview.postMessage({ type: 'detail', detail });
      }
    } catch (err) {
      this.log.appendLine(`detail window: ${target.kind} ${target.id} failed: ${String(err)}`);
      void this.panel.webview.postMessage({ type: 'detailError', message: String(err) });
    } finally {
      this.busy = false;
    }
  }

  private reschedule(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
    if (!this.panel.visible) {
      return;
    }
    const seconds = vscode.workspace.getConfiguration('slurmTop').get<number>('refreshInterval', 3);
    this.timer = setInterval(() => void this.load(), Math.max(1, seconds) * 1000);
  }

  dispose(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
    for (const item of this.disposables) {
      item.dispose();
    }
    this.disposables.length = 0;
    this.onDisposed(this);
  }
}

const windows: DetailWindow[] = [];

export async function openDetailWindow(
  context: vscode.ExtensionContext,
  client: SlurmClient,
  log: vscode.OutputChannel,
  target: Target,
  newWindow = false,
  forceTab = false
): Promise<void> {
  if (!newWindow && windows.length > 0) {
    const existing = windows[windows.length - 1];
    existing.show(target);
    existing.reveal();
    return;
  }

  const presentation = forceTab ? 'tab' : detailPresentation();
  const modal = presentation === 'card';
  const floating = presentation === 'window' && (await canFloat());
  if (presentation === 'window' && !floating) {
    log.appendLine(
      `detail: ${MOVE_TO_NEW_WINDOW} is unavailable in this VS Code, opening a tab instead`
    );
  }
  const panel = vscode.window.createWebviewPanel(
    'slurmTop.detail',
    target.kind === 'job' ? `Job ${target.id}` : `Node ${target.id}`,
    // A modal takes the active column so it covers what you were looking at,
    // and takes focus so Escape reaches it; a tab opens beside and leaves the
    // keyboard where it was.
    // A modal covers what you were looking at, and a floating window has to be
    // the active editor for the move command to pick it up; a plain tab opens
    // beside and leaves the keyboard where it was.
    modal || floating
      ? { viewColumn: vscode.ViewColumn.Active, preserveFocus: false }
      : { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')],
    }
  );
  panel.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'slurm.svg');
  panel.webview.html = renderHtml(panel.webview, context.extensionUri, modal ? 'modal' : 'detail');

  const window = new DetailWindow(panel, client, log, (closed) => {
    const index = windows.indexOf(closed);
    if (index >= 0) {
      windows.splice(index, 1);
    }
  });
  windows.push(window);
  window.show(target);

  if (floating) {
    // The command acts on the active editor, which is the panel we just opened
    // focused. Yielding first lets VS Code finish making it active; without
    // that the wrong editor can be the one that moves.
    await new Promise((resolve) => setTimeout(resolve, 0));
    try {
      await vscode.commands.executeCommand(MOVE_TO_NEW_WINDOW);
    } catch (err) {
      log.appendLine(`detail: could not move the panel to its own window: ${String(err)}`);
    }
  }
}

export function detailPresentation(): DetailPresentation {
  const value = vscode.workspace.getConfiguration('slurmTop').get<string>('detailsIn', 'window');
  if (value === 'overlay' || value === 'popup' || value === 'tab') {
    return value;
  }
  // 'modal' was this card's name before the floating window existed.
  if (value === 'card' || value === 'modal') {
    return 'card';
  }
  return 'window';
}
