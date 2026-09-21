import * as vscode from 'vscode';

import { ClusterClient } from './cluster';
import { detailPresentation, openDetailWindow, openJobOutput } from './detail';
import { showDetailQuickPick } from './quickpick';
import { readMergeSettings } from './servers';
import { DetailTarget, HostMessage, ViewMessage } from './types';

export type Variant = 'sidebar' | 'dashboard' | 'detail' | 'modal';

const ALL_SECTIONS = ['summary', 'jobs', 'nodes', 'gpus', 'disks'];

function nonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let text = '';
  for (let i = 0; i < 32; i += 1) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}

export function renderHtml(webview: vscode.Webview, extensionUri: vscode.Uri, variant: Variant): string {
  const media = (name: string) =>
    webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, 'media', name));
  const n = nonce();
  const csp = [
    `default-src 'none'`,
    `img-src ${webview.cspSource}`,
    `style-src ${webview.cspSource}`,
    `script-src 'nonce-${n}'`,
  ].join('; ');

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="Content-Security-Policy" content="${csp}" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="${media('main.css')}" />
    <title>Slurm</title>
  </head>
  <body data-variant="${variant}">
    <div id="status" class="status">Starting the Slurm collector…</div>
    <div id="root"></div>
    <div id="overlay" class="overlay hidden" role="dialog" aria-modal="true"></div>
    <script nonce="${n}" src="${media('main.js')}"></script>
  </body>
</html>`;
}

function sectionsFor(variant: Variant): string[] {
  if (variant !== 'sidebar') {
    return ALL_SECTIONS;
  }
  const configured = vscode.workspace
    .getConfiguration('slurmTop')
    .get<string[]>('sidebarSections', ALL_SECTIONS);
  const valid = configured.filter((s) => ALL_SECTIONS.includes(s));
  return valid.length > 0 ? valid : ['summary', 'jobs'];
}

function configMessage(variant: Variant, client: ClusterClient): HostMessage {
  const settings = vscode.workspace.getConfiguration('slurmTop');
  return {
    type: 'config',
    sections: sectionsFor(variant),
    ownerFilter: settings.get<string>('defaultOwnerFilter', 'me'),
    interval: settings.get<number>('refreshInterval', 3),
    detailsIn: detailPresentation(),
    servers: client.servers.map((spec) => ({ id: spec.id, name: client.nameOf(spec.id) })),
    merge: readMergeSettings(),
  };
}

/**
 * Read a node's CPU model on request and hand the refreshed detail back.
 *
 * Slurm knows how many cores a node has but not what they are, so this has to
 * go out to the machine itself -- over ssh, or failing that a one-second job.
 * That can take a while, hence the `started` message: the view says what it is
 * waiting for instead of looking frozen.
 */
async function probeCpu(
  client: ClusterClient,
  log: vscode.OutputChannel,
  target: DetailTarget,
  post: (message: HostMessage) => void
): Promise<void> {
  const node = target.id;
  post({ type: 'cpuProbe', server: target.server, node, state: 'started' });
  try {
    const detail = await client.fetchDetail(target, { probeCpu: true });
    post({ type: 'detail', server: target.server, serverName: client.nameOf(target.server), detail });
    const error = detail.kind === 'node' ? detail.cpu?.error : undefined;
    post({ type: 'cpuProbe', server: target.server, node, state: 'done', message: error });
  } catch (err) {
    log.appendLine(`cpu probe failed for ${node}: ${String(err)}`);
    post({ type: 'cpuProbe', server: target.server, node, state: 'done', message: String(err) });
  }
}

/**
 * Wire one webview (sidebar view or dashboard panel) to the shared client.
 *
 * `isVisible` is a callback rather than a value because a WebviewView and a
 * WebviewPanel report visibility through different properties, and the caller
 * is the one that knows which it holds.
 */
export function bindWebview(
  context: vscode.ExtensionContext,
  webview: vscode.Webview,
  variant: Variant,
  client: ClusterClient,
  log: vscode.OutputChannel,
  isVisible: () => boolean
): vscode.Disposable {
  const disposables: vscode.Disposable[] = [];
  const post = (message: HostMessage) => {
    if (isVisible()) {
      void webview.postMessage(message);
    }
  };

  disposables.push(
    webview.onDidReceiveMessage(async (raw: ViewMessage) => {
      switch (raw.type) {
        case 'ready': {
          post(configMessage(variant, client));
          const { state, message } = client.currentState;
          post({ type: 'status', state, message });
          if (client.latest) {
            post({ type: 'snapshot', snapshot: client.latest });
          }
          break;
        }
        case 'refresh':
          await client.restart();
          break;
        case 'openDashboard':
          await vscode.commands.executeCommand('slurmTop.openDashboard');
          break;
        case 'openDetail': {
          const presentation = detailPresentation();
          const target: DetailTarget = { server: raw.server ?? '', kind: raw.kind, id: raw.id };
          if (presentation === 'popup') {
            await showDetailQuickPick(client, log, target, (fallback) => {
              void openDetailWindow(context, client, log, fallback, false, true);
            });
            break;
          }
          if (presentation !== 'overlay') {
            await openDetailWindow(context, client, log, target);
            break;
          }
          try {
            const detail = await client.fetchDetail(target);
            post({
              type: 'detail',
              server: target.server,
              serverName: client.nameOf(target.server),
              detail,
            });
          } catch (err) {
            log.appendLine(`detail lookup failed for ${raw.kind} ${raw.id}: ${String(err)}`);
            post({ type: 'detailError', message: String(err) });
          }
          break;
        }
        case 'copy':
          await vscode.env.clipboard.writeText(raw.text);
          vscode.window.setStatusBarMessage(`Copied ${raw.label ?? 'value'}`, 2000);
          break;
        case 'togglePin':
          try {
            post({
              type: 'pinned',
              server: raw.server,
              pinned: await client.togglePin(raw.server, raw.jobId),
            });
          } catch (err) {
            log.appendLine(`pin toggle failed for ${raw.jobId}: ${String(err)}`);
            vscode.window.showWarningMessage(`Could not pin job ${raw.jobId}: ${String(err)}`);
          }
          break;
        case 'probeCpu':
          await probeCpu(client, log, { server: raw.server ?? '', kind: 'node', id: raw.node }, post);
          break;
        case 'openOutput':
          // Only reachable with `detailsIn: overlay`, where a job's detail --
          // output buttons and all -- is drawn inside this view rather than in
          // a window of its own.
          await openJobOutput(
            client,
            log,
            raw.server,
            String(raw.jobId),
            raw.stream === 'stderr' ? 'stderr' : 'stdout',
            String(raw.path ?? ''),
            Number(raw.size ?? 0)
          );
          break;
        case 'manageServers':
          await vscode.commands.executeCommand('slurmTop.manageServers');
          break;
        case 'showLog':
          log.show(true);
          break;
      }
    })
  );

  disposables.push(client.onSnapshot((snapshot) => post({ type: 'snapshot', snapshot })));
  disposables.push(client.onState(({ state, message }) => post({ type: 'status', state, message })));
  disposables.push(
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration('slurmTop')) {
        post(configMessage(variant, client));
      }
    })
  );

  return vscode.Disposable.from(...disposables);
}
