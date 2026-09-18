/**
 * Headless check that details really do open in a floating window.
 *
 * `vscode` is stubbed, so this verifies the bits that decide the shape: that
 * the panel is created as the *active* editor (the move command acts on that,
 * so getting it wrong would fling someone's source file into a new window),
 * that the move is actually issued, and that an older VS Code without the
 * command degrades to a tab instead of throwing.
 *
 * Run with: node test/detail.smoke.js
 */
const Module = require('module');

let failures = 0;
function check(label, condition, detail) {
  if (condition) console.log(`  ok   ${label}`);
  else {
    failures += 1;
    console.log(`  FAIL ${label}${detail ? ` (${detail})` : ''}`);
  }
}

let settings = {};
let availableCommands = ['workbench.action.moveEditorToNewWindow', 'noise.command'];
const executed = [];
const panels = [];

class StubEmitter {
  get event() { return () => ({ dispose() {} }); }
  fire() {}
  dispose() {}
}

function makePanel(viewType, title, options) {
  const panel = {
    viewType,
    title,
    options,
    visible: true,
    viewColumn: options.viewColumn,
    iconPath: undefined,
    html: '',
    posted: [],
    disposed: false,
    webview: {
      html: '',
      asWebviewUri: (u) => u,
      postMessage(m) { panel.posted.push(m); return Promise.resolve(true); },
      onDidReceiveMessage: () => ({ dispose() {} }),
      set options(_v) {},
    },
    reveal() { this.revealed = true; },
    dispose() { this.disposed = true; this._onDispose && this._onDispose(); },
    onDidChangeViewState: () => ({ dispose() {} }),
    onDidDispose(fn) { this._onDispose = fn; return { dispose() {} }; },
  };
  Object.defineProperty(panel.webview, 'html', { value: '', writable: true });
  panels.push(panel);
  return panel;
}

const vscodeStub = {
  EventEmitter: StubEmitter,
  Disposable: { from: (...i) => ({ dispose: () => i.forEach((x) => x.dispose && x.dispose()) }) },
  ViewColumn: { Active: -1, Beside: -2, One: 1 },
  ThemeIcon: class { constructor(id) { this.id = id; } },
  QuickPickItemKind: { Separator: -1 },
  Uri: {
    joinPath: (base, ...parts) => ({ path: [base && base.path, ...parts].filter(Boolean).join('/') }),
    file: (p) => ({ scheme: 'file', path: p, fsPath: p }),
  },
  workspace: {
    getConfiguration: () => ({ get: (key, fallback) => (key in settings ? settings[key] : fallback) }),
    onDidChangeConfiguration: () => ({ dispose() {} }),
    // A file "exists" for this stub when `readableFiles` has its path; the
    // openJobOutput cases below flip that to test both routes.
    fs: {
      stat: async (uri) => {
        if (!readableFiles.has(uri.path)) throw new Error('ENOENT');
        return { size: 1 };
      },
    },
    openTextDocument: async (arg) => {
      opened.push(arg);
      return arg;
    },
  },
  env: { clipboard: { writeText: async () => {} } },
  commands: {
    getCommands: async () => availableCommands,
    executeCommand: async (id) => { executed.push(id); },
  },
  window: {
    createWebviewPanel: makePanel,
    createQuickPick: () => ({ onDidAccept() {}, onDidTriggerButton() {}, onDidHide() {}, show() {}, dispose() {}, items: [], buttons: [] }),
    setStatusBarMessage: () => {},
    showTextDocument: async (doc) => { shown.push(doc); },
    showWarningMessage: async (message, ...rest) => {
      warnings.push(message);
      // The options object, when passed, is not one of the buttons.
      const buttons = rest.filter((r) => typeof r === 'string');
      return warningAnswer === undefined ? undefined : buttons.find((b) => b === warningAnswer);
    },
    createOutputChannel: () => ({ appendLine: (l) => logLines.push(l), show: () => {}, dispose: () => {} }),
  },
};
const logLines = [];
const readableFiles = new Set();
const opened = [];
const shown = [];
const warnings = [];
let warningAnswer;

const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return 'vscode';
  return originalResolve.call(this, request, ...rest);
};
require.cache.vscode = { id: 'vscode', filename: 'vscode', loaded: true, exports: vscodeStub };

/**
 * Re-require detail.js with an empty module cache.
 *
 * The module caches both the open-window list and the one-time floating-window
 * probe, so cases that need a different starting state have to start from a
 * fresh instance rather than trying to reach in and reset it.
 */
function freshDetail() {
  delete require.cache[require.resolve('../out/detail.js')];
  delete require.cache[require.resolve('../out/view.js')];
  panels.length = 0;
  executed.length = 0;
  logLines.length = 0;
  return require('../out/detail.js');
}

const detail = require('../out/detail.js');
const log = vscodeStub.window.createOutputChannel();
const context = { extensionUri: { path: '/ext' }, extensionPath: '/ext' };
const client = { fetchDetail: async () => ({ kind: 'job', job_id: '1', job: null, detail: {}, usage: {} }) };

/** The collector's answer when the file itself cannot be opened from here. */
function outputClient(text) {
  return {
    fetchDetail: async () => ({ kind: 'job', job_id: '1', job: null, detail: {}, usage: {} }),
    calls: [],
    async fetchJobOutput(jobId, stream, lines) {
      this.calls.push({ jobId, stream, lines });
      return {
        kind: 'job-output',
        job_id: jobId,
        streams: {},
        output: {
          path: '/scratch/alice/logs/1001.out', exists: true, size: 10, modified: 0,
          error: '', stream, merged: false, text, line_count: text.split('\n').length,
          truncated: true,
        },
      };
    },
  };
}

async function checkJobOutput() {
  console.log('\njob output:');
  const fresh = freshDetail();

  // The file is right there: it opens as an ordinary tab, and the collector is
  // never asked for a copy of what the editor can read itself.
  readableFiles.add('/scratch/alice/logs/1001.out');
  opened.length = 0;
  shown.length = 0;
  let client = outputClient('hello');
  await fresh.openJobOutput(client, log, '1001', 'stdout', '/scratch/alice/logs/1001.out', 1024);
  check('a readable file is opened as a tab', opened.length === 1 && opened[0].path === '/scratch/alice/logs/1001.out',
    JSON.stringify(opened));
  check('and it is shown', shown.length === 1);
  check('a readable file costs no collector call', client.calls.length === 0, JSON.stringify(client.calls));

  // The same path when this machine cannot see it -- a remote collector, or a
  // compute-node-local scratch directory.
  readableFiles.clear();
  opened.length = 0;
  client = outputClient('tail from the cluster');
  await fresh.openJobOutput(client, log, '1001', 'stderr', '/scratch/alice/logs/1001.out', 1024);
  check('an unreachable file falls back to the collector', client.calls.length === 1,
    JSON.stringify(client.calls));
  check('the fallback asks for the stream that was clicked', client.calls[0]?.stream === 'stderr');
  check('the tail is opened as an untitled document',
    opened.length === 1 && typeof opened[0].content === 'string' &&
      opened[0].content.includes('tail from the cluster'),
    JSON.stringify(opened).slice(0, 160));
  check('the document says which file and how much of it',
    opened[0].content.includes('/scratch/alice/logs/1001.out') && opened[0].content.includes('the file is longer'),
    opened[0].content.split('\n')[0]);
  check('the failure is explained in the log', logLines.some((l) => l.includes('not readable')),
    logLines.join(' | '));

  // A log too big to pull through whole: the question comes first.
  readableFiles.add('/scratch/alice/logs/huge.out');
  opened.length = 0;
  warnings.length = 0;
  warningAnswer = undefined;
  client = outputClient('the last lines');
  await fresh.openJobOutput(client, log, '1001', 'stdout', '/scratch/alice/logs/huge.out', 512 * 1024 * 1024);
  check('a huge file is not opened without asking', warnings.length === 1 && opened.length === 0,
    JSON.stringify(warnings));
  check('the question says how big it is', warnings[0]?.includes('512.0M'), warnings[0]);

  warningAnswer = 'Open the whole file';
  opened.length = 0;
  await fresh.openJobOutput(client, log, '1001', 'stdout', '/scratch/alice/logs/huge.out', 512 * 1024 * 1024);
  check('answering opens the whole thing', opened.length === 1 && opened[0].path === '/scratch/alice/logs/huge.out',
    JSON.stringify(opened));
  warningAnswer = undefined;
  readableFiles.clear();
}

async function main() {
  console.log('setting resolution:');
  const cases = [
    [undefined, 'window'], ['window', 'window'], ['popup', 'popup'], ['card', 'card'],
    ['tab', 'tab'], ['overlay', 'overlay'],
    // Names this setting used earlier in development must keep resolving.
    ['modal', 'card'],
    ['nonsense', 'window'],
  ];
  for (const [value, expected] of cases) {
    settings = value === undefined ? {} : { detailsIn: value };
    const got = detail.detailPresentation();
    check(`${value ?? '(unset)'} -> ${expected}`, got === expected, got);
  }

  console.log('\nfloating window:');
  let mod = freshDetail();
  settings = { detailsIn: 'window' };
  await mod.openDetailWindow(context, client, log, { kind: 'job', id: '1001' });
  check('a panel was created', panels.length === 1);
  check('panel opens as the ACTIVE editor', panels[0].options.viewColumn === vscodeStub.ViewColumn.Active,
    String(panels[0].options.viewColumn));
  check('panel takes focus, so the move targets it', panels[0].options.preserveFocus === false);
  check('the move-to-new-window command was issued',
    executed.includes('workbench.action.moveEditorToNewWindow'), executed.join(','));
  check('panel is titled after the job', panels[0].title === 'Job 1001', panels[0].title);

  console.log('\nreuse:');
  await mod.openDetailWindow(context, client, log, { kind: 'node', id: 'gpu01' });
  check('a second request reuses the same window', panels.length === 1);
  check('the reused window is revealed', panels[0].revealed === true);
  check('the reused window is retitled', panels[0].title === 'Node gpu01', panels[0].title);
  check('reuse does not spawn another window',
    executed.filter((c) => c === 'workbench.action.moveEditorToNewWindow').length === 1);

  console.log('\nolder VS Code without floating windows:');
  availableCommands = ['noise.command'];
  mod = freshDetail();
  settings = { detailsIn: 'window' };
  await mod.openDetailWindow(context, client, log, { kind: 'job', id: '1002' });
  check('no move command is attempted',
    !executed.includes('workbench.action.moveEditorToNewWindow'), executed.join(','));
  check('it falls back to a tab beside the editor',
    panels[panels.length - 1].options.viewColumn === vscodeStub.ViewColumn.Beside);
  check('the fallback is explained in the log',
    logLines.some((l) => l.includes('unavailable')), logLines.join(' | '));
  availableCommands = ['workbench.action.moveEditorToNewWindow'];

  console.log('\ncard and tab shapes:');
  availableCommands = ['workbench.action.moveEditorToNewWindow'];
  mod = freshDetail();
  settings = { detailsIn: 'tab' };
  await mod.openDetailWindow(context, client, log, { kind: 'job', id: '1003' });

  check('a tab opens beside, unfocused', panels[0].options.viewColumn === vscodeStub.ViewColumn.Beside
    && panels[0].options.preserveFocus === true);
  check('a tab is not moved to a window',
    !executed.includes('workbench.action.moveEditorToNewWindow'));

  mod = freshDetail();
  settings = { detailsIn: 'card' };
  await mod.openDetailWindow(context, client, log, { kind: 'job', id: '1004' });
  check('a card takes the active column so it covers the editor',
    panels[0].options.viewColumn === vscodeStub.ViewColumn.Active);
  check('a card is not moved to its own window',
    !executed.includes('workbench.action.moveEditorToNewWindow'));

  // forceTab is how the popup's escape hatch asks for a tab regardless.
  mod = freshDetail();
  settings = { detailsIn: 'window' };
  await mod.openDetailWindow(context, client, log, { kind: 'job', id: '1005' }, false, true);
  check('forceTab overrides the window setting',
    panels[0].options.viewColumn === vscodeStub.ViewColumn.Beside
      && !executed.includes('workbench.action.moveEditorToNewWindow'));

  await checkJobOutput();
}

main()
  .then(() => {
    console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
    process.exit(failures ? 1 : 0);
  })
  .catch((err) => { console.error('\nerror:', err); process.exit(1); });
