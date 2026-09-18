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
  Uri: { joinPath: (base, ...parts) => ({ path: [base && base.path, ...parts].filter(Boolean).join('/') }) },
  workspace: {
    getConfiguration: () => ({ get: (key, fallback) => (key in settings ? settings[key] : fallback) }),
    onDidChangeConfiguration: () => ({ dispose() {} }),
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
    createOutputChannel: () => ({ appendLine: (l) => logLines.push(l), show: () => {}, dispose: () => {} }),
  },
};
const logLines = [];

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
}

main()
  .then(() => {
    console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
    process.exit(failures ? 1 : 0);
  })
  .catch((err) => { console.error('\nerror:', err); process.exit(1); });
