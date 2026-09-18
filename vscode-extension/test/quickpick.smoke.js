/**
 * Headless check of the quick-pick detail popup.
 *
 * `vscode` is stubbed with a fake QuickPick that records what the code puts on
 * it, so this verifies the field list, the section separators, copy-on-Enter
 * and the node -> job jump without VS Code being present.
 *
 * Run with: node test/quickpick.smoke.js
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

const clipboard = { text: '' };
const statusMessages = [];

class FakeQuickPick {
  constructor() {
    this.items = [];
    this.buttons = [];
    this.selectedItems = [];
    this.busy = false;
    this.title = '';
    this.value = '';
    this.shown = false;
    this.disposed = false;
    this.handlers = { accept: [], button: [], hide: [] };
  }
  onDidAccept(fn) { this.handlers.accept.push(fn); }
  onDidTriggerButton(fn) { this.handlers.button.push(fn); }
  onDidHide(fn) { this.handlers.hide.push(fn); }
  show() { this.shown = true; }
  hide() { this.shown = false; }
  dispose() { this.disposed = true; }
  async accept(item) { this.selectedItems = [item]; for (const fn of this.handlers.accept) await fn(); }
  async trigger(button) { for (const fn of this.handlers.button) await fn(button); }
}

let lastPick;
const vscodeStub = {
  QuickPickItemKind: { Separator: -1, Default: 0 },
  ThemeIcon: class { constructor(id) { this.id = id; } },
  workspace: { getConfiguration: () => ({ get: (_k, fallback) => fallback }) },
  env: { clipboard: { writeText: async (t) => { clipboard.text = t; } } },
  window: {
    createQuickPick: () => (lastPick = new FakeQuickPick()),
    setStatusBarMessage: (m) => statusMessages.push(m),
    createOutputChannel: () => ({ appendLine: () => {}, show: () => {}, dispose: () => {} }),
  },
};

const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return 'vscode';
  return originalResolve.call(this, request, ...rest);
};
require.cache.vscode = { id: 'vscode', filename: 'vscode', loaded: true, exports: vscodeStub };

const { showDetailQuickPick } = require('../out/quickpick.js');

const JOB = {
  kind: 'job',
  job_id: '1001',
  job: { job_id: '1001', user: 'alice', state: 'RUNNING', name: 'train-a', gpu_types: { a100: 2 } },
  detail: { JobId: '1001', WorkDir: '/home/alice', Command: '/home/alice/run.sh' },
  usage: { MaxRSS: '12G' },
};
const NODE = {
  kind: 'node',
  node: 'gpu01',
  detail: { NodeName: 'gpu01', CPUTot: '64' },
  jobs: [{ job_id: '1001', user: 'alice', state: 'RUNNING', ncpus: '16', mem: '64G', name: 'train-a', time_used: '2:11' }],
};

const log = vscodeStub.window.createOutputChannel();
const labels = (pick) => pick.items.filter((i) => i.kind !== -1).map((i) => i.label);
const separators = (pick) => pick.items.filter((i) => i.kind === -1).map((i) => i.label);

async function main() {
  console.log('job popup:');
  const client = { fetchDetail: async (kind, id) => (kind === 'job' ? { ...JOB, job_id: id } : NODE) };
  let openedInTab = null;
  await showDetailQuickPick(client, log, { kind: 'job', id: '1001' }, (t) => (openedInTab = t));
  const pick = lastPick;

  check('the popup is shown', pick.shown);
  check('no editor tab is involved', openedInTab === null);
  check('title names the job and its state', pick.title.includes('1001') && pick.title.includes('RUNNING'), pick.title);
  check('fields are grouped by source', separators(pick).join(',') === 'squeue,scontrol,usage', separators(pick).join(','));
  check('squeue fields are listed', labels(pick).includes('state'));
  check('scontrol fields are listed', labels(pick).includes('WorkDir'));
  check('usage fields are listed', labels(pick).includes('MaxRSS'));
  check('nested objects are skipped, not stringified', !labels(pick).includes('gpu_types'));
  check('values ride along as the description',
    pick.items.find((i) => i.label === 'WorkDir').description === '/home/alice');
  check('busy flag is cleared after loading', pick.busy === false);

  const workDir = pick.items.find((i) => i.label === 'WorkDir');
  await pick.accept(workDir);
  check('Enter copies the selected value', clipboard.text === '/home/alice', clipboard.text);
  check('copying is confirmed in the status bar', statusMessages.some((m) => m.includes('WorkDir')));
  check('the popup stays open after a copy', pick.shown);

  await pick.trigger(pick.buttons[1]);
  check('copy-all copies every field', clipboard.text.split('\n').length === 8, String(clipboard.text.split('\n').length));
  check('copy-all keeps key: value form', clipboard.text.includes('WorkDir: /home/alice'));

  await pick.trigger(pick.buttons[0]);
  check('refresh reloads without closing', pick.shown && labels(pick).includes('WorkDir'));

  await pick.trigger(pick.buttons[2]);
  check('the escape hatch opens a tab and closes the popup', !pick.shown && openedInTab?.id === '1001');

  console.log('\nnode popup:');
  await showDetailQuickPick(client, log, { kind: 'node', id: 'gpu01' }, () => {});
  const nodePick = lastPick;
  check('node title counts its jobs', nodePick.title.includes('gpu01') && nodePick.title.includes('1 job'), nodePick.title);
  check('node fields are listed', labels(nodePick).includes('CPUTot'));
  check('jobs on the node get their own section',
    separators(nodePick).some((s) => s.includes('jobs on gpu01')), separators(nodePick).join(','));

  const jobRow = nodePick.items.find((i) => i.row && i.row.jump);
  check('a job row is marked as a jump, not a copy', !!jobRow && jobRow.detail.includes('open'));
  await nodePick.accept(jobRow);
  check('accepting a job row switches the popup to that job',
    nodePick.title.includes('Job 1001'), nodePick.title);
  check('switching clears the filter text', nodePick.value === '');
  check('the switched popup shows job fields', labels(nodePick).includes('WorkDir'));

  console.log('\nfailure handling:');
  const broken = { fetchDetail: async () => { throw new Error('scontrol timed out'); } };
  await showDetailQuickPick(broken, log, { kind: 'job', id: '9' }, () => {});
  check('an error becomes a visible item, not a crash',
    lastPick.items[0].description.includes('scontrol timed out'), JSON.stringify(lastPick.items[0]));
  check('busy is cleared even when loading failed', lastPick.busy === false);

  const empty = { fetchDetail: async () => ({ kind: 'job', job_id: '9', job: null, detail: {}, usage: {} }) };
  await showDetailQuickPick(empty, log, { kind: 'job', id: '9' }, () => {});
  check('an empty result explains itself', lastPick.items[0].label.includes('no fields'), lastPick.items[0].label);
}

main()
  .then(() => {
    console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
    process.exit(failures ? 1 : 0);
  })
  .catch((err) => { console.error('\nerror:', err); process.exit(1); });
