/**
 * Headless check of the extension-host side (resolve.ts + client.ts).
 *
 * VS Code cannot run here, so `require('vscode')` is stubbed with just the
 * surface those modules touch. This exercises the real spawn, the real NDJSON
 * framing and the real restart path against the real cluster.
 *
 * Run with: node test/host.smoke.js
 */
const Module = require('module');
const path = require('path');

let failures = 0;
function check(label, condition, detail) {
  if (condition) console.log(`  ok   ${label}`);
  else {
    failures += 1;
    console.log(`  FAIL ${label}${detail ? ` (${detail})` : ''}`);
  }
}

// ------------------------------------------------------------- vscode stub

let settings = {};

class StubEmitter {
  constructor() { this.listeners = []; }
  get event() { return (listener) => { this.listeners.push(listener); return { dispose: () => {} }; }; }
  fire(value) { for (const listener of this.listeners.slice()) listener(value); }
  dispose() { this.listeners = []; }
}

const logLines = [];
const vscodeStub = {
  EventEmitter: StubEmitter,
  Disposable: { from: (...items) => ({ dispose: () => items.forEach((i) => i.dispose && i.dispose()) }) },
  workspace: {
    getConfiguration: () => ({ get: (key, fallback) => (key in settings ? settings[key] : fallback) }),
  },
  window: { createOutputChannel: () => ({ appendLine: (line) => logLines.push(line), show: () => {}, dispose: () => {} }) },
};

const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return 'vscode';
  return originalResolve.call(this, request, ...rest);
};
require.cache.vscode = { id: 'vscode', filename: 'vscode', loaded: true, exports: vscodeStub };

const { SlurmClient } = require('../out/client.js');
const { resolveCollector } = require('../out/resolve.js');

const extensionPath = path.join(__dirname, '..');
const log = vscodeStub.window.createOutputChannel();

function waitFor(predicate, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = () => {
      if (predicate()) resolve();
      else if (Date.now() - started > timeoutMs) reject(new Error(`timed out waiting for ${label}`));
      else setTimeout(tick, 50);
    };
    tick();
  });
}

async function main() {
  console.log('resolver:');
  const collector = await resolveCollector(extensionPath, log);
  check('found a collector', !!collector, logLines.join(' | '));
  if (collector) console.log(`       -> ${collector.origin}`);

  // With PATH emptied and no interpreter of its own, only the bundled copy can
  // work -- which is the fallback remote users will actually hit.
  console.log('\nbundled fallback:');
  const realPath = process.env.PATH;
  process.env.PATH = '/nonexistent';
  const fallbackLog = vscodeStub.window.createOutputChannel();
  settings = { pythonPath: '/usr/bin/python3' };
  const fallback = await resolveCollector(extensionPath, fallbackLog);
  process.env.PATH = realPath;
  check('falls back to the bundled copy', !!fallback && !!fallback.env && !!fallback.env.PYTHONPATH,
    fallback ? fallback.origin : 'nothing resolved');
  settings = {};

  console.log('\nstreaming client:');
  settings = { refreshInterval: 1 };
  const client = new SlurmClient(extensionPath, log);
  const snapshots = [];
  const states = [];
  client.onSnapshot((snapshot) => snapshots.push(snapshot));
  client.onState((s) => states.push(s.state));
  await client.start();
  await waitFor(() => snapshots.length >= 2, 30000, 'two snapshots');
  check('received successive snapshots', snapshots.length >= 2);
  check('snapshot timestamps advance', snapshots[1].timestamp > snapshots[0].timestamp);
  check('snapshot carries jobs and nodes', Array.isArray(snapshots[0].jobs) && snapshots[0].nodes.length > 0);
  check('reported running', states.includes('running'));
  check('client caches the latest snapshot', client.latest === snapshots[snapshots.length - 1]);

  console.log('\ndetail lookup:');
  if (snapshots[0].nodes.length) {
    const detail = await client.fetchDetail('node', snapshots[0].nodes[0].name);
    check('node detail returns the right kind', detail.kind === 'node');
    check('node detail has jobs array', Array.isArray(detail.jobs));
  }
  if (snapshots[0].jobs.length) {
    const detail = await client.fetchDetail('job', snapshots[0].jobs[0].job_id);
    check('job detail returns the right kind', detail.kind === 'job');
  }

  console.log('\nlifecycle:');
  const before = snapshots.length;
  client.stop();
  await new Promise((r) => setTimeout(r, 2500));
  check('stop halts the stream', snapshots.length === before, `${before} -> ${snapshots.length}`);
  await client.start();
  await waitFor(() => snapshots.length > before, 30000, 'a snapshot after restart');
  check('start resumes the stream', snapshots.length > before);

  // A collector killed from outside must be brought back automatically.
  const beforeKill = snapshots.length;
  // Reaching into the private field on purpose: this simulates the collector
  // being killed from outside, which is exactly what the restart path is for.
  client.child.kill('SIGKILL');
  await waitFor(() => snapshots.length > beforeKill, 40000, 'recovery after the child is killed');
  check('recovers when the collector dies', snapshots.length > beforeKill);

  client.dispose();
  await new Promise((r) => setTimeout(r, 500));
  check('dispose leaves no child running', !client.child);
}

main()
  .then(() => {
    console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
    process.exit(failures ? 1 : 0);
  })
  .catch((err) => {
    console.error('\nerror:', err);
    process.exit(1);
  });
