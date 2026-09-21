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
const { execFile } = require('child_process');

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

// What the server dialogs will answer, in order, and what they were asked.
const answers = [];
const asked = [];
const informed = [];
const written = [];

function answer(prompt) {
  asked.push(prompt);
  return Promise.resolve(answers.shift());
}

const vscodeStub = {
  EventEmitter: StubEmitter,
  Disposable: { from: (...items) => ({ dispose: () => items.forEach((i) => i.dispose && i.dispose()) }) },
  ConfigurationTarget: { Global: 1, Workspace: 2, WorkspaceFolder: 3 },
  workspace: {
    getConfiguration: () => ({
      get: (key, fallback) => (key in settings ? settings[key] : fallback),
      inspect: () => ({ globalValue: settings.servers }),
      update: (key, value, target) => {
        written.push({ key, value, target });
        settings[key] = value;
        return Promise.resolve();
      },
    }),
  },
  commands: { executeCommand: () => Promise.resolve() },
  window: {
    createOutputChannel: () => ({ appendLine: (line) => logLines.push(line), show: () => {}, dispose: () => {} }),
    showInputBox: (options) => answer(options.title || options.prompt),
    showQuickPick: (items, options) => {
      asked.push(options.title || options.placeHolder);
      const wanted = answers.shift();
      return Promise.resolve(items.find((item) => item.label === wanted));
    },
    showInformationMessage: (message) => {
      informed.push(message);
      return Promise.resolve(undefined);
    },
    showWarningMessage: (message) => {
      informed.push(message);
      return Promise.resolve(undefined);
    },
  },
};

const originalResolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  if (request === 'vscode') return 'vscode';
  return originalResolve.call(this, request, ...rest);
};
require.cache.vscode = { id: 'vscode', filename: 'vscode', loaded: true, exports: vscodeStub };

const { SlurmClient } = require('../out/client.js');
const { resolveCollector } = require('../out/resolve.js');
const { readServers, readMergeSettings } = require('../out/servers.js');
const { mergeSnapshots, aggregateState } = require('../out/merge.js');
const { ClusterClient } = require('../out/cluster.js');
const { addServer } = require('../out/serverui.js');
const { pushedCollector, shellQuote, shortArgv } = require('../out/remote.js');
const { splitCommand, commandFor } = require('../out/servers.js');

const extensionPath = path.join(__dirname, '..');
const log = vscodeStub.window.createOutputChannel();

/** The spec for "this machine", which is what an empty settings file means. */
function localSpec(over) {
  return Object.assign({ id: 'default', name: '', command: [], pythonPath: '', enabled: true }, over || {});
}

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

/**
 * The settings layer: what `slurmTop.servers` turns into.
 *
 * Run before anything spawns, because a mistake here decides which command a
 * collector is started with.
 */
function checkServerSettings() {
  console.log('server settings:');
  settings = {};
  const unset = readServers();
  check('an unset list still yields one server', unset.length === 1, JSON.stringify(unset));
  check('and it is the machine this runs on', unset[0].command.length === 0);
  check('which is labelled by its hostname, not by the default row name',
    unset[0].name === '', JSON.stringify(unset[0]));

  settings = { command: ['ssh', 'login01', 'slurm-top', '--json'] };
  check('the old command setting still describes it',
    readServers()[0].command.join(' ') === 'ssh login01 slurm-top --json');

  // The list is the whole story: one row per server, this machine included.
  settings = {
    servers: {
      'this machine': 'here',
      beta: 'me@login02',
      gamma: 'ssh -J jump login03 /opt/bin/slurm-top --json',
      delta: 'docker exec slurm slurm-top --json',
    },
  };
  const specs = readServers();
  check('one spec per row, in the order the map holds them',
    specs.map((s) => s.id).join(',') === 'this-machine,beta,gamma,delta',
    specs.map((s) => s.id).join(','));
  check('the `here` row is resolved locally', specs[0].command.length === 0);
  check('one word is an ssh destination',
    specs[1].command.join(' ') === 'ssh -o BatchMode=yes me@login02 slurm-top --json',
    specs[1].command.join(' '));
  check('anything with a space is a command line, used as it stands',
    specs[2].command.join('|') === 'ssh|-J|jump|login03|/opt/bin/slurm-top|--json',
    specs[2].command.join('|'));
  check('including one that is not ssh at all',
    specs[3].command.join(' ') === 'docker exec slurm slurm-top --json');
  check('the name is what the tables show', specs[1].name === 'beta');

  // Naming the local row says "call it this"; leaving the generic name says
  // "call it whatever it calls itself".
  settings = { servers: { metis: 'here', beta: 'me@login02' } };
  const named = readServers();
  check('a renamed local row keeps its name', named[0].name === 'metis', named[0].name);
  check('and is still run locally', named[0].command.length === 0);
  settings = { servers: { metis: 'here' }, command: ['my-wrapper', '--json'] };
  check('a local row honours slurmTop.command', readServers()[0].command.join(' ') === 'my-wrapper --json');

  settings = { servers: {} };
  check('emptying the list brings this machine back',
    readServers().length === 1 && readServers()[0].command.length === 0);

  // Names differing only in punctuation still need ids of their own, or pins
  // and selections on one would follow the other.
  settings = { servers: { 'login 01': 'a@b', 'login-01': 'c@d' } };
  const duplicated = readServers();
  check('names that slug the same get distinct ids', duplicated[0].id !== duplicated[1].id,
    duplicated.map((s) => s.id).join(','));

  settings = { servers: { '  ': 'me@login01', real: 'me@login02' } };
  check('a nameless row is ignored', readServers().length === 1);

  console.log('\naddresses:');
  const addresses = [
    // What people actually type for "the machine I am already on". `here` is
    // the word the dialogs use, so it has to mean it.
    ['', ''],
    ['  ', ''],
    ['here', ''],
    ['Here', ''],
    ['local', ''],
    ['this machine', ''],
    // One word: a destination, reached with options we choose.
    ['login01', 'ssh -o BatchMode=yes login01 slurm-top --json'],
    ['me@login01', 'ssh -o BatchMode=yes me@login01 slurm-top --json'],
    // A whole ssh line that stops at the destination is a login shell, not a
    // collector, so it gets finished off the same way the bare word does.
    ['ssh me@login01', 'ssh me@login01 slurm-top --json'],
    ['ssh -J jump me@login02', 'ssh -J jump me@login02 slurm-top --json'],
    ['ssh -tt login01', 'ssh -tt login01 slurm-top --json'],
    ['ssh -o BatchMode=yes login01', 'ssh -o BatchMode=yes login01 slurm-top --json'],
    // One that already says what to run is left exactly as it is.
    ['ssh login01 /opt/bin/slurm-top --json', 'ssh login01 /opt/bin/slurm-top --json'],
    ['docker exec slurm slurm-top --json', 'docker exec slurm slurm-top --json'],
  ];
  for (const [where, expected] of addresses) {
    const got = commandFor(where).join(' ');
    check(`${JSON.stringify(where)} -> ${expected || '(here)'}`, got === expected, got);
  }

  settings = { mergeServers: { jobs: false, nonsense: true } };
  const merge = readMergeSettings();
  check('merge settings default sensibly', merge.summary === false && merge.nodes === true,
    JSON.stringify(merge));
  check('an explicit merge setting wins', merge.jobs === false);
  check('an unknown section is ignored', !('nonsense' in merge));
  settings = {};
}

/**
 * The add-server dialog.
 *
 * It is the only way most people will ever write `slurmTop.servers`, so what it
 * produces matters as much as what the parser accepts.
 */
async function checkAddServer() {
  console.log('\nadding a server:');
  // Someone who already watches a remote cluster through the old setting: the
  // list takes over completely, so that server has to be carried into it or it
  // would silently stop being watched.
  settings = {};
  written.length = 0;
  answers.push('alpha', 'me@login01');
  check('the dialog reported success', (await addServer()) === true);
  const map = written[0].value;
  check('the settings were written to the user scope', written[0].target === 1, String(written[0].target));
  check('the new row is the address as typed', map.alpha === 'me@login01', JSON.stringify(map));
  // Adding the second server writes the first one out, rather than leaving it
  // implicit in the default and then dropping it.
  check('the row that was there by default is written out too',
    map['this machine'] === 'here', JSON.stringify(map));
  check('and both are reachable as specs', readServers().length === 2);

  // Two questions, whatever the answer is: a command line is just another
  // thing to type in the same box.
  written.length = 0;
  answers.push('box', 'docker exec slurm slurm-top --json');
  await addServer();
  check('a typed command is stored as typed',
    written[0].value.box === 'docker exec slurm slurm-top --json', JSON.stringify(written[0].value));
  check('and becomes argv when it is read back',
    readServers().slice(-1)[0].command.join('|') === 'docker|exec|slurm|slurm-top|--json');

  // Backing out of either step must leave the settings alone.
  written.length = 0;
  answers.push('gamma', undefined);
  check('cancelling adds nothing', (await addServer()) === false && written.length === 0);
  answers.length = 0;
  settings = {};
}

/**
 * Running the collector on a machine that has not got it.
 *
 * Proved against `localhost` over a real ssh connection, because the point of
 * the feature is that the far side needs nothing: if this passes, the same
 * command works on any cluster whose ssh lets us in and whose python3 runs.
 * Skipped -- not failed -- where ssh to localhost is not set up, since that is
 * a property of the machine the tests run on and not of the code.
 */
async function checkPushedCollector() {
  console.log('\nsending the collector over ssh:');
  const prefix = ['ssh', '-o', 'BatchMode=yes', 'localhost'];
  const argv = pushedCollector(prefix, extensionPath, 'python3');
  check('a command is built from the bundled copy',
    Array.isArray(argv) && argv.slice(0, 6).join(' ') === `${prefix.join(' ')} python3 -c`,
    JSON.stringify(argv && argv.slice(0, 6)));
  if (!argv) {
    return;
  }
  // Two arguments, both quoted for the shell on the far side: the program, and
  // the modules it rebuilds itself from. Nothing is written over there.
  check('the whole collector rides in one argument',
    argv.length === 8 && argv[7].length > 1000, `${argv.length} args, payload ${argv[7]?.length}`);
  check('both are quoted for the remote shell',
    argv[6].startsWith("'") && argv[6].endsWith("'") && argv[7].startsWith("'"));
  check('a log line does not carry the payload',
    shortArgv(argv).length < 200 && shortArgv(argv).includes('<'), shortArgv(argv));
  // The POSIX way of getting a quote inside single quotes: close, escape, reopen.
  check('shell quoting survives a quote', shellQuote("it's") === "'it'\\''s'", shellQuote("it's"));

  const reachable = await new Promise((resolve) => {
    execFile('ssh', [...prefix.slice(1), 'true'], { timeout: 20000 }, (err) => resolve(!err));
  });
  if (!reachable) {
    console.log('  skip ssh to localhost is not set up here');
    return;
  }

  const { json, why } = await new Promise((resolve) => {
    execFile(argv[0], argv.slice(1), { timeout: 120000, maxBuffer: 64 * 1024 * 1024 }, (err, out, errOut) =>
      resolve({ json: err ? null : out, why: err ? `${err.message} ${errOut}`.slice(0, 200) : '' })
    );
  });
  check('the far side answered with a snapshot', !!json, why);
  if (json) {
    const snapshot = JSON.parse(json);
    check('it is the shape the extension expects',
      snapshot.schema === 1 && Array.isArray(snapshot.jobs) && Array.isArray(snapshot.nodes),
      Object.keys(snapshot).join(','));
    check('and it names the machine it came from', typeof snapshot.host === 'string' && !!snapshot.host,
      snapshot.host);
  }
}

/** The merge layer: two clusters in one snapshot. */
function checkMerge(snapshot) {
  console.log('\nmerge:');
  const entries = [
    { id: 'alpha', name: 'alpha', state: 'running', snapshot },
    { id: 'beta', name: '', state: 'running', snapshot },
  ];
  const merged = mergeSnapshots(entries);
  check('rows from both servers are present',
    merged.jobs.length === snapshot.jobs.length * 2 && merged.nodes.length === snapshot.nodes.length * 2);
  check('every row is tagged with its server',
    merged.jobs.every((j) => j.server === 'alpha' || j.server === 'beta'));
  check('a server with no label falls back to its hostname',
    merged.servers[1].name === snapshot.host, merged.servers[1].name);
  check('GPU totals are summed', merged.gpu.total === (snapshot.gpu.total || 0) * 2,
    `${merged.gpu.total} vs ${(snapshot.gpu.total || 0) * 2}`);
  check('job statistics are summed',
    merged.summary.all.running.jobs === snapshot.summary.all.running.jobs * 2);
  check('each server keeps its own totals for a separated panel',
    merged.servers[0].gpu.total === snapshot.gpu.total);

  // One sick server among several must be reported without hiding the rest.
  const partly = aggregateState([
    { id: 'alpha', name: 'alpha', state: 'running', snapshot },
    { id: 'beta', name: 'beta', state: 'error', message: 'ssh: connection refused' },
  ]);
  check('one failing server does not stop the view',
    partly.state === 'running' && partly.message.includes('beta'), JSON.stringify(partly));
  const allBroken = aggregateState([
    { id: 'alpha', name: 'alpha', state: 'error', message: 'boom' },
    { id: 'beta', name: 'beta', state: 'error', message: 'boom' },
  ]);
  check('all of them failing is an error', allBroken.state === 'error', JSON.stringify(allBroken));

  const alone = mergeSnapshots([{ id: 'default', name: '', state: 'running', snapshot }]);
  check('one server merges to the same shape it came in as',
    alone.jobs.length === snapshot.jobs.length && alone.host === snapshot.host &&
      alone.gpu.total === snapshot.gpu.total);
}

async function main() {
  checkServerSettings();
  await checkAddServer();

  await checkPushedCollector();

  console.log('\nresolver:');
  const collector = await resolveCollector(localSpec(), extensionPath, log);
  check('found a collector', !!collector, logLines.join(' | '));
  if (collector) console.log(`       -> ${collector.origin}`);

  // With PATH emptied and no interpreter of its own, only the bundled copy can
  // work -- which is the fallback remote users will actually hit.
  console.log('\nbundled fallback:');
  const realPath = process.env.PATH;
  process.env.PATH = '/nonexistent';
  const fallbackLog = vscodeStub.window.createOutputChannel();
  const fallback = await resolveCollector(localSpec({ pythonPath: '/usr/bin/python3' }), extensionPath, fallbackLog);
  process.env.PATH = realPath;
  check('falls back to the bundled copy', !!fallback && !!fallback.env && !!fallback.env.PYTHONPATH,
    fallback ? fallback.origin : 'nothing resolved');
  settings = {};

  console.log('\nstreaming client:');
  settings = { refreshInterval: 1 };
  const client = new SlurmClient(localSpec({ name: 'here' }), extensionPath, log);
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
  check('log lines say which server they are about', logLines.some((l) => l.startsWith('[here]')),
    logLines.slice(-1).join(''));

  checkMerge(snapshots[0]);

  console.log('\ndetail lookup:');
  if (snapshots[0].nodes.length) {
    const detail = await client.fetchDetail('node', snapshots[0].nodes[0].name);
    check('node detail returns the right kind', detail.kind === 'node');
    check('node detail has jobs array', Array.isArray(detail.jobs));
  }
  if (snapshots[0].jobs.length) {
    const jobId = snapshots[0].jobs[0].job_id;
    const detail = await client.fetchDetail('job', jobId);
    check('job detail returns the right kind', detail.kind === 'job');
    check('job detail carries the usage metrics', Array.isArray(detail.metrics), JSON.stringify(detail.metrics));
    check('job detail names both output streams',
      !!detail.output && 'stdout' in detail.output && 'stderr' in detail.output,
      JSON.stringify(detail.output));

    // The tail is a second, explicit request; a detail lookup must not carry it.
    const output = await client.fetchJobOutput(jobId, 'stdout', 5);
    check('job output returns the right kind', output.kind === 'job-output', JSON.stringify(output).slice(0, 120));
    check('job output carries a tail block', typeof output.output?.text === 'string');
    check('an unreadable output still explains itself',
      output.output.exists || !!output.output.error,
      'the view shows the reason rather than an empty box');
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

  await checkCluster();
}

/**
 * Two collectors at once, against the real cluster.
 *
 * Both are the local one -- there is no second cluster to test against -- but
 * they are two independent processes merged into one snapshot, which is what
 * the routing and the merge have to survive.
 */
async function checkCluster() {
  console.log('\ncluster of two:');
  settings = { refreshInterval: 1, servers: { alpha: 'here', beta: 'here' } };
  const cluster = new ClusterClient(extensionPath, log);
  const merged = [];
  cluster.onSnapshot((snapshot) => merged.push(snapshot));
  await cluster.start();
  await waitFor(
    () => merged.some((m) => m.servers.every((s) => s.state === 'running')),
    40000,
    'both servers reporting'
  );
  const latest = cluster.latest;
  check('both servers are in the snapshot', latest.servers.length === 2,
    latest.servers.map((s) => s.id).join(','));
  check('each keeps the name it was given',
    latest.servers.map((s) => s.name).join(',') === 'alpha,beta', latest.servers.map((s) => s.name).join(','));
  check('every node is tagged with the server it came from',
    latest.nodes.length > 0 && latest.nodes.every((n) => n.server === 'alpha' || n.server === 'beta'));
  check('the same node name appears once per server',
    latest.nodes.filter((n) => n.name === latest.nodes[0].name).length === 2,
    latest.nodes[0].name);

  // A click has to reach the collector that knows the row.
  const node = latest.nodes.find((n) => n.server === 'beta');
  const detail = await cluster.fetchDetail({ server: 'beta', kind: 'node', id: node.name });
  check('a detail lookup is routed to the named server', detail.kind === 'node' && detail.node === node.name);
  let rejected = false;
  await cluster.fetchDetail({ server: 'nowhere', kind: 'node', id: node.name }).catch(() => (rejected = true));
  check('a lookup for an unwatched server is refused, not guessed', rejected);

  // Dropping one server must leave the other's process alone.
  settings = { refreshInterval: 1, servers: { alpha: 'here' } };
  const changed = await cluster.reload();
  check('reload reports that the set changed', changed === true);
  check('only the remaining server is watched',
    cluster.servers.map((spec) => spec.id).join(',') === 'alpha', cluster.servers.map((s) => s.id).join(','));
  check('and it still has its data', !!cluster.latest.servers[0].timestamp);
  check('a reload with no change is a no-op', (await cluster.reload()) === false);

  cluster.dispose();
  settings = {};
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
