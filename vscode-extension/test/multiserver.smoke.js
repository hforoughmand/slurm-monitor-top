/**
 * Headless check that media/main.js draws two clusters at once.
 *
 * The fixture snapshot is one server's; this builds a merged one out of two
 * copies of it -- with colliding job ids and node names, because that is the
 * case a single-cluster view gets wrong -- and asserts the things that only
 * matter once there is more than one: the SERVER column, the server filter,
 * one panel or box per server where configured, and clicks that carry the
 * server back to the host.
 *
 * Run with: node test/multiserver.smoke.js
 */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const base = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'snapshot.json'), 'utf8'));
const script = fs.readFileSync(path.join(__dirname, '..', 'media', 'main.js'), 'utf8');

let failures = 0;
function check(label, condition, detail) {
  if (condition) {
    console.log(`  ok   ${label}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${label}${detail ? ` (${detail})` : ''}`);
  }
}

const SERVERS = [
  { id: 'alpha', name: 'alpha', user: 'alice', pinned: ['1003'] },
  { id: 'beta', name: 'beta', user: 'bob', pinned: [] },
];

function tag(rows, server) {
  return rows.map((row) => Object.assign({}, row, { server: server.id, server_name: server.name }));
}

function sumSummary(parts) {
  const out = {};
  for (const bucket of ['all', 'me', 'others']) {
    out[bucket] = {};
    for (const phase of ['running', 'pending']) {
      const cell = { jobs: 0, cpus: 0, mem_mb: 0, gpus: 0 };
      for (const part of parts) {
        const from = (part[bucket] || {})[phase] || {};
        for (const key of Object.keys(cell)) cell[key] += Number(from[key]) || 0;
      }
      out[bucket][phase] = cell;
    }
  }
  return out;
}

/**
 * Two clusters that deliberately share every job id and node name, so a view
 * that keyed on the id alone would collapse the second one into the first.
 */
function mergedSnapshot(options) {
  const states = options?.states || {};
  const servers = SERVERS.map((server) => ({
    id: server.id,
    name: server.name,
    host: `${server.id}-login01`,
    user: server.user,
    state: states[server.id] || 'running',
    message: states[server.id] === 'error' ? 'ssh: connection refused' : undefined,
    timestamp: base.timestamp,
    gpu: base.gpu,
    summary: base.summary,
    pinned: server.pinned,
    counts: { jobs: base.jobs.length, nodes: base.nodes.length, disks: base.disks.length },
  }));
  const gpu = JSON.parse(JSON.stringify(base.gpu));
  for (const stats of Object.values(gpu.per_type_stats)) {
    for (const key of Object.keys(stats)) stats[key] *= 2;
  }
  return Object.assign({}, base, {
    servers,
    jobs: [].concat(...servers.map((s) => tag(base.jobs, s))),
    nodes: [].concat(...servers.map((s) => tag(base.nodes, s))),
    disks: [].concat(...servers.map((s) => tag(base.disks, s))),
    gpu,
    summary: sumSummary([base.summary, base.summary]),
  });
}

function makeWindow(variant) {
  const dom = new JSDOM(
    `<body data-variant="${variant}">
       <div id="status" class="status"></div><div id="root"></div>
       <div id="overlay" class="overlay hidden"></div>
     </body>`,
    { runScripts: 'outside-only' }
  );
  const { window } = dom;
  window.eval(`
    var __posted = [];
    var __state = null;
    function acquireVsCodeApi() {
      return {
        postMessage: function (m) { __posted.push(m); },
        getState: function () { return __state; },
        setState: function (s) { __state = s; }
      };
    }
  `);
  window.eval(script);
  const send = (message) => window.dispatchEvent(new window.MessageEvent('message', { data: message }));
  return { window, send, posted: () => window.eval('__posted') };
}

const ALL = ['summary', 'jobs', 'nodes', 'gpus', 'disks'];

function config(merge) {
  return {
    type: 'config',
    sections: ALL,
    ownerFilter: 'all',
    interval: 3,
    detailsIn: 'overlay',
    servers: SERVERS.map((s) => ({ id: s.id, name: s.name })),
    merge,
  };
}

function headersOf(panel) {
  return Array.from(panel.querySelectorAll('thead th')).map((th) => th.textContent);
}

function runMerged() {
  console.log('\nmerged panels:');
  const { window, send, posted: postedOf } = makeWindow('dashboard');
  const doc = window.document;
  const snapshot = mergedSnapshot();

  send(config({ summary: true, jobs: true, nodes: true, gpus: true, disks: true }));
  send({ type: 'snapshot', snapshot });

  check('one panel per section', doc.querySelectorAll('#root .panel').length === ALL.length,
    String(doc.querySelectorAll('#root .panel').length));

  for (const section of ['jobs', 'nodes', 'disks']) {
    const panel = doc.querySelector(`[data-section='${section}']`);
    const rows = panel.querySelectorAll('tbody tr');
    const expected = snapshot[section].length;
    check(`${section} shows both clusters' rows`, rows.length === expected, `${rows.length} vs ${expected}`);
    check(`${section} has a SERVER column`, headersOf(panel).includes('SERVER'), headersOf(panel).join(','));
    const index = headersOf(panel).indexOf('SERVER');
    const names = new Set(Array.from(rows).map((tr) => tr.querySelectorAll('td')[index].textContent));
    check(`${section} names both clusters`, names.has('alpha') && names.has('beta'),
      Array.from(names).join(','));
    check(`${section} cells match the header count`,
      rows[0].querySelectorAll('td').length === headersOf(panel).length);
  }

  // The case a single-cluster view gets wrong: the same id on both servers.
  const jobsPanel = doc.querySelector("[data-section='jobs']");
  const ids = Array.from(jobsPanel.querySelectorAll('tbody tr')).map((tr) => tr.dataset.id);
  check('a job id present on both servers appears twice',
    ids.filter((id) => id === '1001').length === 2, ids.join(','));
  const uids = Array.from(jobsPanel.querySelectorAll('tbody tr')).map((tr) => tr.dataset.uid);
  check('but each row has an identity of its own', new Set(uids).size === uids.length);

  // GPUs are summed per model across the clusters that have them.
  const gpuPanel = doc.querySelector("[data-section='gpus']");
  const gpuRows = Array.from(gpuPanel.querySelectorAll('tbody tr'));
  check('GPU models are merged, not listed twice',
    gpuRows.length === Object.keys(base.gpu.per_type_stats).length, String(gpuRows.length));
  const totalIndex = headersOf(gpuPanel).indexOf('TOTAL');
  const a100 = gpuRows.find((tr) => tr.dataset.id === 'a100');
  check('a merged GPU row adds the clusters up',
    Number(a100.querySelectorAll('td')[totalIndex].textContent) ===
      base.gpu.per_type_stats.a100.total * 2,
    a100.querySelectorAll('td')[totalIndex].textContent);
  const serverIndex = headersOf(gpuPanel).indexOf('SERVER');
  check('and says which clusters it came from',
    a100.querySelectorAll('td')[serverIndex].textContent === 'alpha, beta',
    a100.querySelectorAll('td')[serverIndex].textContent);

  // The server filter is the merged jobs panel's own control.
  const selects = jobsPanel.querySelectorAll('.panel-head select');
  check('the jobs panel gains a server filter', selects.length === 3, String(selects.length));
  const serverSelect = selects[0];
  check('the filter lists every server plus "all"',
    Array.from(serverSelect.options).map((o) => o.value).join(',') === 'all,alpha,beta',
    Array.from(serverSelect.options).map((o) => o.value).join(','));
  serverSelect.value = 'beta';
  serverSelect.dispatchEvent(new window.Event('change'));
  check('filtering by server halves the table',
    jobsPanel.querySelectorAll('tbody tr').length === base.jobs.length,
    String(jobsPanel.querySelectorAll('tbody tr').length));
  check('and leaves only that server',
    Array.from(jobsPanel.querySelectorAll('tbody tr')).every((tr) => tr.dataset.server === 'beta'));
  serverSelect.value = 'all';
  serverSelect.dispatchEvent(new window.Event('change'));

  // The machines table gets the same cluster menu, ahead of its own filters.
  const nodesPanel = doc.querySelector("[data-section='nodes']");
  const nodeSelects = nodesPanel.querySelectorAll('.panel-head select');
  check('the nodes panel gains a server filter too', nodeSelects.length === 4,
    String(nodeSelects.length));
  const nodeServerSelect = nodeSelects[0];
  check('it lists every server plus "all"',
    Array.from(nodeServerSelect.options).map((o) => o.value).join(',') === 'all,alpha,beta',
    Array.from(nodeServerSelect.options).map((o) => o.value).join(','));
  nodeServerSelect.value = 'alpha';
  nodeServerSelect.dispatchEvent(new window.Event('change'));
  check('filtering machines by server leaves only that cluster',
    Array.from(nodesPanel.querySelectorAll('tbody tr')).every((tr) => tr.dataset.server === 'alpha') &&
    nodesPanel.querySelectorAll('tbody tr').length === base.nodes.length,
    String(nodesPanel.querySelectorAll('tbody tr').length));
  nodeServerSelect.value = 'all';
  nodeServerSelect.dispatchEvent(new window.Event('change'));
  check('the node and job server filters are independent',
    nodesPanel.querySelectorAll('tbody tr').length === snapshot.nodes.length &&
    jobsPanel.querySelectorAll('tbody tr').length === snapshot.jobs.length);

  // "me" means a different account on each cluster.
  const ownerSelect = selects[1];
  ownerSelect.value = 'me';
  ownerSelect.dispatchEvent(new window.Event('change'));
  const mine = snapshot.jobs.filter((j) => j.user === (j.server === 'alpha' ? 'alice' : 'bob')).length;
  check('"me" is resolved per server, not against one global user',
    jobsPanel.querySelectorAll('tbody tr').length === mine,
    `${jobsPanel.querySelectorAll('tbody tr').length} vs ${mine}`);
  ownerSelect.value = 'all';
  ownerSelect.dispatchEvent(new window.Event('change'));

  // Pins are per server: the same id pinned on alpha must not star beta's job.
  const pinIndex = 0;
  const pinned = Array.from(jobsPanel.querySelectorAll('tbody tr'))
    .filter((tr) => tr.querySelectorAll('td')[pinIndex].className.includes('pinned'));
  check('a pin applies to one server only',
    pinned.length === 1 && pinned[0].dataset.server === 'alpha',
    pinned.map((tr) => tr.dataset.uid).join(','));
  check('and the pinned row still leads the table',
    jobsPanel.querySelector('tbody tr').dataset.uid === 'alpha/1003',
    jobsPanel.querySelector('tbody tr').dataset.uid);

  const betaRow = Array.from(jobsPanel.querySelectorAll('tbody tr')).find((tr) => tr.dataset.server === 'beta');
  betaRow.querySelectorAll('td')[pinIndex].dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  const pinMessage = postedOf().filter((m) => m.type === 'togglePin').pop();
  check('pinning tells the host which server', pinMessage && pinMessage.server === 'beta',
    JSON.stringify(pinMessage));

  // Opening a row has to name the server too, or the host cannot route it.
  betaRow.dispatchEvent(new window.Event('dblclick'));
  const open = postedOf().filter((m) => m.type === 'openDetail').pop();
  check('opening a row names its server', open && open.server === 'beta', JSON.stringify(open));
  check('the loading overlay names the server too',
    doc.getElementById('overlay').textContent.includes('beta'),
    doc.getElementById('overlay').textContent.slice(0, 60));

  // The merged summary box lists the clusters behind its totals.
  const summary = doc.querySelector("[data-section='summary']");
  check('the merged summary is titled for all of them',
    summary.querySelector('.panel-title').textContent.includes('All servers'),
    summary.querySelector('.panel-title').textContent);
  const chips = summary.querySelectorAll('.server-chip');
  check('and lists one chip per cluster', chips.length === 2, String(chips.length));
  check('a merged total is the sum of both',
    summary.textContent.includes(String(
      (base.summary.all.running.jobs + base.summary.all.running.jobs)
    )));

  // A dead server must be visible rather than silently missing from the totals.
  send({ type: 'snapshot', snapshot: mergedSnapshot({ states: { beta: 'error' } }) });
  const broken = Array.from(summary.querySelectorAll('.server-chip')).find((c) => c.textContent.startsWith('beta'));
  check('a failing server is called out in the chips',
    broken && broken.textContent.includes('error') && broken.className.includes('error-text'),
    broken && broken.textContent);
}

function runSeparate() {
  console.log('\none panel per server:');
  const { window, send } = makeWindow('dashboard');
  const doc = window.document;
  const snapshot = mergedSnapshot();

  send(config({ summary: false, jobs: false, nodes: true, gpus: true, disks: true }));
  send({ type: 'snapshot', snapshot });

  const summaries = doc.querySelectorAll("[data-section='summary']");
  check('the summary becomes one box per server', summaries.length === 2, String(summaries.length));
  check('each box is titled after its server',
    Array.from(summaries).map((p) => p.querySelector('.panel-title').textContent).join(',') === 'alpha,beta',
    Array.from(summaries).map((p) => p.querySelector('.panel-title').textContent).join(','));

  const jobPanels = doc.querySelectorAll("[data-section='jobs']");
  check('jobs split into one panel per server', jobPanels.length === 2, String(jobPanels.length));
  check('each jobs panel names its server in the title',
    jobPanels[1].querySelector('.panel-title').textContent === 'Jobs · beta',
    jobPanels[1].querySelector('.panel-title').textContent);
  check('a per-server panel holds only that server',
    Array.from(jobPanels[1].querySelectorAll('tbody tr')).every((tr) => tr.dataset.server === 'beta'));
  check('and drops the SERVER column it does not need',
    !headersOf(jobPanels[1]).includes('SERVER'), headersOf(jobPanels[1]).join(','));
  check('a separated panel shows every row of its own server',
    jobPanels[0].querySelectorAll('tbody tr').length === base.jobs.length,
    String(jobPanels[0].querySelectorAll('tbody tr').length));

  // Sections left merged keep one panel with the column.
  check('a section left merged stays one panel', doc.querySelectorAll("[data-section='nodes']").length === 1);
  check('and keeps its SERVER column',
    headersOf(doc.querySelector("[data-section='nodes']")).includes('SERVER'));

  // A GPU panel scoped to one server counts only that server's cards.
  const gpuPanels = doc.querySelectorAll("[data-section='gpus']");
  check('gpus stay merged when configured so', gpuPanels.length === 1);

  // Switching the setting must rebuild the layout, not leave stale panels.
  send(config({ summary: true, jobs: true, nodes: true, gpus: true, disks: true }));
  check('turning merging back on rebuilds one panel per section',
    doc.querySelectorAll('#root .panel').length === ALL.length,
    String(doc.querySelectorAll('#root .panel').length));
}

function runSingleServer() {
  console.log('\nback to one server:');
  const { window, send } = makeWindow('dashboard');
  const doc = window.document;

  // One server configured, and every section set to "separate": with a single
  // cluster that has to look exactly like the view always did.
  send(Object.assign(config({ summary: false, jobs: false, nodes: false, gpus: false, disks: false }), {
    servers: [{ id: 'alpha', name: 'alpha' }],
  }));
  send({
    type: 'snapshot',
    snapshot: Object.assign({}, base, {
      servers: [{ id: 'alpha', name: 'alpha', host: 'login01', user: 'alice', state: 'running',
        timestamp: base.timestamp, gpu: base.gpu, summary: base.summary, pinned: base.pinned,
        counts: { jobs: base.jobs.length, nodes: base.nodes.length, disks: base.disks.length } }],
      jobs: base.jobs.map((j) => Object.assign({}, j, { server: 'alpha', server_name: 'alpha' })),
      nodes: base.nodes.map((n) => Object.assign({}, n, { server: 'alpha', server_name: 'alpha' })),
      disks: base.disks.map((d) => Object.assign({}, d, { server: 'alpha', server_name: 'alpha' })),
    }),
  });

  // "Separate" and "merged" are the same picture with one cluster, so neither
  // the extra panels nor the column that would only ever say "alpha" appear.
  check('one server means one panel per section',
    doc.querySelectorAll('#root .panel').length === ALL.length,
    String(doc.querySelectorAll('#root .panel').length));
  check('and no SERVER column', !headersOf(doc.querySelector("[data-section='jobs']")).includes('SERVER'));
  check('and no server filter',
    doc.querySelectorAll("[data-section='jobs'] .panel-head select").length === 2);
  check('the summary is titled after the cluster',
    doc.querySelector("[data-section='summary'] .panel-title").textContent === 'alpha',
    doc.querySelector("[data-section='summary'] .panel-title").textContent);
}

function runDetailWindow() {
  console.log('\ndetail window on a second server:');
  const { window, send, posted: postedOf } = makeWindow('detail');
  const doc = window.document;

  send(config({ summary: true, jobs: true, nodes: true, gpus: true, disks: true }));
  send({ type: 'detailTarget', server: 'beta', serverName: 'beta', kind: 'node', id: 'gpu01' });
  send({ type: 'pinned', server: 'beta', pinned: [] });
  send({
    type: 'detail',
    server: 'beta',
    serverName: 'beta',
    detail: { kind: 'node', node: 'gpu01', detail: { NodeName: 'gpu01' }, jobs: [], cpu: base.nodes[1].cpu },
  });

  check('the page is titled with the server', doc.querySelector('#root .panel-title').textContent === 'Node gpu01 · beta',
    doc.querySelector('#root .panel-title').textContent);

  const probe = Array.from(doc.querySelectorAll('#root button')).find((b) => b.textContent.includes('Read it from the node'));
  check('an unprobed node still offers to be read', !!probe);
  probe.dispatchEvent(new window.Event('click'));
  const sent = postedOf().filter((m) => m.type === 'probeCpu').pop();
  check('the probe request names the server', sent && sent.server === 'beta', JSON.stringify(sent));
  check('the view shows it is waiting on that node',
    doc.getElementById('root').textContent.includes('Reading the CPU'),
    doc.getElementById('root').textContent.slice(0, 80));

  // The same node name on the other server must not look like this one.
  send({ type: 'cpuProbe', server: 'alpha', node: 'gpu01', state: 'started' });
  send({
    type: 'detail',
    server: 'beta',
    serverName: 'beta',
    detail: { kind: 'node', node: 'gpu01', detail: { NodeName: 'gpu01' }, jobs: [], cpu: base.nodes[1].cpu },
  });
  check('a probe on another server does not claim this node',
    !doc.getElementById('root').textContent.includes('Reading the CPU'),
    doc.getElementById('root').textContent.slice(0, 80));
}

runMerged();
runSeparate();
runSingleServer();
runDetailWindow();

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
