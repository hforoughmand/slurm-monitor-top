/**
 * Headless check that media/main.js survives a real snapshot.
 *
 * It runs the webview script in jsdom with a stub `acquireVsCodeApi`, feeds it
 * the same messages the extension host sends, and asserts the tables actually
 * fill. Run with: node test/webview.smoke.js path/to/snapshot.json
 */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const snapshotPath = process.argv[2] || path.join(__dirname, 'fixtures', 'snapshot.json');
const snapshot = JSON.parse(fs.readFileSync(snapshotPath, 'utf8'));
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

function run(variant, sections, detailsIn) {
  console.log(`\n${variant} (details in ${detailsIn}):`);
  const { window, send, posted: postedOf } = makeWindow(variant);

  send({ type: 'config', sections, ownerFilter: 'all', interval: 3, detailsIn });
  send({ type: 'snapshot', snapshot });

  const doc = window.document;
  check('panels built', doc.querySelectorAll('#root .panel').length === sections.length,
    `${doc.querySelectorAll('#root .panel').length} of ${sections.length}`);

  for (const section of sections) {
    const panel = doc.querySelector(`[data-section='${section}']`);
    check(`${section} panel present`, !!panel);
    if (!panel || section === 'summary') continue;
    const rows = panel.querySelectorAll('tbody tr');
    const expected = { jobs: snapshot.jobs.length, nodes: snapshot.nodes.length, disks: snapshot.disks.length,
      gpus: Object.keys(snapshot.gpu.per_type_stats).length }[section];
    check(`${section} rows rendered`, rows.length === expected, `${rows.length} vs ${expected}`);
    const cells = rows.length ? rows[0].querySelectorAll('td').length : 0;
    const headers = panel.querySelectorAll('thead th').length;
    check(`${section} cell count matches header count`, rows.length === 0 || cells === headers, `${cells} vs ${headers}`);
    check(`${section} first row is not all dashes`,
      rows.length === 0 || Array.from(rows[0].querySelectorAll('td')).some((td) => td.textContent.trim() !== '-'));
  }

  // The machine and disk panels must actually carry the enriched fields, not
  // fall back to placeholders.
  const nodesPanel = doc.querySelector("[data-section='nodes']");
  if (nodesPanel) {
    const headerText = Array.from(nodesPanel.querySelectorAll('thead th')).map((th) => th.textContent);
    check('nodes panel has a GPU used/total column', headerText.includes('GPU U/T'), headerText.join(','));
    const gpuIndex = headerText.indexOf('GPU U/T');
    const gpuRow = Array.from(nodesPanel.querySelectorAll('tbody tr')).find((tr) => tr.dataset.id === 'gpu01');
    check('node GPU cell shows used/total', gpuRow.querySelectorAll('td')[gpuIndex].textContent === '2/4',
      gpuRow.querySelectorAll('td')[gpuIndex].textContent);
    if (headerText.includes('LOAD')) {
      const loadIndex = headerText.indexOf('LOAD');
      const cell = gpuRow.querySelectorAll('td')[loadIndex];
      check('node load cell shows a number', cell.textContent.trim() === '12.40', cell.textContent);
      check('node load cell draws a bar', !!cell.querySelector('.bar'));
    }
    if (headerText.includes('STATE')) {
      const stateIndex = headerText.indexOf('STATE');
      const drained = Array.from(nodesPanel.querySelectorAll('tbody tr')).find((tr) => tr.dataset.id === 'cpu03');
      check('drain reason is shown next to the state',
        drained.querySelectorAll('td')[stateIndex].textContent.includes('disk full'),
        drained.querySelectorAll('td')[stateIndex].textContent);
      check('drained node is styled as a problem',
        drained.querySelectorAll('td')[stateIndex].className.includes('state-failed'));
    }
  }

  const disksPanel = doc.querySelector("[data-section='disks']");
  if (disksPanel) {
    const headerText = Array.from(disksPanel.querySelectorAll('thead th')).map((th) => th.textContent);
    check('disk panel has a free-space column', headerText.includes('FREE'), headerText.join(','));
    const freeIndex = headerText.indexOf('FREE');
    const home = Array.from(disksPanel.querySelectorAll('tbody tr')).find((tr) => tr.dataset.id === '/home');
    check('disk free cell is populated', home.querySelectorAll('td')[freeIndex].textContent === '2T',
      home.querySelectorAll('td')[freeIndex].textContent);
    const useCell = home.querySelectorAll('td')[headerText.indexOf('USE%')];
    check('a nearly full disk gets the high-usage bar', useCell.querySelector('.bar').className.includes('high'),
      useCell.querySelector('.bar').className);
    // Sorting a size column must use bytes, not the "2T" / "176G" label: by
    // text "2T" would sort before "176G", by size it comes after.
    disksPanel.querySelectorAll('thead th')[freeIndex].dispatchEvent(new window.Event('click'));
    const order = Array.from(disksPanel.querySelectorAll('tbody tr')).map((tr) => tr.dataset.id);
    check('disks sort by real size, not by label', order.join(',') === '/,/home,/scratch', order.join(','));
  }

  // Second snapshot: rows must be updated in place, not duplicated.
  const jobsPanel = doc.querySelector("[data-section='jobs']");
  const before = jobsPanel ? jobsPanel.querySelectorAll('tbody tr').length : 0;
  send({ type: 'snapshot', snapshot });
  const after = jobsPanel ? jobsPanel.querySelectorAll('tbody tr').length : 0;
  check('rows are not duplicated on refresh', before === after, `${before} -> ${after}`);

  // Filtering to one user must shrink the table.
  if (jobsPanel) {
    const ownerSelect = jobsPanel.querySelector('select');
    ownerSelect.value = 'me';
    ownerSelect.dispatchEvent(new window.Event('change'));
    const mine = snapshot.jobs.filter((j) => j.user === snapshot.user).length;
    check('owner filter applies', jobsPanel.querySelectorAll('tbody tr').length === mine,
      `${jobsPanel.querySelectorAll('tbody tr').length} vs ${mine}`);
    ownerSelect.value = 'all';
    ownerSelect.dispatchEvent(new window.Event('change'));

    // Sorting by a header must not lose or duplicate rows.
    const th = jobsPanel.querySelectorAll('thead th')[0];
    th.dispatchEvent(new window.Event('click'));
    check('sort keeps row count', jobsPanel.querySelectorAll('tbody tr').length === snapshot.jobs.length);

    // Clicking a row selects exactly one; double-click asks the host for details.
    const firstRow = jobsPanel.querySelector('tbody tr');
    firstRow.dispatchEvent(new window.Event('click'));
    check('one row selected', jobsPanel.querySelectorAll('tbody tr.selected').length === 1);
    firstRow.dispatchEvent(new window.Event('dblclick'));
  }

  // Collapsing a sidebar panel must hide its body and survive as state.
  if (variant === 'sidebar' && nodesPanel) {
    const title = nodesPanel.querySelector('.panel-title');
    check('sidebar panel titles are collapsible', title.classList.contains('collapsible'));
    title.dispatchEvent(new window.Event('click'));
    check('clicking a title collapses the panel', nodesPanel.classList.contains('collapsed'));
    check('collapse is persisted', (window.eval('__state') || {}).collapsed.nodes === true);
    title.dispatchEvent(new window.Event('click'));
    check('clicking again expands it', !nodesPanel.classList.contains('collapsed'));
  }

  const messages = postedOf();
  check('posted ready to the host', messages.some((m) => m.type === 'ready'));
  if (jobsPanel) {
    check('posted openDetail on double click', messages.some((m) => m.type === 'openDetail' && m.kind === 'job'));
    // In window mode the host opens a tab, so the view must not also cover
    // itself with a loading overlay.
    check(
      detailsIn === 'overlay'
        ? 'a loading overlay is opened in overlay mode'
        : `no in-view overlay is opened in ${detailsIn} mode`,
      doc.getElementById('overlay').classList.contains('hidden') === (detailsIn !== 'overlay')
    );
  }

  // Detail payloads must render without throwing.
  const job = snapshot.jobs[0];
  send({ type: 'detail', detail: { kind: 'job', job_id: job.job_id, job, detail: { JobId: job.job_id, Partition: job.partition }, usage: { MaxRSS: '1G' } } });
  check('job detail overlay opens', !doc.getElementById('overlay').classList.contains('hidden'));
  check('job detail shows scontrol fields', doc.getElementById('overlay').textContent.includes('JobId'));

  send({ type: 'detail', detail: { kind: 'node', node: snapshot.nodes[0].name, detail: { NodeName: snapshot.nodes[0].name }, jobs: snapshot.jobs.slice(0, 3) } });
  check('node detail overlay opens', doc.getElementById('overlay').textContent.includes('NodeName'));
  const jobLink = doc.getElementById('overlay').querySelector('td.link');
  check('node detail links through to a job', !!jobLink);
  if (jobLink) {
    jobLink.dispatchEvent(new window.Event('click'));
    check('clicking a linked job asks the host for it',
      postedOf().filter((m) => m.type === 'openDetail' && m.kind === 'job').length > 0);
  }

  send({ type: 'status', state: 'error', message: 'collector died' });
  check('error status is shown', doc.getElementById('status').textContent.includes('collector died'));

  // An empty cluster must not throw or leave stale rows behind.
  send({ type: 'snapshot', snapshot: { ...snapshot, jobs: [], nodes: [], disks: [], gpu: { total: 0, free_est: 0, per_type_stats: {} },
    summary: { all: {}, me: {}, others: {} } } });
  if (jobsPanel) {
    check('empty snapshot clears rows', jobsPanel.querySelectorAll('tbody tr').length === 0);
    check('empty placeholder is shown', !jobsPanel.querySelector('.empty').classList.contains('hidden'));
  }
}

function runDetailWindow() {
  console.log('\ndetail window:');
  const { window, send, posted: postedOf } = makeWindow('detail');
  const doc = window.document;

  send({ type: 'config', sections: ['summary', 'jobs', 'nodes', 'gpus', 'disks'], ownerFilter: 'all', interval: 3, detailsIn: 'window' });
  check('no tables are built in a detail tab', doc.querySelectorAll('#root .panel-body table').length === 0);
  check('shows a loading message until the payload arrives', doc.getElementById('status').textContent.length > 0);

  send({ type: 'detailTarget', kind: 'job', id: '1001' });
  const job = snapshot.jobs[0];
  send({ type: 'detail', detail: { kind: 'job', job_id: job.job_id, job, detail: { JobId: job.job_id, WorkDir: '/home/alice' }, usage: { MaxRSS: '1G' } } });

  check('renders into the page, not the overlay', doc.getElementById('overlay').classList.contains('hidden'));
  check('page is titled after the job', doc.querySelector('#root .panel-title').textContent === 'Job 1001',
    doc.querySelector('#root .panel-title').textContent);
  check('page shows scontrol fields', doc.getElementById('root').textContent.includes('WorkDir'));
  check('page shows the squeue and usage sections',
    doc.getElementById('root').textContent.includes('squeue') && doc.getElementById('root').textContent.includes('usage'));

  const buttons = Array.from(doc.querySelectorAll('#root button')).map((b) => b.textContent);
  check('page offers refresh and copy', buttons.includes('Refresh') && buttons.includes('Copy all'), buttons.join(','));
  doc.querySelectorAll('#root button')[0].dispatchEvent(new window.Event('click'));
  check('refresh asks the host to reload', postedOf().some((m) => m.type === 'refreshDetail'));

  // Switching the same tab to a node must replace the content, not append.
  send({ type: 'detailTarget', kind: 'node', id: 'gpu01' });
  send({ type: 'detail', detail: { kind: 'node', node: 'gpu01', detail: { NodeName: 'gpu01' }, jobs: snapshot.jobs.slice(0, 2) } });
  check('page swaps to the node', doc.querySelector('#root .panel-title').textContent === 'Node gpu01');
  check('old job content is gone', !doc.getElementById('root').textContent.includes('WorkDir'));
  check('only one panel is on the page', doc.querySelectorAll('#root .panel').length === 1);

  // A snapshot arriving on a detail tab must be ignored, not crash it.
  send({ type: 'snapshot', snapshot });
  check('snapshots do not disturb a detail tab', doc.querySelector('#root .panel-title').textContent === 'Node gpu01');

  send({ type: 'detailError', message: 'scontrol timed out' });
  check('detail errors are surfaced', doc.getElementById('status').textContent.includes('scontrol timed out'));
}

function runModal() {
  console.log('\nmodal:');
  const { window, send, posted: postedOf } = makeWindow('modal');
  const doc = window.document;

  send({ type: 'config', sections: ['summary', 'jobs'], ownerFilter: 'all', interval: 3, detailsIn: 'modal' });
  const job = snapshot.jobs[0];
  send({ type: 'detail', detail: { kind: 'job', job_id: job.job_id, job, detail: { JobId: job.job_id, WorkDir: '/home/alice' }, usage: {} } });

  const scrim = doc.querySelector('#root .scrim');
  check('renders a dimmed backdrop', !!scrim);
  check('the card sits on the backdrop', !!scrim && scrim.querySelector('.panel'));
  check('card is titled after the job', doc.querySelector('#root .panel-title').textContent === 'Job 1001');
  check('card shows scontrol fields', doc.getElementById('root').textContent.includes('WorkDir'));

  const buttons = Array.from(doc.querySelectorAll('#root button')).map((b) => b.textContent);
  check('modal offers a Close button', buttons.includes('Close'), buttons.join(','));

  // Escape must ask the host to dispose the panel, not just blank the content.
  doc.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' }));
  check('Escape asks the host to close', postedOf().some((m) => m.type === 'closeDetail'));

  // Clicking the card itself must not dismiss; clicking the backdrop must.
  const before = postedOf().filter((m) => m.type === 'closeDetail').length;
  const card = scrim.querySelector('.panel');
  const inside = new window.MouseEvent('click', { bubbles: true });
  card.dispatchEvent(inside);
  check('clicking inside the card keeps it open',
    postedOf().filter((m) => m.type === 'closeDetail').length === before);
  scrim.dispatchEvent(new window.MouseEvent('click', { bubbles: false }));
  check('clicking the backdrop closes it',
    postedOf().filter((m) => m.type === 'closeDetail').length > before);

  doc.querySelectorAll('#root button').forEach((b) => {
    if (b.textContent === 'Close') b.dispatchEvent(new window.Event('click'));
  });
  check('the Close button closes it',
    postedOf().filter((m) => m.type === 'closeDetail').length > before + 1);
}

run('dashboard', ['summary', 'jobs', 'nodes', 'gpus', 'disks'], 'overlay');
run('sidebar', ['summary', 'jobs', 'nodes', 'gpus', 'disks'], 'popup');
runDetailWindow();
runModal();

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
