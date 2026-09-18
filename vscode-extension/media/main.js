// @ts-check
/**
 * Webview front end shared by the sidebar view and the dashboard panel.
 *
 * The extension host streams snapshots in; this script owns filtering, sorting,
 * selection and rendering. Both variants run the same code -- `variant` only
 * picks which sections appear and which columns survive in a narrow view.
 *
 * Panels are built once and then updated in place. A full re-render on every
 * snapshot would reset each table's scroll offset and, worse, destroy the
 * focused search box mid-keystroke -- on a 3-second refresh that makes the view
 * unusable. Only table bodies and counters are rewritten per tick.
 */
(function () {
  const vscode = acquireVsCodeApi();
  const variant = ['dashboard', 'detail', 'modal'].indexOf(document.body.dataset.variant) >= 0
    ? document.body.dataset.variant
    : 'sidebar';
  const compact = variant === 'sidebar';
  /** True in the two variants that show one job or node instead of the tables. */
  const isDetailView = variant === 'detail' || variant === 'modal';

  const statusEl = /** @type {HTMLElement} */ (document.getElementById('status'));
  const rootEl = /** @type {HTMLElement} */ (document.getElementById('root'));
  const overlayEl = /** @type {HTMLElement} */ (document.getElementById('overlay'));

  /** Restored on reload so a hidden tab comes back where it was. */
  const saved = vscode.getState() || {};

  const state = {
    snapshot: /** @type {any} */ (null),
    sections: ['summary', 'jobs', 'nodes', 'gpus', 'disks'],
    detailsIn: 'window',
    /** Which sidebar panels the user has folded away. */
    collapsed: saved.collapsed || {},
    ownerFilter: saved.ownerFilter || 'me',
    stateFilter: saved.stateFilter || 'all',
    search: saved.search || '',
    sort: Object.assign(
      { jobs: { key: 'default', desc: false }, nodes: { key: 'name', desc: false }, gpus: { key: 'type', desc: false }, disks: { key: 'usage', desc: true } },
      saved.sort || {}
    ),
    selected: Object.assign({ jobs: null, nodes: null, gpus: null, disks: null }, saved.selected || {}),
    /** Rows currently on screen per section, so keyboard nav agrees with the DOM. */
    visible: { jobs: [], nodes: [], gpus: [], disks: [] },
    /**
     * Job ids pinned to the top. Owned by the collector's config file, not by
     * this view, so a job pinned in the terminal UI is pinned here too; the
     * local copy is updated optimistically so a click feels instant instead of
     * waiting for the next snapshot.
     */
    pinned: /** @type {string[]} */ (saved.pinned || []),
    /** Node whose CPU is being read right now, if any. */
    probing: /** @type {string|null} */ (null),
  };

  function persist() {
    vscode.setState({
      ownerFilter: state.ownerFilter,
      stateFilter: state.stateFilter,
      search: state.search,
      sort: state.sort,
      selected: state.selected,
      collapsed: state.collapsed,
      pinned: state.pinned,
    });
  }

  // ------------------------------------------------------------- formatting

  function formatMb(mb) {
    const value = Number(mb) || 0;
    if (value < 1024) return value + 'M';
    const gb = value / 1024;
    if (gb < 1024) return gb.toFixed(1) + 'G';
    return (gb / 1024).toFixed(2) + 'T';
  }

  function stateClass(value) {
    const s = String(value || '').toUpperCase();
    if (s.startsWith('R')) return 'state-running';
    if (s.startsWith('P')) return 'state-pending';
    if (s.startsWith('F') || s.startsWith('CA') || s.startsWith('TO') || s.startsWith('NF')) return 'state-failed';
    return 'state-other';
  }

  /** Short state tag: RUNNING -> R, PENDING -> PD, COMPLETING -> CG. */
  function shortState(value) {
    const s = String(value || '').toUpperCase();
    if (s.startsWith('RUNNING')) return 'R';
    if (s.startsWith('PENDING')) return 'PD';
    if (s.startsWith('COMPLETING')) return 'CG';
    if (s.startsWith('COMPLETED')) return 'CD';
    return s.slice(0, 4);
  }

  function compareValues(a, b) {
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
  }

  function jobIdKey(jobId) {
    const match = /^(\d+)/.exec(String(jobId));
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
  }

  function jobStateRank(value) {
    const s = String(value || '').toUpperCase();
    if (s.startsWith('R')) return 0;
    if (s.startsWith('CG')) return 1;
    if (s.startsWith('P')) return 2;
    return 3;
  }

  function el(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value === undefined || value === null) continue;
        if (key === 'class') node.className = String(value);
        else if (key === 'text') node.textContent = String(value);
        else if (key === 'dataset') Object.assign(node.dataset, value);
        else if (key.startsWith('on')) node.addEventListener(key.slice(2), /** @type {any} */ (value));
        else node.setAttribute(key, String(value));
      }
    }
    for (const child of children || []) {
      if (child) node.appendChild(child);
    }
    return node;
  }

  function show(value) {
    return value === undefined || value === null || value === '' ? '-' : String(value);
  }

  function isPinned(jobId) {
    return state.pinned.indexOf(String(jobId)) >= 0;
  }

  /**
   * Flip a pin now and tell the host to persist it.
   *
   * Applied locally first: the round trip spawns a collector process, and a
   * star that takes half a second to light up feels broken.
   */
  function togglePin(jobId) {
    const id = String(jobId);
    state.pinned = isPinned(id) ? state.pinned.filter((x) => x !== id) : state.pinned.concat([id]);
    persist();
    updateSection('jobs');
    vscode.postMessage({ type: 'togglePin', jobId: id });
  }

  // ------------------------------------------------------------ column sets

  /**
   * Column definitions per section. `compactHide: true` drops a column in the
   * sidebar, where there is room for roughly four of them.
   */
  const JOB_COLUMNS = [
    {
      key: 'pin',
      label: '',
      title: 'Pin this job to the top of the list',
      // Clicking the star must not also open the job, so the cell swallows the
      // row's double-click as well as its own click.
      value: (j) => (isPinned(j.job_id) ? '\u2605' : '\u2606'),
      cls: (j) => (isPinned(j.job_id) ? 'pin pinned' : 'pin'),
      sortable: false,
      onclick: (j) => togglePin(j.job_id),
    },
    { key: 'job_id', label: 'JOBID', numeric: true, value: (j) => j.job_id, sort: (j) => jobIdKey(j.job_id) },
    { key: 'user', label: 'USER', value: (j) => j.user, compactHide: true },
    { key: 'state', label: 'ST', value: (j) => shortState(j.state), cls: (j) => stateClass(j.state), sort: (j) => jobStateRank(j.state) },
    { key: 'partition', label: 'PART', value: (j) => j.partition, compactHide: true },
    { key: 'name', label: 'NAME', value: (j) => j.name },
    { key: 'nodes', label: 'N', numeric: true, value: (j) => j.nodes, compactHide: true },
    { key: 'cpus', label: 'CPUS', numeric: true, value: (j) => j.cpu_count, compactHide: true },
    { key: 'gpus', label: 'GPUS', numeric: true, value: (j) => j.gpu_count },
    { key: 'mem', label: 'MEM', numeric: true, value: (j) => j.mem, sort: (j) => j.mem_mb, compactHide: true },
    { key: 'time', label: 'TIME', numeric: true, value: (j) => j.time_used },
    { key: 'node_list', label: 'NODELIST', value: (j) => j.node_list, compactHide: true },
  ];

  const NODE_COLUMNS = [
    { key: 'name', label: 'NODE', value: (n) => n.name },
    {
      key: 'state',
      label: 'STATE',
      // A drain reason is the one thing you want without opening the node, so
      // it rides along in the state cell's tooltip and text.
      value: (n) => (n.reason ? `${n.state} (${n.reason})` : n.state),
      sort: (n) => n.state,
      cls: (n) => (/drain|down|fail|err/i.test(String(n.state)) ? 'state-failed' : /idle/i.test(String(n.state)) ? 'state-running' : 'state-other'),
      compactHide: true,
    },
    { key: 'partition', label: 'PART', value: (n) => n.partition, compactHide: true },
    {
      key: 'cpus',
      label: 'CPU A/T',
      numeric: true,
      value: (n) => `${n.cpus_alloc_n}/${n.cpus_total_n}`,
      sort: (n) => n.cpus_alloc_n,
      bar: (n) => (n.cpus_total_n ? n.cpus_alloc_n / n.cpus_total_n : 0),
    },
    { key: 'cpus_idle', label: 'IDLE', numeric: true, value: (n) => n.cpus_idle_n, compactHide: true },
    {
      key: 'load',
      label: 'LOAD',
      numeric: true,
      value: (n) => (n.cpu_load_n || 0).toFixed(2),
      sort: (n) => n.cpu_load_ratio,
      bar: (n) => n.cpu_load_ratio,
      compactHide: true,
    },
    { key: 'mem_total', label: 'MEM', numeric: true, value: (n) => n.mem_total_human, sort: (n) => n.mem_total_mb, compactHide: true },
    {
      key: 'mem_free',
      label: 'MEM FREE',
      numeric: true,
      value: (n) => n.mem_free_human,
      sort: (n) => n.mem_free_mb,
      bar: (n) => (n.mem_total_mb ? 1 - n.mem_free_mb / n.mem_total_mb : 0),
    },
    {
      key: 'gpus',
      label: 'GPU U/T',
      numeric: true,
      value: (n) => (n.gpu_total ? `${n.gpu_used}/${n.gpu_total}` : '-'),
      sort: (n) => n.gpu_free,
      cls: (n) => (n.gpu_total && n.gpu_free > 0 ? 'state-running' : 'state-other'),
    },
    { key: 'gres', label: 'GRES', value: (n) => (n.gpu_types || []).join(', ') || n.gres, compactHide: true },
    {
      key: 'cpu',
      label: 'CPU',
      // Slurm reports the layout but never the model, so an unprobed node
      // shows what it does know: 2 x 64C/2T. Open the node to read the rest.
      value: (n) => (n.cpu && n.cpu.known ? n.cpu.summary : (n.cpu || {}).topology || '-'),
      sort: (n) => (n.cpu && n.cpu.known ? n.cpu.summary : ''),
      cls: (n) => (n.cpu && n.cpu.known ? '' : 'dim'),
      title: (n) =>
        n.cpu && n.cpu.known
          ? `${n.cpu.model} (read over ${n.cpu.source})`
          : 'Open this node to read its CPU model',
      compactHide: true,
    },
  ];

  const DISK_COLUMNS = [
    {
      key: 'usage',
      label: 'USE%',
      numeric: true,
      value: (d) => d.usage_percent,
      sort: (d) => d.usage_pct,
      bar: (d) => (d.usage_pct || 0) / 100,
    },
    { key: 'mount', label: 'MOUNT', value: (d) => d.mount },
    { key: 'size', label: 'SIZE', numeric: true, value: (d) => d.size, sort: (d) => d.size_mb, compactHide: true },
    { key: 'used', label: 'USED', numeric: true, value: (d) => d.used, sort: (d) => d.used_mb, compactHide: true },
    // Free space is what you actually check before submitting a job, so it
    // survives into the narrow view where SIZE and USED do not.
    { key: 'avail', label: 'FREE', numeric: true, value: (d) => d.avail, sort: (d) => d.avail_mb },
    { key: 'type', label: 'TYPE', value: (d) => d.fs_type, compactHide: true },
  ];

  const GPU_COLUMNS = [
    { key: 'type', label: 'GPU TYPE', value: (g) => g.type },
    { key: 'total', label: 'TOTAL', numeric: true, value: (g) => g.total },
    {
      key: 'active',
      label: 'ACTIVE',
      numeric: true,
      value: (g) => g.active,
      bar: (g) => (g.total ? g.active / g.total : 0),
    },
    { key: 'reserved', label: 'RESRV', numeric: true, value: (g) => g.reserved, compactHide: true },
    { key: 'free', label: 'FREE', numeric: true, value: (g) => g.free_est, cls: (g) => (g.free_est > 0 ? 'state-running' : 'state-other') },
  ];

  const SECTION_SPEC = {
    jobs: { title: 'Jobs', columns: JOB_COLUMNS, id: (j) => j.job_id },
    nodes: { title: 'Nodes', columns: NODE_COLUMNS, id: (n) => n.name },
    gpus: { title: 'GPUs', columns: GPU_COLUMNS, id: (g) => g.type },
    disks: { title: 'Disks', columns: DISK_COLUMNS, id: (d) => d.mount },
  };

  function columnsFor(all) {
    return compact ? all.filter((c) => !c.compactHide) : all;
  }

  // ---------------------------------------------------------------- filters

  function matchesOwner(job) {
    const me = state.snapshot ? state.snapshot.user : '';
    if (state.ownerFilter === 'me') return job.user === me;
    if (state.ownerFilter === 'others') return job.user !== me;
    return true;
  }

  function matchesState(job) {
    const s = String(job.state || '').toUpperCase();
    if (state.stateFilter === 'running') return s.startsWith('R');
    if (state.stateFilter === 'pending') return s.startsWith('P');
    return true;
  }

  function matchesSearch(job) {
    const needle = state.search.trim().toLowerCase();
    if (!needle) return true;
    return [job.job_id, job.user, job.name, job.partition, job.state, job.node_list]
      .join(' ')
      .toLowerCase()
      .includes(needle);
  }

  /**
   * Move pinned jobs to the front, keeping the order within both groups.
   *
   * Done after sorting rather than inside the comparator, so reversing the
   * sort direction does not flip the pinned rows to the bottom.
   */
  function pinnedFirst(rows) {
    if (!state.pinned.length) return rows;
    return rows.filter((j) => isPinned(j.job_id)).concat(rows.filter((j) => !isPinned(j.job_id)));
  }

  function sortRows(section, rows, columns) {
    const setting = state.sort[section] || { key: 'default', desc: false };
    if (section === 'jobs' && setting.key === 'default') {
      // The TUI's natural order: running first, then by job id.
      return pinnedFirst(rows.slice().sort((a, b) => jobStateRank(a.state) - jobStateRank(b.state) || jobIdKey(a.job_id) - jobIdKey(b.job_id)));
    }
    const column = columns.find((c) => c.key === setting.key);
    if (!column) return section === 'jobs' ? pinnedFirst(rows.slice()) : rows.slice();
    const keyOf = column.sort || column.value;
    const sorted = rows.slice().sort((a, b) => compareValues(keyOf(a), keyOf(b)));
    const ordered = setting.desc ? sorted.reverse() : sorted;
    return section === 'jobs' ? pinnedFirst(ordered) : ordered;
  }

  function rowsFor(section) {
    if (!state.snapshot) return [];
    if (section === 'jobs') {
      return (state.snapshot.jobs || []).filter((j) => matchesOwner(j) && matchesState(j) && matchesSearch(j));
    }
    if (section === 'nodes') return state.snapshot.nodes || [];
    if (section === 'disks') return state.snapshot.disks || [];
    if (section === 'gpus') {
      const stats = (state.snapshot.gpu || {}).per_type_stats || {};
      return Object.keys(stats).map((type) => Object.assign({ type }, stats[type]));
    }
    return [];
  }

  // ------------------------------------------------------- panel scaffolding

  /** Built-once DOM handles, keyed by section name. */
  const panels = {};

  function bar() {
    return el('span', { class: 'bar' }, [el('span', {})]);
  }

  function setBar(node, fraction) {
    const pct = Math.max(0, Math.min(1, Number(fraction) || 0));
    node.className = pct >= 0.9 ? 'bar high' : pct >= 0.7 ? 'bar warn' : 'bar';
    /** @type {HTMLElement} */ (node.firstChild).style.width = (pct * 100).toFixed(1) + '%';
  }

  function makePanel(section, title, controls, body) {
    const count = el('span', { class: 'panel-count' });
    const titleEl = el('span', { class: 'panel-title', text: title });
    const head = el('div', { class: 'panel-head' }, [titleEl, count, el('span', { class: 'spacer' }), ...(controls || [])]);
    const panel = el('section', { class: 'panel', dataset: { section } }, [head, body]);

    // Five panels stacked in a 300px sidebar only works if you can fold the
    // ones you are not watching. The dashboard has the room, so it stays fixed.
    if (compact) {
      titleEl.classList.add('collapsible');
      titleEl.title = 'Click to collapse or expand';
      const apply = () => panel.classList.toggle('collapsed', !!state.collapsed[section]);
      titleEl.addEventListener('click', () => {
        state.collapsed[section] = !state.collapsed[section];
        persist();
        apply();
      });
      apply();
    }
    return { panel, count, titleEl };
  }

  function makeTableSection(section) {
    const spec = SECTION_SPEC[section];
    const columns = columnsFor(spec.columns);

    const headRow = el('tr');
    const headers = columns.map((column) => {
      const sortable = column.sortable !== false;
      const th = el('th', {
        class: [column.numeric ? 'numeric' : '', sortable ? '' : 'unsortable'].filter(Boolean).join(' '),
        text: column.label,
        title: sortable ? `Sort by ${column.label}` : column.title || '',
        onclick: sortable
          ? () => {
              const current = state.sort[section] || {};
              state.sort[section] = current.key === column.key ? { key: column.key, desc: !current.desc } : { key: column.key, desc: false };
              persist();
              updateSection(section);
            }
          : null,
      });
      headRow.appendChild(th);
      return { column, th };
    });

    const tbody = el('tbody');
    const table = el('table', { tabindex: '0' }, [el('thead', {}, [headRow]), tbody]);
    table.addEventListener('keydown', (event) => onTableKey(section, /** @type {KeyboardEvent} */ (event)));

    const empty = el('div', { class: 'empty hidden', text: 'Nothing to show' });
    const body = el('div', { class: 'panel-body' }, [table, empty]);

    const controls = section === 'jobs' ? jobControls() : [];
    const { panel, count } = makePanel(section, spec.title, controls, body);
    panels[section] = { panel, count, tbody, headers, columns, table, empty, spec, rowNodes: new Map(), rowData: new Map() };
    return panel;
  }

  function jobControls() {
    const ownerSelect = el('select', {
      title: 'Whose jobs to show',
      onchange: (e) => {
        state.ownerFilter = e.target.value;
        persist();
        updateSection('jobs');
      },
    });
    for (const [value, label] of [['all', 'all users'], ['me', 'me'], ['others', 'others']]) {
      ownerSelect.appendChild(el('option', { value, text: label }));
    }
    ownerSelect.value = state.ownerFilter;

    const stateSelect = el('select', {
      title: 'Job state',
      onchange: (e) => {
        state.stateFilter = e.target.value;
        persist();
        updateSection('jobs');
      },
    });
    for (const [value, label] of [['all', 'any state'], ['running', 'running'], ['pending', 'pending']]) {
      stateSelect.appendChild(el('option', { value, text: label }));
    }
    stateSelect.value = state.stateFilter;

    const search = el('input', {
      type: 'search',
      placeholder: 'search',
      title: 'Filter by id, user, name, partition or node',
      oninput: (e) => {
        state.search = e.target.value;
        persist();
        updateSection('jobs');
      },
    });
    /** @type {HTMLInputElement} */ (search).value = state.search;

    return [ownerSelect, stateSelect, search];
  }

  function makeSummarySection() {
    const grid = el('div', { class: 'summary-grid' });
    grid.appendChild(el('span', { class: 'head' }));
    for (const head of ['jobs', 'gpus', 'cpus', 'mem']) {
      grid.appendChild(el('span', { class: 'head value', text: head }));
    }
    const cells = {};
    for (const bucket of ['all', 'me', 'others']) {
      for (const phase of ['running', 'pending']) {
        grid.appendChild(el('span', { class: 'label', text: `${bucket} ${phase}` }));
        cells[`${bucket}.${phase}`] = ['jobs', 'gpus', 'cpus', 'mem'].map(() => {
          const node = el('span', { class: 'value', text: '-' });
          grid.appendChild(node);
          return node;
        });
      }
    }
    const { panel, count, titleEl } = makePanel('summary', 'Cluster', [], grid);
    panels.summary = { panel, count, cells, title: titleEl };
    return panel;
  }

  /** Rebuild the panel scaffolding; only on first paint or a section change. */
  function buildLayout() {
    rootEl.className = variant;
    rootEl.textContent = '';
    for (const key of Object.keys(panels)) delete panels[key];
    for (const section of state.sections) {
      if (section === 'summary') rootEl.appendChild(makeSummarySection());
      else if (SECTION_SPEC[section]) rootEl.appendChild(makeTableSection(section));
    }
    updateAll();
  }

  // ----------------------------------------------------------- in-place update

  function updateSection(section) {
    const panel = panels[section];
    if (!panel || !state.snapshot) return;
    if (section === 'summary') {
      updateSummary();
      return;
    }

    const rows = sortRows(section, rowsFor(section), panel.columns);
    state.visible[section] = rows;

    const setting = state.sort[section] || {};
    for (const { column, th } of panel.headers) {
      if (setting.key === column.key) th.setAttribute('aria-sort', setting.desc ? 'descending' : 'ascending');
      else th.removeAttribute('aria-sort');
    }

    const me = state.snapshot.user;
    const seen = new Set();
    let previous = null;
    for (const row of rows) {
      const id = String(panel.spec.id(row));
      seen.add(id);
      panel.rowData.set(id, row);
      let node = panel.rowNodes.get(id);
      if (!node) {
        node = makeRow(section, panel, id);
        panel.rowNodes.set(id, node);
      }
      fillRow(node, panel, row, section === 'jobs' && row.user === me, state.selected[section] === id);
      // Re-inserting a node already in the right place is a no-op in the DOM,
      // so a steady cluster produces no layout churn between ticks.
      if (previous ? previous.nextSibling !== node.tr : panel.tbody.firstChild !== node.tr) {
        panel.tbody.insertBefore(node.tr, previous ? previous.nextSibling : panel.tbody.firstChild);
      }
      previous = node.tr;
    }
    for (const [id, node] of panel.rowNodes) {
      if (!seen.has(id)) {
        node.tr.remove();
        panel.rowNodes.delete(id);
        panel.rowData.delete(id);
      }
    }

    panel.empty.classList.toggle('hidden', rows.length > 0);
    panel.table.classList.toggle('hidden', rows.length === 0);

    if (section === 'jobs') {
      panel.count.textContent = `${rows.length}/${(state.snapshot.jobs || []).length}`;
    } else if (section === 'gpus') {
      const gpu = state.snapshot.gpu || {};
      panel.count.textContent = `${gpu.free_est || 0} free / ${gpu.total || 0}`;
    } else {
      panel.count.textContent = String(rows.length);
    }
  }

  function makeRow(section, panel, id) {
    const tr = el('tr', {
      dataset: { id },
      onclick: () => select(section, id),
      ondblclick: () => openDetail(section, id),
    });
    const cells = panel.columns.map((column) => {
      const td = el('td', { class: column.numeric ? 'numeric' : '' });
      if (column.onclick) {
        td.addEventListener('click', (event) => {
          // Without this the click would also select the row, and a
          // double-click on the star would open the job behind it.
          event.stopPropagation();
          const row = panel.rowData.get(id);
          if (row) column.onclick(row);
        });
        td.addEventListener('dblclick', (event) => event.stopPropagation());
      }
      let barNode = null;
      if (column.bar) {
        td.classList.add('with-bar');
        barNode = bar();
        td.appendChild(barNode);
      }
      const text = document.createTextNode('');
      td.appendChild(text);
      tr.appendChild(td);
      return { td, text, barNode, column };
    });
    return { tr, cells };
  }

  function fillRow(node, panel, row, mine, selected) {
    for (const cell of node.cells) {
      const value = show(cell.column.value(row));
      const tooltip = typeof cell.column.title === 'function' ? cell.column.title(row) : cell.column.title;
      if (cell.text.nodeValue !== value) {
        cell.text.nodeValue = value;
      }
      const wanted = tooltip || value;
      if (cell.td.title !== wanted) cell.td.title = wanted;
      if (cell.barNode) setBar(cell.barNode, cell.column.bar(row));
      const extra = cell.column.cls ? cell.column.cls(row) : '';
      const base = [cell.column.numeric ? 'numeric' : '', cell.barNode ? 'with-bar' : '', extra].filter(Boolean).join(' ');
      if (cell.td.className !== base) cell.td.className = base;
    }
    const cls = [selected ? 'selected' : '', mine ? 'mine' : ''].filter(Boolean).join(' ');
    if (node.tr.className !== cls) node.tr.className = cls;
  }

  function updateSummary() {
    const panel = panels.summary;
    if (!panel || !state.snapshot) return;
    const summary = state.snapshot.summary || {};
    for (const bucket of ['all', 'me', 'others']) {
      for (const phase of ['running', 'pending']) {
        const cell = (summary[bucket] || {})[phase] || { jobs: 0, gpus: 0, cpus: 0, mem_mb: 0 };
        const nodes = panel.cells[`${bucket}.${phase}`];
        const values = [cell.jobs, cell.gpus, cell.cpus, formatMb(cell.mem_mb)];
        nodes.forEach((node, index) => {
          const text = String(values[index]);
          if (node.textContent !== text) node.textContent = text;
        });
      }
    }
    panel.title.textContent = state.snapshot.host || 'Cluster';
    panel.count.textContent = state.snapshot.timestamp
      ? new Date(state.snapshot.timestamp * 1000).toLocaleTimeString()
      : '';
  }

  function updateAll() {
    for (const section of state.sections) updateSection(section);
  }

  // -------------------------------------------------------------- selection

  function select(section, id) {
    const previous = state.selected[section];
    state.selected[section] = String(id);
    persist();
    const panel = panels[section];
    if (!panel || !panel.rowNodes) return;
    const before = panel.rowNodes.get(String(previous));
    if (before) before.tr.classList.remove('selected');
    const now = panel.rowNodes.get(String(id));
    if (now) now.tr.classList.add('selected');
  }

  function onTableKey(section, event) {
    const rows = state.visible[section] || [];
    if (!rows.length) return;
    const panel = panels[section];
    const idOf = (row) => String(panel.spec.id(row));
    const index = rows.findIndex((row) => idOf(row) === String(state.selected[section]));

    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const next = event.key === 'ArrowDown' ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
      const id = idOf(rows[next < 0 ? 0 : next]);
      select(section, id);
      const node = panel.rowNodes.get(id);
      if (node) node.tr.scrollIntoView({ block: 'nearest' });
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (index >= 0) openDetail(section, idOf(rows[index]));
    } else if (event.key.toLowerCase() === 'c' && index >= 0) {
      event.preventDefault();
      copyRow(section, rows[index]);
    } else if (event.key.toLowerCase() === 'p' && index >= 0 && section === 'jobs') {
      event.preventDefault();
      togglePin(rows[index].job_id);
    }
  }

  function copyRow(section, row) {
    const text = Object.keys(row)
      .filter((key) => typeof row[key] !== 'object')
      .map((key) => `${key}: ${row[key]}`)
      .join('\n');
    vscode.postMessage({ type: 'copy', text, label: `${section} row` });
  }

  // ---------------------------------------------------------------- overlay

  function openDetail(section, id) {
    if (section === 'jobs') {
      requestDetail('job', id);
    } else if (section === 'nodes') {
      requestDetail('node', id);
    } else if (section === 'gpus') {
      showGpuJobs(String(id));
    } else if (section === 'disks') {
      const disk = (state.snapshot.disks || []).find((d) => d.mount === id);
      if (disk) showOverlay(`Disk ${id}`, [kvList(disk)], JSON.stringify(disk, null, 2));
    }
  }

  function kvList(record) {
    const list = el('dl', { class: 'kv' });
    for (const key of Object.keys(record)) {
      const value = record[key];
      if (value === undefined || value === null || typeof value === 'object') continue;
      list.appendChild(el('dt', { text: key }));
      list.appendChild(el('dd', { text: String(value) }));
    }
    return list;
  }

  function showOverlay(title, children, copyText) {
    overlayEl.textContent = '';
    const head = el('div', { class: 'overlay-head' }, [
      el('span', { text: title }),
      el('span', { class: 'spacer' }),
      copyText ? el('button', { text: 'Copy all', onclick: () => vscode.postMessage({ type: 'copy', text: copyText, label: title }) }) : null,
      el('button', { text: 'Close', onclick: closeOverlay }),
    ]);
    const body = el('div', { class: 'overlay-body' }, children);
    overlayEl.appendChild(el('div', { class: 'overlay-card' }, [head, body]));
    overlayEl.classList.remove('hidden');
  }

  function closeOverlay() {
    overlayEl.classList.add('hidden');
    overlayEl.textContent = '';
  }

  overlayEl.addEventListener('click', (event) => {
    if (event.target === overlayEl) closeOverlay();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    if (variant === 'modal') closeDetailView();
    else closeOverlay();
  });

  function jobTable(jobs, headers, cellsOf) {
    return el('table', {}, [
      el('thead', {}, [el('tr', {}, headers.map((h) => el('th', { text: h })))]),
      el('tbody', {}, jobs.map((job) => el('tr', {}, cellsOf(job)))),
    ]);
  }

  function showGpuJobs(type) {
    const jobs = (state.snapshot.jobs || []).filter((j) => Object.keys(j.gpu_types || {}).indexOf(type) >= 0);
    const children = [el('h3', { text: `Jobs holding ${type}` })];
    if (!jobs.length) {
      children.push(el('p', { class: 'hint', text: 'No job currently holds this GPU type.' }));
    } else {
      children.push(
        jobTable(jobs, ['JOBID', 'USER', 'ST', 'GPUS', 'NAME', 'TIME'], (j) => [
          el('td', { class: 'link', text: j.job_id, title: 'Open this job', onclick: () => requestDetail('job', j.job_id) }),
          el('td', { text: j.user }),
          el('td', { text: shortState(j.state), class: stateClass(j.state) }),
          el('td', { class: 'numeric', text: (j.gpu_types || {})[type] || 0 }),
          el('td', { text: j.name }),
          el('td', { class: 'numeric', text: j.time_used }),
        ])
      );
    }
    showOverlay(`GPU ${type}`, children);
  }

  /**
   * Colour a metric bar by which end of its scale is the bad one.
   *
   * `setBar` treats a full bar as the dangerous one, which is right for a
   * memory limit and exactly wrong for CPU efficiency: a job using 5% of the
   * cores it reserved is the one wasting the machine.
   */
  function setMetricBar(node, percent, risk) {
    const pct = Math.max(0, Math.min(100, Number(percent) || 0));
    const bad = risk === 'low' ? pct < 25 : pct >= 90;
    const warn = risk === 'low' ? pct < 60 : pct >= 75;
    node.className = `bar${bad ? ' high' : warn ? ' warn' : ''}`;
    /** @type {HTMLElement} */ (node.firstChild).style.width = pct.toFixed(1) + '%';
  }

  function metricsTable(metrics) {
    const rows = [];
    for (const metric of metrics) {
      if (metric.kind === 'note') {
        rows.push(el('tr', {}, [el('td', { class: 'hint', colspan: '5', text: metric.note })]));
        continue;
      }
      const cells = [el('td', { class: 'metric-label', text: metric.label })];
      if (metric.kind === 'bar') {
        const gauge = bar();
        setMetricBar(gauge, metric.percent, metric.risk);
        cells.push(el('td', { class: 'with-bar' }, [gauge]));
        cells.push(el('td', { class: 'numeric', text: `${Number(metric.percent).toFixed(1)}%` }));
        cells.push(el('td', { class: 'numeric', text: `${metric.value} of ${metric.total}` }));
      } else {
        cells.push(el('td', { class: 'with-bar' }));
        cells.push(el('td', { class: 'numeric' }));
        cells.push(el('td', { class: 'numeric', text: metric.value }));
      }
      cells.push(el('td', { class: 'dim', text: metric.note }));
      rows.push(el('tr', {}, cells));
    }
    return el('table', { class: 'metrics' }, [el('tbody', {}, rows)]);
  }

  /**
   * The two files a job writes to.
   *
   * Slurm records the paths and nothing else, so the buttons go to the host:
   * it opens the real file when this machine can see it, and falls back to
   * fetching the tail through the collector when it cannot.
   */
  function outputChildren(payload) {
    const streams = payload.output || {};
    const jobId = String(payload.job_id);
    const rows = [];
    for (const name of ['stdout', 'stderr']) {
      const file = streams[name];
      if (!file) continue;
      if (name === 'stderr' && file.merged) {
        // One file, so one row: a second Open button for the same path would
        // only invite the question of how the two differ.
        rows.push(
          el('tr', {}, [
            el('td', { class: 'metric-label', text: 'stderr' }),
            el('td', { class: 'dim', colspan: '3', text: 'merged into stdout — the job asked for one file' }),
          ])
        );
        continue;
      }
      const where = file.exists
        ? `${formatBytes(file.size)} · written ${new Date(file.modified * 1000).toLocaleTimeString()}`
        : file.error;
      rows.push(
        el('tr', {}, [
          el('td', { class: 'metric-label', text: name }),
          el('td', { class: file.path ? '' : 'dim', text: file.path || '—' }),
          el('td', { class: 'dim', text: where }),
          el('td', {}, [
            file.exists
              ? el('button', {
                  text: 'Open',
                  title: 'Opens the file in an editor tab, or its tail when this machine cannot see it',
                  onclick: () =>
                    vscode.postMessage({
                      type: 'openOutput', jobId, stream: name, path: file.path, size: file.size,
                    }),
                })
              : null,
            file.path
              ? el('button', {
                  text: 'Copy path',
                  onclick: () => vscode.postMessage({ type: 'copy', text: file.path, label: `${name} path` }),
                })
              : null,
          ]),
        ])
      );
    }
    if (!rows.length) return [];
    return [el('h3', { text: 'output' }), el('table', { class: 'metrics outputs' }, [el('tbody', {}, rows)])];
  }

  function formatBytes(size) {
    const units = ['B', 'K', 'M', 'G', 'T'];
    let value = Number(size) || 0;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return unit === 0 ? `${value}B` : `${value.toFixed(1)}${units[unit]}`;
  }

  function detailJobChildren(payload) {
    const children = [];
    const jobId = String(payload.job_id);
    children.push(
      el('p', { class: 'hint' }, [
        el('button', {
          text: isPinned(jobId) ? '\u2605 Unpin this job' : '\u2606 Pin this job',
          title: 'Pinned jobs are listed first, in the editor and in the terminal UI',
          onclick: () => {
            togglePin(jobId);
            presentDetail(payload);
          },
        }),
      ])
    );
    if ((payload.metrics || []).length) {
      // Lead with the comparison: "16 CPUs" is in the scontrol block below,
      // but "one of those 16 is working" is only here.
      children.push(el('h3', { text: 'asked for vs used' }));
      children.push(metricsTable(payload.metrics));
    }
    children.push(...outputChildren(payload));
    if (payload.job) {
      children.push(el('h3', { text: 'squeue' }));
      children.push(kvList(payload.job));
    }
    if (payload.detail && Object.keys(payload.detail).length) {
      children.push(el('h3', { text: 'scontrol' }));
      children.push(kvList(payload.detail));
    }
    if (payload.usage && Object.keys(payload.usage).length) {
      children.push(el('h3', { text: 'usage' }));
      children.push(kvList(payload.usage));
    }
    if (!children.length) {
      children.push(el('p', { class: 'hint', text: 'Slurm returned nothing for this job; it may have finished.' }));
    }
    return children;
  }

  /**
   * The "what are these processors" block.
   *
   * Slurm answers "how many" on every refresh but never "which model" or "how
   * fast": that lives on the node itself. So the block always shows the layout
   * and offers to go and read the rest, once, on request.
   */
  function detailCpuChildren(payload) {
    const cpu = payload.cpu || {};
    const node = payload.node;
    const busy = state.probing === node;
    const children = [el('h3', { text: 'processors' })];

    const facts = {};
    const total = cpu.cpus_total || (cpu.sockets || 0) * (cpu.cores_per_socket || 0) * (cpu.threads_per_core || 0);
    // Lead with what the machine adds up to; the breakdown that produces it
    // follows, so "how many of what" is answered before the arithmetic.
    if (cpu.known && cpu.count_summary) facts.processors = cpu.count_summary;
    if (total) facts['logical CPUs'] = total;
    if (cpu.cores) facts['physical cores'] = cpu.cores;
    if (cpu.sockets) facts.sockets = cpu.sockets;
    if (cpu.cores_per_socket) facts['cores per socket'] = cpu.cores_per_socket;
    if (cpu.threads_per_core) facts['threads per core'] = cpu.threads_per_core;
    if (cpu.arch) facts.architecture = cpu.arch;
    if (cpu.known) {
      facts.model = cpu.model;
      // Each clock says what it is: a boost ceiling and an idling governor
      // both read as "the speed" when the label is dropped.
      for (const speed of cpu.speeds || []) {
        facts[speed.label === 'now' ? 'clock right now' : `${speed.label} clock`] = speed.value;
      }
      if (!(cpu.speeds || []).length && cpu.speed) facts.clock = cpu.speed;
      if (cpu.vendor) facts.vendor = cpu.vendor;
      facts['read over'] = cpu.source;
    }
    children.push(kvList(facts));

    if (!cpu.known) {
      children.push(
        el('p', { class: 'hint' }, [
          el('span', {
            text: busy
              ? 'Reading the CPU from the node\u2026 '
              : 'Slurm does not report the CPU model or its speed. ',
          }),
          busy
            ? null
            : el('button', {
                text: 'Read it from the node',
                title: 'Connects over ssh, or failing that runs a one-second job there',
                onclick: () => {
                  state.probing = node;
                  vscode.postMessage({ type: 'probeCpu', node });
                  presentDetail(payload);
                },
              }),
        ])
      );
    }
    if (cpu.error) {
      children.push(el('p', { class: 'hint error-text', text: `Could not read it: ${cpu.error}` }));
    }
    return children;
  }

  function detailNodeChildren(payload) {
    const children = detailCpuChildren(payload);
    if (payload.detail && Object.keys(payload.detail).length) {
      children.push(el('h3', { text: 'scontrol' }));
      children.push(kvList(payload.detail));
    }
    const jobs = payload.jobs || [];
    children.push(el('h3', { text: `Jobs on ${payload.node} (${jobs.length})` }));
    if (!jobs.length) {
      children.push(el('p', { class: 'hint', text: 'No jobs placed here.' }));
    } else {
      children.push(
        jobTable(jobs, ['JOBID', 'USER', 'ST', 'CPUS', 'GPUS', 'MEM', 'NAME', 'TIME'], (j) => [
          el('td', { class: 'link', text: j.job_id, title: 'Open this job', onclick: () => requestDetail('job', j.job_id) }),
          el('td', { text: j.user }),
          el('td', { text: shortState(j.state), class: stateClass(j.state) }),
          el('td', { class: 'numeric', text: j.cpu_count }),
          el('td', { class: 'numeric', text: j.gpu_count }),
          el('td', { class: 'numeric', text: j.mem }),
          el('td', { text: j.name }),
          el('td', { class: 'numeric', text: j.time_used }),
        ])
      );
    }
    return children;
  }

  /**
   * Show a detail payload where it belongs: filling this tab when we are the
   * detail window, otherwise as an overlay over the tables.
   */
  function presentDetail(payload) {
    const isJob = payload.kind === 'job';
    const title = isJob ? `Job ${payload.job_id}` : `Node ${payload.node}`;
    const children = isJob ? detailJobChildren(payload) : detailNodeChildren(payload);
    const copyText = JSON.stringify(payload, null, 2);
    if (isDetailView) {
      renderDetailPage(title, children, copyText);
    } else {
      showOverlay(title, children, copyText);
    }
  }

  function renderDetailPage(title, children, copyText) {
    rootEl.className = variant;
    rootEl.textContent = '';
    const head = el('div', { class: 'panel-head' }, [
      el('span', { class: 'panel-title', text: title }),
      el('span', { class: 'panel-count', text: new Date().toLocaleTimeString() }),
      el('span', { class: 'spacer' }),
      el('button', { text: 'Refresh', onclick: () => vscode.postMessage({ type: 'refreshDetail' }) }),
      el('button', { text: 'Copy all', onclick: () => vscode.postMessage({ type: 'copy', text: copyText, label: title }) }),
      variant === 'modal' ? el('button', { text: 'Close', title: 'Escape', onclick: closeDetailView }) : null,
    ]);
    const body = el('div', { class: 'panel-body overlay-body' }, children);
    const card = el('section', { class: 'panel', tabindex: '-1' }, [head, body]);

    if (variant === 'modal') {
      // A scrim the card floats on, so this reads as a dialog over the editor
      // rather than as one more tab. Clicking it dismisses, as a dialog should.
      const scrim = el('div', { class: 'scrim', onclick: (event) => {
        if (event.target === scrim) closeDetailView();
      } }, [card]);
      rootEl.appendChild(scrim);
    } else {
      rootEl.appendChild(card);
    }
    setStatus(null);
    // Escape only reaches us while the webview holds focus, and a fresh render
    // has just replaced whatever had it.
    if (variant === 'modal') card.focus();
  }

  /** Dismiss a modal: the host disposes the panel, so nothing is left behind. */
  function closeDetailView() {
    vscode.postMessage({ type: 'closeDetail' });
  }

  /** Ask the host for details; it decides between a new tab and an overlay. */
  function requestDetail(kind, id) {
    if (state.detailsIn === 'overlay' && !isDetailView) {
      showOverlay(`${kind === 'job' ? 'Job' : 'Node'} ${id}`, [el('p', { class: 'hint', text: 'Loading scontrol details…' })]);
    }
    vscode.postMessage({ type: 'openDetail', kind, id: String(id) });
  }

  // ----------------------------------------------------------------- status

  function setStatus(message, kind) {
    if (!message) {
      statusEl.classList.add('hidden');
      return;
    }
    statusEl.textContent = '';
    statusEl.className = `status${kind === 'error' ? ' error' : ''}`;
    statusEl.appendChild(el('span', { text: message }));
    if (kind === 'error') {
      statusEl.appendChild(el('button', { text: 'Show log', onclick: () => vscode.postMessage({ type: 'showLog' }) }));
      statusEl.appendChild(el('button', { text: 'Retry', onclick: () => vscode.postMessage({ type: 'refresh' }) }));
    }
  }

  // --------------------------------------------------------------- messages

  function sameSections(a, b) {
    return a.length === b.length && a.every((value, index) => value === b[index]);
  }

  window.addEventListener('message', (event) => {
    const message = event.data;
    switch (message.type) {
      case 'config': {
        if (!state.snapshot && !saved.ownerFilter) state.ownerFilter = message.ownerFilter;
        state.detailsIn = message.detailsIn || 'window';
        if (isDetailView) break;
        if (!sameSections(state.sections, message.sections) || !Object.keys(panels).length) {
          state.sections = message.sections;
          buildLayout();
        }
        break;
      }
      case 'snapshot':
        if (Array.isArray(message.snapshot.pinned)) state.pinned = message.snapshot.pinned;
        if (isDetailView) break;
        state.snapshot = message.snapshot;
        setStatus(null);
        if (!Object.keys(panels).length) buildLayout();
        else updateAll();
        break;
      case 'status':
        if (isDetailView) break;
        if (message.state === 'error') setStatus(message.message || 'Collector error', 'error');
        else if (message.state === 'starting' && !state.snapshot) setStatus('Starting the Slurm collector…');
        else if (message.state === 'paused' && !state.snapshot) setStatus('Paused — open a Slurm view to start polling.');
        else if (state.snapshot) setStatus(null);
        break;
      case 'detail':
        presentDetail(message.detail);
        break;
      case 'detailError':
        if (isDetailView) setStatus(message.message, 'error');
        else showOverlay('Details unavailable', [el('p', { class: 'hint', text: message.message })]);
        break;
      case 'detailTarget':
        setStatus(`Loading ${message.kind} ${message.id}…`);
        break;
      case 'pinned':
        // The host is the authority: a pin made in the terminal UI, or one of
        // ours that failed to persist, corrects the optimistic local list.
        state.pinned = message.pinned || [];
        persist();
        if (!isDetailView) updateSection('jobs');
        break;
      case 'cpuProbe':
        state.probing = message.state === 'started' ? message.node : null;
        if (message.state === 'started') {
          setStatus(`Reading ${message.node}'s CPU — ssh, then a one-second job…`);
        } else if (message.message) {
          setStatus(`Could not read ${message.node}'s CPU: ${message.message}`, 'error');
        } else {
          setStatus(null);
        }
        break;
    }
  });

  if (isDetailView) setStatus('Loading details…');
  else buildLayout();
  vscode.postMessage({ type: 'ready' });
})();
