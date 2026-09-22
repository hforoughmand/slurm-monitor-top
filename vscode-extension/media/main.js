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
 *
 * One snapshot can carry several clusters. Every row then knows which server it
 * came from, and each section is drawn either as one merged panel with a SERVER
 * column or as one panel per server, depending on `slurmTop.mergeServers`. A
 * row's identity is therefore (server, id) and never the id alone: two clusters
 * will happily both have a job 1234 and a node called gpu01.
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

  /** Pins used to be one flat list; they are now per server. */
  function restorePins(value) {
    if (Array.isArray(value)) return { '': value.map(String) };
    return value && typeof value === 'object' ? value : {};
  }

  const state = {
    snapshot: /** @type {any} */ (null),
    sections: ['summary', 'jobs', 'nodes', 'gpus', 'disks'],
    /** The watched clusters, as the host configured them: `[{id, name}]`. */
    servers: /** @type {{id: string, name: string}[]} */ ([]),
    /** Per section: one merged panel, or one panel per server. */
    merge: { summary: false, jobs: true, nodes: true, gpus: true, disks: true },
    detailsIn: 'window',
    /** Which panels the user has folded away, by panel key. */
    collapsed: saved.collapsed || {},
    ownerFilter: saved.ownerFilter || 'me',
    stateFilter: saved.stateFilter || 'all',
    serverFilter: saved.serverFilter || 'all',
    search: saved.search || '',
    /**
     * The same idea as the job filters, for machines: which cluster, what the
     * node is doing, which partition it belongs to, whether it has a GPU going
     * spare, and free text over everything else.
     */
    nodeServerFilter: saved.nodeServerFilter || 'all',
    nodeStateFilter: saved.nodeStateFilter || 'all',
    nodePartitionFilter: saved.nodePartitionFilter || 'all',
    nodeGpuFilter: saved.nodeGpuFilter || 'all',
    nodeSearch: saved.nodeSearch || '',
    sort: Object.assign(
      { jobs: { key: 'default', desc: false }, nodes: { key: 'name', desc: false }, gpus: { key: 'type', desc: false }, disks: { key: 'usage', desc: true } },
      saved.sort || {}
    ),
    /** Selected row per panel, as a `server/id` key. */
    selected: Object.assign({}, saved.selected || {}),
    /** Rows currently on screen per panel, so keyboard nav agrees with the DOM. */
    visible: {},
    /**
     * Job ids pinned to the top, per server. Owned by each collector's config
     * file, not by this view, so a job pinned in the terminal UI is pinned here
     * too; the local copy is updated optimistically so a click feels instant
     * instead of waiting for the next snapshot.
     */
    pinned: restorePins(saved.pinned),
    /** `server/node` whose CPU is being read right now, if any. */
    probing: /** @type {string|null} */ (null),
    /** Which server the detail on screen belongs to. */
    detail: { server: '', name: '' },
  };

  function persist() {
    vscode.setState({
      ownerFilter: state.ownerFilter,
      stateFilter: state.stateFilter,
      serverFilter: state.serverFilter,
      search: state.search,
      nodeServerFilter: state.nodeServerFilter,
      nodeStateFilter: state.nodeStateFilter,
      nodePartitionFilter: state.nodePartitionFilter,
      nodeGpuFilter: state.nodeGpuFilter,
      nodeSearch: state.nodeSearch,
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

  // ----------------------------------------------------------------- servers

  function serverOf(row) {
    return String((row && row.server) || '');
  }

  /** Every server in the snapshot, falling back to what the host configured. */
  function serverList() {
    const fromSnapshot = (state.snapshot && state.snapshot.servers) || [];
    if (fromSnapshot.length) return fromSnapshot;
    return state.servers;
  }

  function serverInfo(id) {
    return serverList().find((s) => s.id === String(id || '')) || null;
  }

  /** The label for a server id: its name, then its hostname, then its id. */
  function serverName(id) {
    const info = serverInfo(id);
    if (info) return info.name || info.host || info.id;
    return String(id || '');
  }

  /** True once more than one cluster is being watched. */
  function manyServers() {
    return serverList().length > 1 || state.servers.length > 1;
  }

  /** Whose account the collector on that server runs as. */
  function userOn(server) {
    const info = serverInfo(server);
    if (info && info.user) return info.user;
    return state.snapshot ? state.snapshot.user : '';
  }

  // -------------------------------------------------------------------- pins

  function pinsOn(server) {
    return state.pinned[String(server || '')] || [];
  }

  function isPinnedOn(server, jobId) {
    return pinsOn(server).indexOf(String(jobId)) >= 0;
  }

  function isPinned(job) {
    return isPinnedOn(serverOf(job), job.job_id);
  }

  /**
   * Flip a pin now and tell the host to persist it.
   *
   * Applied locally first: the round trip spawns a collector process, and a
   * star that takes half a second to light up feels broken.
   */
  function togglePin(server, jobId) {
    const key = String(server || '');
    const id = String(jobId);
    const current = pinsOn(key);
    state.pinned[key] = current.indexOf(id) >= 0 ? current.filter((x) => x !== id) : current.concat([id]);
    persist();
    updateSection('jobs');
    vscode.postMessage({ type: 'togglePin', server: key, jobId: id });
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
      value: (j) => (isPinned(j) ? '★' : '☆'),
      cls: (j) => (isPinned(j) ? 'pin pinned' : 'pin'),
      sortable: false,
      onclick: (j) => togglePin(serverOf(j), j.job_id),
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

  /**
   * The column that says which cluster a row is from.
   *
   * Only added to a merged panel: on a panel that already shows one server it
   * would repeat the panel's own title in every row. A merged GPU row is a
   * total across clusters, so there it names all of them.
   */
  function serverColumn(section) {
    return {
      key: 'server',
      label: compact ? 'SRV' : 'SERVER',
      cls: () => 'server-cell',
      value: (row) =>
        section === 'gpus' && (row.servers || []).length
          ? row.servers.join(', ')
          : row.server_name || serverName(serverOf(row)),
      sort: (row) => row.server_name || serverOf(row),
    };
  }

  const SECTION_SPEC = {
    jobs: { title: 'Jobs', columns: JOB_COLUMNS, id: (j) => j.job_id },
    nodes: { title: 'Nodes', columns: NODE_COLUMNS, id: (n) => n.name },
    gpus: { title: 'GPUs', columns: GPU_COLUMNS, id: (g) => g.type },
    disks: { title: 'Disks', columns: DISK_COLUMNS, id: (d) => d.mount },
  };

  /** A row's identity: two clusters can both have a job 1234. */
  function uidOf(section, row) {
    return `${serverOf(row)}/${SECTION_SPEC[section].id(row)}`;
  }

  function columnsFor(section, merged) {
    const all = SECTION_SPEC[section].columns;
    const visible = compact ? all.filter((c) => !c.compactHide) : all.slice();
    if (!merged || !manyServers()) return visible;
    // After the pin star, which is a control rather than a field, and before
    // everything that identifies the row within its own cluster.
    const at = section === 'jobs' ? 1 : 0;
    const withServer = visible.slice();
    withServer.splice(at, 0, serverColumn(section));
    return withServer;
  }

  // ---------------------------------------------------------------- filters

  function matchesOwner(job) {
    const me = userOn(serverOf(job));
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

  function matchesServer(job) {
    if (state.serverFilter === 'all') return true;
    return serverOf(job) === state.serverFilter;
  }

  function matchesSearch(job) {
    const needle = state.search.trim().toLowerCase();
    if (!needle) return true;
    return [job.job_id, job.user, job.name, job.partition, job.state, job.node_list, job.server_name]
      .join(' ')
      .toLowerCase()
      .includes(needle);
  }

  /** Every partition named by a node, with sinfo's default-partition `*` off. */
  function partitionsOf(nodes) {
    const seen = new Set();
    for (const node of nodes || []) {
      for (const part of partitionsOn(node)) seen.add(part);
    }
    return Array.from(seen).sort();
  }

  /** A node's partitions; sinfo names several as one comma-separated cell. */
  function partitionsOn(node) {
    return String(node.partition || '')
      .split(',')
      .map((part) => part.trim().replace(/\*$/, ''))
      .filter(Boolean);
  }

  function matchesNodeServer(node) {
    if (state.nodeServerFilter === 'all') return true;
    return serverOf(node) === state.nodeServerFilter;
  }

  /**
   * What the machine is doing, in the four buckets worth asking about.
   *
   * Slurm decorates a state with flags -- `idle*` for an unreachable node,
   * `mixed+drain` for one draining while it still has work -- so these match
   * inside the string rather than against it. "unavailable" is checked first:
   * a draining node is not somewhere you can land work, whatever else it says.
   */
  function matchesNodeState(node) {
    if (state.nodeStateFilter === 'all') return true;
    const s = String(node.state || '').toLowerCase();
    const unavailable = /drain|down|fail|err|maint|unk|\*/.test(s);
    if (state.nodeStateFilter === 'unavailable') return unavailable;
    if (unavailable) return false;
    if (state.nodeStateFilter === 'idle') return s.indexOf('idle') >= 0;
    if (state.nodeStateFilter === 'mixed') return s.indexOf('mix') >= 0;
    if (state.nodeStateFilter === 'alloc') return s.indexOf('alloc') >= 0;
    return true;
  }

  function matchesNodePartition(node) {
    if (state.nodePartitionFilter === 'all') return true;
    return partitionsOn(node).indexOf(state.nodePartitionFilter) >= 0;
  }

  function matchesNodeGpu(node) {
    if (state.nodeGpuFilter === 'all') return true;
    if (state.nodeGpuFilter === 'gpu') return (Number(node.gpu_total) || 0) > 0;
    if (state.nodeGpuFilter === 'free') return (Number(node.gpu_free) || 0) > 0;
    if (state.nodeGpuFilter === 'none') return (Number(node.gpu_total) || 0) === 0;
    return true;
  }

  /** Free text over a node's name, state, partition and hardware. */
  function matchesNodeSearch(node) {
    const needle = state.nodeSearch.trim().toLowerCase();
    if (!needle) return true;
    const cpu = node.cpu || {};
    return [
      node.name, node.state, node.reason, node.partition, node.gres,
      (node.gpu_types || []).join(' '), cpu.model, cpu.summary, node.server_name,
    ]
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
    if (!Object.keys(state.pinned).some((key) => state.pinned[key].length)) return rows;
    return rows.filter((j) => isPinned(j)).concat(rows.filter((j) => !isPinned(j)));
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

  /**
   * GPU totals per model, summed over the servers a panel covers.
   *
   * A merged row is "how many of these exist between all the clusters", which
   * is the question the panel answers, so it also carries the names of the
   * clusters that contributed to it.
   */
  function gpuRows(servers) {
    const byType = new Map();
    for (const server of servers) {
      const stats = (server.gpu || {}).per_type_stats || {};
      for (const type of Object.keys(stats)) {
        const cell = stats[type];
        const into = byType.get(type) || {
          type, total: 0, active: 0, reserved: 0, free_est: 0,
          server: servers.length === 1 ? server.id : '',
          server_name: server.name,
          servers: [],
        };
        into.total += Number(cell.total) || 0;
        into.active += Number(cell.active) || 0;
        into.reserved += Number(cell.reserved) || 0;
        into.free_est += Number(cell.free_est) || 0;
        if (into.servers.indexOf(server.name) < 0) into.servers.push(server.name);
        byType.set(type, into);
      }
    }
    return Array.from(byType.values());
  }

  /** The rows one panel shows: its section, narrowed to its server if it has one. */
  function rowsFor(panel) {
    if (!state.snapshot) return [];
    const section = panel.section;
    const onlyHere = (row) => !panel.server || serverOf(row) === panel.server;
    if (section === 'jobs') {
      return (state.snapshot.jobs || []).filter(
        (j) => onlyHere(j) && matchesServer(j) && matchesOwner(j) && matchesState(j) && matchesSearch(j)
      );
    }
    if (section === 'nodes') {
      return (state.snapshot.nodes || []).filter(
        (n) =>
          onlyHere(n) && matchesNodeServer(n) && matchesNodeState(n) &&
          matchesNodePartition(n) && matchesNodeGpu(n) && matchesNodeSearch(n)
      );
    }
    if (section === 'disks') return (state.snapshot.disks || []).filter(onlyHere);
    if (section === 'gpus') {
      const servers = serverList().filter((s) => !panel.server || s.id === panel.server);
      if (servers.length) return gpuRows(servers);
      // A collector older than multi-server support sends no server list.
      const stats = (state.snapshot.gpu || {}).per_type_stats || {};
      return Object.keys(stats).map((type) => Object.assign({ type, server: '', servers: [] }, stats[type]));
    }
    return [];
  }

  // ------------------------------------------------------- panel scaffolding

  /** Built-once DOM handles, keyed by panel key. */
  const panels = {};

  /**
   * Which panels to draw, in order.
   *
   * A section is one panel per server only when it is configured not to merge
   * and there is more than one server: with a single cluster, "merged" and
   * "separate" are the same picture and the extra title would be noise.
   */
  function panelPlan() {
    const servers = serverList();
    const plan = [];
    for (const section of state.sections) {
      if (servers.length > 1 && state.merge[section] === false) {
        for (const server of servers) {
          plan.push({ key: `${section}#${server.id}`, section, server: server.id });
        }
      } else {
        plan.push({ key: section, section, server: null });
      }
    }
    return plan;
  }

  /**
   * What the layout depends on.
   *
   * The server count is part of it, not just the panel list: going from one
   * cluster to two adds the SERVER column to panels whose keys have not
   * changed at all.
   */
  function planSignature(plan) {
    return `${manyServers() ? 'many' : 'one'}:${plan.map((entry) => entry.key).join('|')}`;
  }

  function bar() {
    return el('span', { class: 'bar' }, [el('span', {})]);
  }

  function setBar(node, fraction) {
    const pct = Math.max(0, Math.min(1, Number(fraction) || 0));
    node.className = pct >= 0.9 ? 'bar high' : pct >= 0.7 ? 'bar warn' : 'bar';
    /** @type {HTMLElement} */ (node.firstChild).style.width = (pct * 100).toFixed(1) + '%';
  }

  function makePanel(entry, title, controls, body) {
    const count = el('span', { class: 'panel-count' });
    const titleEl = el('span', { class: 'panel-title', text: title });
    const head = el('div', { class: 'panel-head' }, [titleEl, count, el('span', { class: 'spacer' }), ...(controls || [])]);
    const panel = el('section', {
      class: 'panel',
      dataset: { section: entry.section, panel: entry.key, server: entry.server || '' },
    }, [head, body]);

    // Five panels stacked in a 300px sidebar only works if you can fold the
    // ones you are not watching. The dashboard has the room, so it stays fixed.
    if (compact) {
      titleEl.classList.add('collapsible');
      titleEl.title = 'Click to collapse or expand';
      const apply = () => panel.classList.toggle('collapsed', !!state.collapsed[entry.key]);
      titleEl.addEventListener('click', () => {
        state.collapsed[entry.key] = !state.collapsed[entry.key];
        persist();
        apply();
      });
      apply();
    }
    return { panel, count, titleEl };
  }

  /** `Jobs` on its own, or `Jobs · alpha` on a panel showing one cluster. */
  function panelTitle(entry) {
    const base = SECTION_SPEC[entry.section].title;
    return entry.server ? `${base} · ${serverName(entry.server)}` : base;
  }

  function makeTableSection(entry) {
    const spec = SECTION_SPEC[entry.section];
    const columns = columnsFor(entry.section, !entry.server);

    const headRow = el('tr');
    const headers = columns.map((column) => {
      const sortable = column.sortable !== false;
      const th = el('th', {
        class: [column.numeric ? 'numeric' : '', sortable ? '' : 'unsortable'].filter(Boolean).join(' '),
        text: column.label,
        title: sortable ? `Sort by ${column.label}` : column.title || '',
        onclick: sortable
          ? () => {
              const current = state.sort[entry.section] || {};
              state.sort[entry.section] = current.key === column.key ? { key: column.key, desc: !current.desc } : { key: column.key, desc: false };
              persist();
              updateSection(entry.section);
            }
          : null,
      });
      headRow.appendChild(th);
      return { column, th };
    });

    const tbody = el('tbody');
    const table = el('table', { tabindex: '0' }, [el('thead', {}, [headRow]), tbody]);
    table.addEventListener('keydown', (event) => onTableKey(entry.key, /** @type {KeyboardEvent} */ (event)));

    const empty = el('div', { class: 'empty hidden', text: 'Nothing to show' });
    const body = el('div', { class: 'panel-body' }, [table, empty]);

    const built =
      entry.section === 'jobs' ? { controls: jobControls(entry) }
      : entry.section === 'nodes' ? nodeControls(entry)
      : { controls: [] };
    const { panel, count } = makePanel(entry, panelTitle(entry), built.controls, body);
    panels[entry.key] = {
      key: entry.key, section: entry.section, server: entry.server,
      panel, count, tbody, headers, columns, table, empty, spec,
      // Partitions are not known until a snapshot arrives, so the nodes panel
      // hands back a way to restock that one menu without a relayout.
      refreshFilters: built.refresh || null,
      rowNodes: new Map(), rowData: new Map(),
    };
    return panel;
  }

  function jobControls(entry) {
    const controls = [];

    // Only on a merged panel: one that already shows a single cluster is its
    // own server filter.
    if (!entry.server && manyServers()) {
      const serverSelect = el('select', {
        title: 'Which cluster to show jobs from',
        onchange: (e) => {
          state.serverFilter = e.target.value;
          persist();
          updateSection('jobs');
        },
      });
      serverSelect.appendChild(el('option', { value: 'all', text: 'all servers' }));
      for (const server of serverList()) {
        serverSelect.appendChild(el('option', { value: server.id, text: server.name || server.id }));
      }
      if (!serverList().some((s) => s.id === state.serverFilter)) state.serverFilter = 'all';
      serverSelect.value = state.serverFilter;
      controls.push(serverSelect);
    }

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
    controls.push(ownerSelect);

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
    controls.push(stateSelect);

    const search = el('input', {
      type: 'search',
      placeholder: 'search',
      title: 'Filter by id, user, name, partition, node or server',
      oninput: (e) => {
        state.search = e.target.value;
        persist();
        updateSection('jobs');
      },
    });
    /** @type {HTMLInputElement} */ (search).value = state.search;
    controls.push(search);

    return controls;
  }

  /**
   * The jobs panel's filter strip, for machines.
   *
   * Returns the partition menu alongside the controls: its options come from
   * the snapshot rather than the config, so it has to be restocked as clusters
   * report in, and rebuilding the whole panel for that would lose the user's
   * scroll position and selection.
   */
  function nodeControls(entry) {
    const controls = [];

    const select = (title, key, options, onPick) => {
      const node = el('select', {
        title,
        onchange: (e) => {
          state[key] = e.target.value;
          persist();
          updateSection('nodes');
        },
      });
      for (const [value, label] of options) {
        node.appendChild(el('option', { value, text: label }));
      }
      node.value = state[key];
      controls.push(node);
      if (onPick) onPick(node);
      return node;
    };

    // Only on a merged panel: one that already shows a single cluster is its
    // own server filter.
    if (!entry.server && manyServers()) {
      const options = [['all', 'all servers']].concat(
        serverList().map((s) => [s.id, s.name || s.id])
      );
      if (!serverList().some((s) => s.id === state.nodeServerFilter)) state.nodeServerFilter = 'all';
      select('Which cluster to show machines from', 'nodeServerFilter', options);
    }

    select('What the machine is doing', 'nodeStateFilter', [
      ['all', 'any state'],
      ['idle', 'idle'],
      ['mixed', 'mixed'],
      ['alloc', 'allocated'],
      ['unavailable', 'drained / down'],
    ]);

    let partitionSelect = null;
    select('Which partition the machine belongs to', 'nodePartitionFilter',
      [['all', 'all partitions']], (node) => { partitionSelect = node; });

    select('Whether the machine has GPUs, and any going spare', 'nodeGpuFilter', [
      ['all', 'any GPUs'],
      ['gpu', 'has GPUs'],
      ['free', 'GPUs free'],
      ['none', 'no GPUs'],
    ]);

    const search = el('input', {
      type: 'search',
      placeholder: 'search',
      title: 'Filter by name, state, reason, partition, GRES or CPU model',
      oninput: (e) => {
        state.nodeSearch = e.target.value;
        persist();
        updateSection('nodes');
      },
    });
    /** @type {HTMLInputElement} */ (search).value = state.nodeSearch;
    controls.push(search);

    /** Restock the partition menu, keeping the choice if it still exists. */
    const refresh = () => {
      if (!partitionSelect) return;
      const nodes = (state.snapshot && state.snapshot.nodes) || [];
      const wanted = partitionsOf(
        entry.server ? nodes.filter((n) => serverOf(n) === entry.server) : nodes
      );
      const have = Array.from(partitionSelect.options).slice(1).map((o) => o.value);
      if (have.length === wanted.length && have.every((v, i) => v === wanted[i])) return;
      const chosen = state.nodePartitionFilter;
      while (partitionSelect.options.length > 1) partitionSelect.remove(1);
      for (const part of wanted) {
        partitionSelect.appendChild(el('option', { value: part, text: part }));
      }
      // A partition that has gone away takes its filter with it, rather than
      // leaving the panel mysteriously empty.
      state.nodePartitionFilter = wanted.indexOf(chosen) >= 0 ? chosen : 'all';
      partitionSelect.value = state.nodePartitionFilter;
    };

    return { controls, refresh };
  }

  function makeSummarySection(entry) {
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
    // A merged box hides which clusters it added up, and a sick one would go
    // unnoticed behind the totals, so the servers are listed under it.
    const chips = el('div', { class: 'server-chips hidden' });
    const note = el('div', { class: 'server-note hidden' });
    const body = el('div', { class: 'summary-body' }, [grid, chips, note]);
    // One way in to the server list is enough, so only the first box carries it.
    const first = !entry.server || (serverList()[0] || {}).id === entry.server;
    const manage = first
      ? el('button', {
          class: 'linkish',
          text: 'servers',
          title: 'Add, rename or stop watching a cluster',
          onclick: () => vscode.postMessage({ type: 'manageServers' }),
        })
      : null;
    const { panel, count, titleEl } = makePanel(entry, 'Cluster', [manage], body);
    panels[entry.key] = { key: entry.key, section: 'summary', server: entry.server, panel, count, cells, chips, note, title: titleEl };
    return panel;
  }

  /**
   * Rebuild the panel scaffolding; only on first paint or a layout change.
   *
   * A section split across servers becomes several panels inside one group,
   * rather than several panels loose in the layout: the dashboard places its
   * sections in named grid areas, and two panels claiming the same area would
   * sit on top of each other.
   */
  function buildLayout() {
    const plan = panelPlan();
    rootEl.className = variant;
    rootEl.textContent = '';
    for (const key of Object.keys(panels)) delete panels[key];
    layoutSignature = planSignature(plan);

    const bySection = new Map();
    for (const entry of plan) {
      const node = entry.section === 'summary'
        ? makeSummarySection(entry)
        : SECTION_SPEC[entry.section] && makeTableSection(entry);
      if (!node) continue;
      if (!bySection.has(entry.section)) bySection.set(entry.section, []);
      bySection.get(entry.section).push(node);
    }
    for (const [section, nodes] of bySection) {
      if (nodes.length === 1) rootEl.appendChild(nodes[0]);
      else rootEl.appendChild(el('div', { class: 'panel-group', dataset: { group: section } }, nodes));
    }
    updateAll();
  }

  let layoutSignature = '';

  // ----------------------------------------------------------- in-place update

  /** Refresh every panel drawing this section -- one merged, or one per server. */
  function updateSection(section) {
    for (const key of Object.keys(panels)) {
      if (panels[key].section === section) updatePanel(key);
    }
  }

  function updatePanel(key) {
    const panel = panels[key];
    if (!panel || !state.snapshot) return;
    if (panel.section === 'summary') {
      updateSummary(key);
      return;
    }
    const section = panel.section;

    // Before the rows, so a partition that just appeared can be chosen and one
    // that just went away stops narrowing the table to nothing.
    if (panel.refreshFilters) panel.refreshFilters();

    const rows = sortRows(section, rowsFor(panel), panel.columns);
    state.visible[key] = rows;

    const setting = state.sort[section] || {};
    for (const { column, th } of panel.headers) {
      if (setting.key === column.key) th.setAttribute('aria-sort', setting.desc ? 'descending' : 'ascending');
      else th.removeAttribute('aria-sort');
    }

    const seen = new Set();
    let previous = null;
    for (const row of rows) {
      const uid = uidOf(section, row);
      seen.add(uid);
      panel.rowData.set(uid, row);
      let node = panel.rowNodes.get(uid);
      if (!node) {
        node = makeRow(panel, uid, row);
        panel.rowNodes.set(uid, node);
      }
      const mine = section === 'jobs' && row.user === userOn(serverOf(row));
      fillRow(node, panel, row, mine, state.selected[key] === uid);
      // Re-inserting a node already in the right place is a no-op in the DOM,
      // so a steady cluster produces no layout churn between ticks.
      if (previous ? previous.nextSibling !== node.tr : panel.tbody.firstChild !== node.tr) {
        panel.tbody.insertBefore(node.tr, previous ? previous.nextSibling : panel.tbody.firstChild);
      }
      previous = node.tr;
    }
    for (const [uid, node] of panel.rowNodes) {
      if (!seen.has(uid)) {
        node.tr.remove();
        panel.rowNodes.delete(uid);
        panel.rowData.delete(uid);
      }
    }

    panel.empty.classList.toggle('hidden', rows.length > 0);
    panel.table.classList.toggle('hidden', rows.length === 0);

    if (section === 'jobs') {
      const all = (state.snapshot.jobs || []).filter((j) => !panel.server || serverOf(j) === panel.server);
      panel.count.textContent = `${rows.length}/${all.length}`;
    } else if (section === 'nodes') {
      const all = (state.snapshot.nodes || []).filter((n) => !panel.server || serverOf(n) === panel.server);
      panel.count.textContent = rows.length === all.length ? String(all.length) : `${rows.length}/${all.length}`;
    } else if (section === 'gpus') {
      const free = rows.reduce((sum, row) => sum + (Number(row.free_est) || 0), 0);
      const total = rows.reduce((sum, row) => sum + (Number(row.total) || 0), 0);
      panel.count.textContent = `${free} free / ${total}`;
    } else {
      panel.count.textContent = String(rows.length);
    }
  }

  function makeRow(panel, uid, row) {
    const key = panel.key;
    const tr = el('tr', {
      dataset: { id: String(panel.spec.id(row)), server: serverOf(row), uid },
      onclick: () => select(key, uid),
      ondblclick: () => openDetail(key, uid),
    });
    const cells = panel.columns.map((column) => {
      const td = el('td', { class: column.numeric ? 'numeric' : '' });
      if (column.onclick) {
        td.addEventListener('click', (event) => {
          // Without this the click would also select the row, and a
          // double-click on the star would open the job behind it.
          event.stopPropagation();
          const current = panel.rowData.get(uid);
          if (current) column.onclick(current);
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

  function stateLabel(server) {
    if (!server || server.state === 'running') return '';
    if (server.state === 'error') return 'error';
    if (server.state === 'starting') return 'starting…';
    return 'paused';
  }

  function updateSummary(key) {
    const panel = panels[key];
    if (!panel || !state.snapshot) return;
    const server = panel.server ? serverInfo(panel.server) : null;
    const summary = (server ? server.summary : state.snapshot.summary) || {};
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

    const servers = serverList();
    if (server) {
      panel.title.textContent = server.name || server.host || server.id;
    } else if (servers.length > 1) {
      panel.title.textContent = `All servers (${servers.length})`;
    } else {
      panel.title.textContent = (servers[0] && (servers[0].name || servers[0].host)) || state.snapshot.host || 'Cluster';
    }

    const at = server ? server.timestamp : state.snapshot.timestamp;
    panel.count.textContent = stateLabel(server) || (at ? new Date(at * 1000).toLocaleTimeString() : '');
    panel.count.classList.toggle('error-text', !!server && server.state === 'error');

    // On a merged box, one line per cluster: a server that stopped answering
    // must not simply drop out of totals that still look plausible.
    const chips = panel.chips;
    if (!server && servers.length > 1) {
      chips.textContent = '';
      for (const info of servers) {
        const label = stateLabel(info) || `${info.counts ? info.counts.jobs : 0} jobs`;
        chips.appendChild(
          el('span', {
            class: `server-chip${info.state === 'error' ? ' error-text' : ''}`,
            title: info.message || info.host || info.id,
            text: `${info.name || info.id}: ${label}`,
          })
        );
      }
      chips.classList.remove('hidden');
    } else {
      chips.classList.add('hidden');
    }

    const message = server && server.state === 'error' ? server.message : '';
    panel.note.textContent = message || '';
    panel.note.className = `server-note error-text${message ? '' : ' hidden'}`;
  }

  function updateAll() {
    for (const key of Object.keys(panels)) updatePanel(key);
  }

  // -------------------------------------------------------------- selection

  function select(key, uid) {
    const previous = state.selected[key];
    state.selected[key] = String(uid);
    persist();
    const panel = panels[key];
    if (!panel || !panel.rowNodes) return;
    const before = panel.rowNodes.get(String(previous));
    if (before) before.tr.classList.remove('selected');
    const now = panel.rowNodes.get(String(uid));
    if (now) now.tr.classList.add('selected');
  }

  function onTableKey(key, event) {
    const rows = state.visible[key] || [];
    if (!rows.length) return;
    const panel = panels[key];
    const idOf = (row) => uidOf(panel.section, row);
    const index = rows.findIndex((row) => idOf(row) === String(state.selected[key]));

    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const next = event.key === 'ArrowDown' ? Math.min(rows.length - 1, index + 1) : Math.max(0, index - 1);
      const uid = idOf(rows[next < 0 ? 0 : next]);
      select(key, uid);
      const node = panel.rowNodes.get(uid);
      if (node) node.tr.scrollIntoView({ block: 'nearest' });
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (index >= 0) openDetail(key, idOf(rows[index]));
    } else if (event.key.toLowerCase() === 'c' && index >= 0) {
      event.preventDefault();
      copyRow(panel.section, rows[index]);
    } else if (event.key.toLowerCase() === 'p' && index >= 0 && panel.section === 'jobs') {
      event.preventDefault();
      togglePin(serverOf(rows[index]), rows[index].job_id);
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

  function openDetail(key, uid) {
    const panel = panels[key];
    if (!panel) return;
    const row = panel.rowData.get(String(uid));
    if (!row) return;
    const server = serverOf(row);
    if (panel.section === 'jobs') {
      requestDetail('job', row.job_id, server);
    } else if (panel.section === 'nodes') {
      requestDetail('node', row.name, server);
    } else if (panel.section === 'gpus') {
      showGpuJobs(row);
    } else if (panel.section === 'disks') {
      showOverlay(titleWithServer(`Disk ${row.mount}`, server), [kvList(row)], JSON.stringify(row, null, 2));
    }
  }

  /** `Job 1234` on one cluster, `Job 1234 · alpha` when there are several. */
  function titleWithServer(title, server) {
    return manyServers() && server ? `${title} · ${serverName(server)}` : title;
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

  /**
   * The jobs behind one GPU row.
   *
   * A merged row covers every cluster that has that model, so the list does
   * too -- and then it has to say which cluster each job is on.
   */
  function showGpuJobs(row) {
    const type = String(row.type);
    const scope = String(row.server || '');
    const jobs = (state.snapshot.jobs || []).filter(
      (j) => Object.keys(j.gpu_types || {}).indexOf(type) >= 0 && (!scope || serverOf(j) === scope)
    );
    const showServer = !scope && manyServers();
    const children = [el('h3', { text: `Jobs holding ${type}` })];
    if (!jobs.length) {
      children.push(el('p', { class: 'hint', text: 'No job currently holds this GPU type.' }));
    } else {
      const headers = ['JOBID', 'USER', 'ST', 'GPUS', 'NAME', 'TIME'];
      if (showServer) headers.splice(1, 0, 'SERVER');
      children.push(
        jobTable(jobs, headers, (j) => {
          const cells = [
            el('td', { class: 'link', text: j.job_id, title: 'Open this job', onclick: () => requestDetail('job', j.job_id, serverOf(j)) }),
            el('td', { text: j.user }),
            el('td', { text: shortState(j.state), class: stateClass(j.state) }),
            el('td', { class: 'numeric', text: (j.gpu_types || {})[type] || 0 }),
            el('td', { text: j.name }),
            el('td', { class: 'numeric', text: j.time_used }),
          ];
          if (showServer) cells.splice(1, 0, el('td', { text: j.server_name || serverOf(j) }));
          return cells;
        })
      );
    }
    showOverlay(titleWithServer(`GPU ${type}`, scope), children);
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
    const server = state.detail.server;
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
                      type: 'openOutput', server, jobId, stream: name, path: file.path, size: file.size,
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
    const server = state.detail.server;
    children.push(
      el('p', { class: 'hint' }, [
        el('button', {
          text: isPinnedOn(server, jobId) ? '★ Unpin this job' : '☆ Pin this job',
          title: 'Pinned jobs are listed first, in the editor and in the terminal UI',
          onclick: () => {
            togglePin(server, jobId);
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
    const server = state.detail.server;
    const busy = state.probing === `${server || ''}/${node}`;
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
              ? 'Reading the CPU from the node… '
              : 'Slurm does not report the CPU model or its speed. ',
          }),
          busy
            ? null
            : el('button', {
                text: 'Read it from the node',
                title: 'Connects over ssh, or failing that runs a one-second job there',
                onclick: () => {
                  state.probing = `${server || ''}/${node}`;
                  vscode.postMessage({ type: 'probeCpu', server, node });
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
          el('td', { class: 'link', text: j.job_id, title: 'Open this job', onclick: () => requestDetail('job', j.job_id, state.detail.server) }),
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
    const what = isJob ? `Job ${payload.job_id}` : `Node ${payload.node}`;
    const title = state.detail.name ? `${what} · ${state.detail.name}` : what;
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
  function requestDetail(kind, id, server) {
    if (state.detailsIn === 'overlay' && !isDetailView) {
      showOverlay(
        titleWithServer(`${kind === 'job' ? 'Job' : 'Node'} ${id}`, server),
        [el('p', { class: 'hint', text: 'Loading scontrol details…' })]
      );
    }
    vscode.postMessage({ type: 'openDetail', kind, id: String(id), server: String(server || '') });
  }

  // ----------------------------------------------------------------- status

  /** The banner's current text, so a repeated status does not rebuild it. */
  let statusShown = null;

  function setStatus(message, kind) {
    if (!message) {
      statusShown = null;
      statusEl.classList.add('hidden');
      return;
    }
    // Several servers means several status messages a second, most of them the
    // one already on screen; rebuilding it every time makes the buttons
    // unclickable.
    if (statusShown === `${kind || ''}:${message}`) {
      return;
    }
    statusShown = `${kind || ''}:${message}`;
    statusEl.textContent = '';
    statusEl.className = `status${kind === 'error' ? ' error' : ''}`;
    statusEl.appendChild(el('span', { text: message }));
    if (kind === 'error') {
      statusEl.appendChild(el('button', { text: 'Show log', onclick: () => vscode.postMessage({ type: 'showLog' }) }));
      statusEl.appendChild(el('button', { text: 'Retry', onclick: () => vscode.postMessage({ type: 'refresh' }) }));
    }
  }

  // --------------------------------------------------------------- messages

  /** Rebuild only when the panels themselves would differ. */
  function relayoutIfNeeded() {
    if (!Object.keys(panels).length || planSignature(panelPlan()) !== layoutSignature) {
      buildLayout();
      return true;
    }
    return false;
  }

  /** Take the pin lists from a snapshot: per server, or the one flat list. */
  function adoptPins(snapshot) {
    const servers = snapshot.servers || [];
    if (servers.length) {
      const pins = {};
      for (const server of servers) pins[server.id] = (server.pinned || []).map(String);
      state.pinned = pins;
    } else if (Array.isArray(snapshot.pinned)) {
      state.pinned = { '': snapshot.pinned.map(String) };
    }
  }

  window.addEventListener('message', (event) => {
    const message = event.data;
    switch (message.type) {
      case 'config': {
        if (!state.snapshot && !saved.ownerFilter) state.ownerFilter = message.ownerFilter;
        state.detailsIn = message.detailsIn || 'window';
        if (Array.isArray(message.servers)) state.servers = message.servers;
        if (message.merge) state.merge = Object.assign({}, state.merge, message.merge);
        if (isDetailView) break;
        if (Array.isArray(message.sections)) state.sections = message.sections;
        relayoutIfNeeded();
        break;
      }
      case 'snapshot':
        adoptPins(message.snapshot);
        if (isDetailView) break;
        state.snapshot = message.snapshot;
        setStatus(null);
        if (!relayoutIfNeeded()) updateAll();
        break;
      case 'status':
        if (isDetailView) break;
        if (message.state === 'error') setStatus(message.message || 'Collector error', 'error');
        // Some servers are answering and some are not: the working ones keep
        // the tables live, and the banner names the ones that are down.
        else if (message.state === 'running' && message.message) setStatus(message.message, 'error');
        else if (message.state === 'paused' && message.message) setStatus(message.message);
        else if (message.state === 'starting' && !state.snapshot) setStatus('Starting the Slurm collector…');
        else if (message.state === 'paused' && !state.snapshot) setStatus('Paused — open a Slurm view to start polling.');
        else if (state.snapshot) setStatus(null);
        break;
      case 'detail':
        state.detail = {
          server: String(message.server || ''),
          // Only worth showing when there is more than one cluster to confuse.
          name: manyServers() ? message.serverName || serverName(message.server) : '',
        };
        presentDetail(message.detail);
        break;
      case 'detailError':
        if (isDetailView) setStatus(message.message, 'error');
        else showOverlay('Details unavailable', [el('p', { class: 'hint', text: message.message })]);
        break;
      case 'detailTarget':
        state.detail = {
          server: String(message.server || ''),
          name: manyServers() ? message.serverName || serverName(message.server) : '',
        };
        setStatus(`Loading ${message.kind} ${message.id}…`);
        break;
      case 'pinned': {
        // The host is the authority: a pin made in the terminal UI, or one of
        // ours that failed to persist, corrects the optimistic local list.
        const key = String(message.server || '');
        state.pinned[key] = (message.pinned || []).map(String);
        persist();
        if (!isDetailView) updateSection('jobs');
        break;
      }
      case 'cpuProbe': {
        const key = `${message.server || ''}/${message.node}`;
        state.probing = message.state === 'started' ? key : null;
        if (message.state === 'started') {
          setStatus(`Reading ${message.node}'s CPU — ssh, then a one-second job…`);
        } else if (message.message) {
          setStatus(`Could not read ${message.node}'s CPU: ${message.message}`, 'error');
        } else {
          setStatus(null);
        }
        break;
      }
    }
  });

  if (isDetailView) setStatus('Loading details…');
  else buildLayout();
  vscode.postMessage({ type: 'ready' });
})();
