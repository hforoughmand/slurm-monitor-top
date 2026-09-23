// Columns are described in three places that have to agree: the real column
// definitions in media/main.js, the host's defaults in src/columns.ts, and the
// checkbox schema in package.json that the settings page draws from. Nothing
// stops them drifting except this.
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SECTIONS = ['jobs', 'nodes', 'gpus', 'disks'];

let failures = 0;
function check(name, ok, detail) {
  if (ok) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.log(`  not ok ${name}${detail ? ` -- ${detail}` : ''}`);
  }
}

/** The column keys the webview actually renders, per section. */
function columnsInWebview() {
  const source = fs.readFileSync(path.join(ROOT, 'media', 'main.js'), 'utf8');
  const names = { jobs: 'JOB_COLUMNS', nodes: 'NODE_COLUMNS', gpus: 'GPU_COLUMNS', disks: 'DISK_COLUMNS' };
  const out = {};
  for (const [section, constant] of Object.entries(names)) {
    const start = source.indexOf(`const ${constant} = [`);
    if (start < 0) throw new Error(`${constant} not found in media/main.js`);
    const end = source.indexOf('\n  ];', start);
    const body = source.slice(start, end);
    out[section] = Array.from(body.matchAll(/key: '([a-z_]+)'/g))
      .map((m) => m[1])
      // The pin star is a control, not a field, so it is never configurable.
      .filter((key) => key !== 'pin');
  }
  return out;
}

/** The defaults the extension host applies. */
function columnsInHost() {
  const { defaultsFromTs } = require('../scripts/sync-columns.js');
  return defaultsFromTs();
}

/** The checkboxes the settings page draws. */
function columnsInManifest() {
  const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
  const properties = manifest.contributes.configuration.properties;
  const out = {};
  for (const section of SECTIONS) {
    const key = `slurmTop.${section.replace(/s$/, '')}Columns`;
    const setting = properties[key];
    if (!setting) throw new Error(`${key} is not declared in package.json`);
    out[section] = setting;
  }
  return out;
}

console.log('columns:');
const webview = columnsInWebview();
const host = columnsInHost();
const manifest = columnsInManifest();

for (const section of SECTIONS) {
  const inView = webview[section];
  const inHost = Object.keys(host[section]);
  const inManifest = Object.keys(manifest[section].properties);

  check(`${section}: the settings page offers every column the table has`,
    inView.every((k) => inManifest.includes(k)),
    inView.filter((k) => !inManifest.includes(k)).join(','));
  check(`${section}: the settings page offers nothing the table lacks`,
    inManifest.every((k) => inView.includes(k)),
    inManifest.filter((k) => !inView.includes(k)).join(','));
  check(`${section}: the host has a default for every column`,
    inView.every((k) => inHost.includes(k)) && inHost.every((k) => inView.includes(k)),
    `host=${inHost.join(',')} view=${inView.join(',')}`);
  check(`${section}: the schema's defaults match the host's`,
    JSON.stringify(manifest[section].default) === JSON.stringify(host[section]),
    `schema=${JSON.stringify(manifest[section].default)}`);
  check(`${section}: at least one column is on by default`,
    Object.values(host[section]).some(Boolean));
}

// A column turned off by default in the host must be marked `off` in the
// webview, or the two disagree about a fresh install.
const source = fs.readFileSync(path.join(ROOT, 'media', 'main.js'), 'utf8');
for (const section of SECTIONS) {
  for (const [key, on] of Object.entries(host[section])) {
    if (on) continue;
    const constant = { jobs: 'JOB_COLUMNS', nodes: 'NODE_COLUMNS', gpus: 'GPU_COLUMNS', disks: 'DISK_COLUMNS' }[section];
    const start = source.indexOf(`const ${constant} = [`);
    const body = source.slice(start, source.indexOf('\n  ];', start));
    const entry = new RegExp(`key: '${key}'[\\s\\S]{0,400}?off: true`).test(body);
    check(`${section}.${key} is marked off in the webview too`, entry);
  }
}

if (failures) {
  console.log(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nall checks passed');
