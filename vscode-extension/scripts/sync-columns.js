#!/usr/bin/env node
/**
 * Write the four column settings into package.json from one list.
 *
 * VS Code needs a static schema to draw the checkboxes, `src/columns.ts` needs
 * the defaults, and `media/main.js` needs the columns themselves. Three lists
 * of the same thing drift, so this generates the schema and `npm test` checks
 * it against the webview's real columns.
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');

/** Label and one-line description per column, in the order they appear. */
const COLUMNS = {
  jobs: {
    title: 'jobs table',
    columns: {
      job_id: 'The job id.',
      user: 'Who submitted it.',
      state: 'Running, pending, completing.',
      partition: 'The partition it is queued on.',
      name: 'The job name.',
      nodes: 'How many machines it asked for.',
      cpus: 'How many cores it asked for.',
      gpus: 'How many GPUs it holds.',
      mem: 'Memory requested.',
      time: 'Elapsed run time.',
      node_list: 'The machines it is running on.',
    },
  },
  nodes: {
    title: 'machines table',
    columns: {
      name: 'The machine name.',
      state: 'Idle, mixed, allocated, drained, with the drain reason inline.',
      partition: 'The partitions it belongs to.',
      cpus: 'Allocated out of total cores, with a bar.',
      cpus_idle: 'Cores going spare.',
      cpus_total: 'Total cores, as a column of its own.',
      load: '1-minute load against the core count, with a bar.',
      mem_total: 'Installed memory.',
      mem_free: 'Free memory, with a bar.',
      mem_used: 'Memory currently reserved.',
      gpus: 'GPUs in use out of installed.',
      gpu_free: 'GPUs going spare.',
      gres: 'The GPU models installed.',
      reason: 'Why a machine is drained or down, as a sortable column.',
      topology: 'Sockets x cores per socket x threads per core.',
      cpu: 'The CPU model, once it has been read from the machine.',
    },
  },
  gpus: {
    title: 'GPU table',
    columns: {
      node: 'The machine the GPUs are in.',
      type: 'The GPU model.',
      used: 'In use out of installed on that machine, with a bar.',
      free: 'How many of them are spare.',
      mem_free: "The machine's free memory, with a bar.",
      cpus_idle: "The machine's spare cores.",
      state: 'What the machine is doing.',
      partition: 'The partitions the machine belongs to.',
      load: "The machine's load, with a bar.",
    },
  },
  disks: {
    title: 'disks table',
    columns: {
      usage: 'Percentage used, with a bar.',
      mount: 'The mount point.',
      size: 'Total size.',
      used: 'Space used.',
      avail: 'Space free.',
      type: 'The filesystem type.',
    },
  },
};

/** `nodes` -> `nodeColumns`, matching `settingKey` in src/columns.ts. */
function settingKey(section) {
  return `${section.replace(/s$/, '')}Columns`;
}

function build(defaults) {
  const out = {};
  for (const [section, spec] of Object.entries(COLUMNS)) {
    const properties = {};
    for (const [key, description] of Object.entries(spec.columns)) {
      properties[key] = { type: 'boolean', description };
    }
    out[`slurmTop.${settingKey(section)}`] = {
      type: 'object',
      additionalProperties: false,
      default: defaults[section],
      markdownDescription:
        `Which columns the ${spec.title} shows. Untick one to hide it.\n\n` +
        'The sidebar drops the widest columns to fit; once you have chosen ' +
        'columns here yourself, it shows exactly what you asked for instead.',
      properties,
    };
  }
  return out;
}

function defaultsFromTs() {
  // Read the defaults out of src/columns.ts rather than repeating them, so the
  // schema's defaults and the host's defaults are the same list.
  const source = fs.readFileSync(path.join(ROOT, 'src', 'columns.ts'), 'utf8');
  const body = source.slice(
    source.indexOf('export const COLUMN_DEFAULTS'),
    source.indexOf('/** The settings key')
  );
  const defaults = {};
  for (const section of Object.keys(COLUMNS)) {
    const block = new RegExp(`\\b${section}:\\s*\\{([\\s\\S]*?)\\n  \\}`).exec(body);
    if (!block) throw new Error(`no defaults for ${section} in src/columns.ts`);
    defaults[section] = {};
    for (const line of block[1].split('\n')) {
      const m = /^\s*([a-z_]+):\s*(true|false),/.exec(line);
      if (m) defaults[section][m[1]] = m[2] === 'true';
    }
  }
  return defaults;
}

function main() {
  const manifestPath = path.join(ROOT, 'package.json');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const generated = build(defaultsFromTs());

  const properties = manifest.contributes.configuration.properties;
  for (const [key, value] of Object.entries(generated)) {
    properties[key] = value;
  }
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  console.log(`synced ${Object.keys(generated).length} column settings`);
}

if (require.main === module) main();
module.exports = { COLUMNS, settingKey, defaultsFromTs };
