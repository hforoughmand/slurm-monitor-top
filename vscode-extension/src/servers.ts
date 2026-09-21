import * as vscode from 'vscode';

/**
 * The clusters to watch, read from `slurmTop.servers`.
 *
 * One collector process per server: the parsing lives in the collector, so
 * watching a second cluster is a matter of running a second one of those --
 * over ssh, in a container, wherever `squeue` answers -- and tagging what comes
 * back with which server it came from.
 *
 * The setting is a plain `name -> where` map so that VS Code's settings editor
 * can edit it itself, as a two-column grid with an Add Item button. An array of
 * objects would be richer and would offer nothing but "Edit in settings.json".
 */

/** One cluster: how to reach it, and what to call it. */
export interface ServerSpec {
  /** Stable key used in row ids, messages and selections; never contains `/`. */
  id: string;
  /** What to show. Empty only for the unnamed server of a pre-0.7 setup. */
  name: string;
  /** Full argv that prints slurm-top JSON. Empty means: resolve one locally. */
  command: string[];
  /**
   * The `ssh …` prefix, when the address named a destination rather than a
   * whole command. What lets the resolver try other ways of running the
   * collector there -- including sending it a copy.
   */
  ssh?: string[];
  /** Interpreter for the local resolution path; ignored when `command` is set. */
  pythonPath: string;
}

/**
 * `BatchMode` so a key that needs a passphrase fails immediately instead of
 * hanging on a password prompt no one can answer: the collector is a spawned
 * process with no terminal attached.
 */
export const DEFAULT_SSH_ARGS = ['-o', 'BatchMode=yes'];

/** What to run on the far side of an ssh destination. */
export const DEFAULT_REMOTE_COMMAND = ['slurm-top', '--json'];

/** The id given to a server list that has been emptied out entirely. */
export const LEGACY_SERVER_ID = 'default';

/** The row `slurmTop.servers` starts with, matching the default in package.json. */
export const DEFAULT_LOCAL_NAME = 'this machine';

/** What an unconfigured list holds: the machine the extension runs on. */
export const DEFAULT_SERVERS: Record<string, string> = { [DEFAULT_LOCAL_NAME]: 'here' };

/**
 * Split a command line into argv, respecting quotes.
 *
 * Part of reading the settings rather than of any dialog: `slurmTop.servers`
 * holds one string per server, and a command is one of the things that string
 * can be.
 */
export function splitCommand(line: string): string[] {
  const out: string[] = [];
  let current = '';
  let quote: string | null = null;
  let started = false;
  for (const char of String(line).trim()) {
    if (quote) {
      if (char === quote) {
        quote = null;
      } else {
        current += char;
      }
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      started = true;
      continue;
    }
    if (/\s/.test(char)) {
      if (started || current) {
        out.push(current);
      }
      current = '';
      started = false;
      continue;
    }
    current += char;
  }
  if (started || current) {
    out.push(current);
  }
  return out;
}

/**
 * Ways of writing "the machine this is already running on".
 *
 * Empty is the documented one, but `here` is the word the dialogs and the
 * settings description use for it, so someone will type that instead -- and a
 * setting that turns a plain-English answer into `ssh here` has only itself to
 * blame.
 */
const HERE = new Set(['', 'here', 'local', 'localhost', 'this machine', 'this']);

/**
 * ssh options that swallow the next word, so that the word after them is not
 * mistaken for the destination.
 */
const SSH_OPTION_TAKES_VALUE = new Set([
  '-b', '-c', '-D', '-E', '-e', '-F', '-I', '-i', '-J', '-L', '-l',
  '-m', '-O', '-o', '-P', '-p', '-Q', '-R', '-S', '-W', '-w',
]);

/**
 * Whether an ssh command line says what to run on the far side.
 *
 * `ssh login01` on its own is a login shell, not a collector: appending
 * `--watch 3` to it would send that to ssh, which answers with its usage. So
 * the default remote command is added for it, exactly as it would be for the
 * bare destination `login01`.
 */
function sshNamesACommand(argv: string[]): boolean {
  let at = 1;
  while (at < argv.length && argv[at].startsWith('-')) {
    const option = argv[at];
    if (SSH_OPTION_TAKES_VALUE.has(option)) {
      at += 2;
      continue;
    }
    // Bundled single-letter flags (`-tt`, `-4v`); only a value-taking one at
    // the end takes the next word.
    const last = `-${option.slice(-1)}`;
    at += /^-[A-Za-z0-9]{2,}$/.test(option) && SSH_OPTION_TAKES_VALUE.has(last) ? 2 : 1;
  }
  // argv[at] is the destination; anything past it is the remote command.
  return at + 1 < argv.length;
}

/**
 * What a server's `where` says to run.
 *
 * Three forms, told apart by shape alone so that the settings grid needs no
 * syntax of its own:
 *
 * - empty, or `here`      -- the machine the extension is running on
 * - `me@login01`          -- an ssh destination, one word, as you would type it
 * - anything with a space -- a command line, run as it stands, except that an
 *   `ssh` line naming no remote command is finished off with `slurm-top --json`
 */
export interface Where {
  /** The argv to run. Empty means: find a collector on this machine. */
  command: string[];
  /** Set when the address named an ssh destination rather than a whole command. */
  ssh?: string[];
}

export function whereFor(where: string): Where {
  const value = String(where ?? '').trim();
  if (HERE.has(value.toLowerCase())) {
    return { command: [] };
  }
  if (/\s/.test(value)) {
    const argv = splitCommand(value);
    if (argv[0] === 'ssh' && !sshNamesACommand(argv)) {
      return { command: [...argv, ...DEFAULT_REMOTE_COMMAND], ssh: argv };
    }
    // A command that says what to run says all of it: there is nothing to
    // substitute a different collector into.
    return { command: argv };
  }
  const ssh = ['ssh', ...DEFAULT_SSH_ARGS, value];
  return { command: [...ssh, ...DEFAULT_REMOTE_COMMAND], ssh };
}

export function commandFor(where: string): string[] {
  return whereFor(where).command;
}

/** How a spec will be run, for a list or a tooltip. */
export function describeCommand(command: string[]): string {
  return command.length ? command.join(' ') : 'here, wherever the extension runs';
}

function slug(value: string): string {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function uniqueId(base: string, taken: Set<string>): string {
  const root = base || 'server';
  let id = root;
  let n = 2;
  while (taken.has(id)) {
    id = `${root}-${n}`;
    n += 1;
  }
  taken.add(id);
  return id;
}

/**
 * The raw setting: what the settings grid shows, in the order it holds it.
 *
 * Falls back to the default row here rather than relying on VS Code to supply
 * the manifest default, so that the list is the same one thing everywhere --
 * including in the tests, which have no manifest.
 */
export function readServerSettings(): Record<string, string> {
  const value = vscode.workspace.getConfiguration('slurmTop').get<Record<string, string>>('servers', {});
  const configured = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  return Object.keys(configured).some((name) => name.trim()) ? configured : { ...DEFAULT_SERVERS };
}

/**
 * The servers to watch right now.
 *
 * An empty `slurmTop.servers` means the old single-server settings are still in
 * force, so one spec is synthesised from them; that is what every installation
 * before multi-server support has, and it must keep working untouched.
 */
export function readServers(): ServerSpec[] {
  const settings = vscode.workspace.getConfiguration('slurmTop');
  const configured = readServerSettings();
  const pythonPath = (settings.get<string>('pythonPath', '') ?? '').trim();
  const legacy = settings.get<string[]>('command', []) ?? [];

  const taken = new Set<string>();
  return Object.keys(configured)
    .filter((name) => name.trim())
    .map((key) => {
      const name = key.trim();
      const { command, ssh } = whereFor(configured[key]);
      const local = command.length === 0;
      return {
        id: uniqueId(slug(name), taken),
        ssh,
        // A local server called `this machine` -- the row everyone starts with
        // -- is labelled with the hostname it reports instead, which says more.
        // Rename it and that name is used, as it is for any other server.
        name: local && HERE.has(name.toLowerCase()) ? '' : name,
        // A row that says `here` is the machine this runs on, so it is run the
        // way that machine always was -- `slurmTop.command` and all.
        command: local ? legacy : command,
        pythonPath,
      };
    });
}

/** Everything that decides how a server is reached; a change here needs a respawn. */
export function specSignature(spec: ServerSpec): string {
  return JSON.stringify([spec.id, spec.command, spec.ssh, spec.pythonPath]);
}

/** Which sections show one merged panel rather than one panel per server. */
export type MergeSettings = Record<string, boolean>;

const MERGE_DEFAULTS: MergeSettings = {
  // The summary box names the cluster it describes, so one box per server is
  // the honest default; the tables lose nothing by being stacked.
  summary: false,
  jobs: true,
  nodes: true,
  gpus: true,
  disks: true,
};

export function readMergeSettings(): MergeSettings {
  const configured = vscode.workspace
    .getConfiguration('slurmTop')
    .get<MergeSettings>('mergeServers', {}) ?? {};
  const merged: MergeSettings = { ...MERGE_DEFAULTS };
  for (const key of Object.keys(MERGE_DEFAULTS)) {
    if (typeof configured[key] === 'boolean') {
      merged[key] = configured[key];
    }
  }
  return merged;
}
