import { ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Running the collector on a machine that does not have it.
 *
 * The data layer is pure standard library and the extension already carries a
 * copy of it, so a cluster reachable by ssh does not have to have
 * slurm-monitor-top installed: its source is piped, as plain readable Python,
 * into `python3 -` on the far side. Nothing is written to that machine and
 * nothing is left behind.
 *
 * Plain text on stdin rather than anything packed into an argument: a
 * compressed, base64'd blob unpacked by `exec` looks to a cluster's process
 * accounting and security tooling exactly like what malware does, and the
 * process list would carry 26KB of it. This way `ps` shows `python3 -`, and
 * what ran is the same source anyone can read in this repository.
 *
 * It is the last thing tried, not the first -- an installed collector needs no
 * source sent at all -- but it is what makes "watch that cluster too" a matter
 * of typing its hostname.
 */

/**
 * One script out of the two modules the exporter needs: `data`, then `export`
 * with its `from .data import (...)` dropped, since both now share one
 * namespace. `__init__` only carries a version string.
 */
const RELATIVE_IMPORT = /^from \.data import \([^)]*\)\n/m;

/**
 * Quote one word for the shell on the other end of ssh.
 *
 * ssh joins its arguments with spaces and hands the result to a shell there, so
 * anything with a space or a quote in it has to arrive already quoted.
 */
export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

let script: string | null | undefined;

/** The collector as one plain-text script; null when the copy is not bundled. */
function bundledScript(extensionPath: string): string | null {
  if (script !== undefined) {
    return script;
  }
  try {
    const dir = path.join(extensionPath, 'python', 'slurm_top');
    const data = fs.readFileSync(path.join(dir, 'data.py'), 'utf8');
    const exporter = fs.readFileSync(path.join(dir, 'export.py'), 'utf8');
    const joined = exporter.replace(RELATIVE_IMPORT, '');
    // A module that grew another relative import would fail on the far side
    // with a NameError nobody could explain; better not to offer it at all.
    if (joined === exporter || /^\s*from \.|^\s*import \./m.test(data + joined)) {
      throw new Error('the bundled modules no longer join into one script');
    }
    script =
      '# slurm-monitor-top collector, sent by the VS Code extension over ssh stdin.\n' +
      '# slurm_top/data.py and slurm_top/export.py, joined; nothing is written to disk.\n\n' +
      `${data}\n\n# ---- slurm_top/export.py ----\n\n${joined}`;
  } catch {
    script = null;
  }
  return script;
}

/** How big the sent copy is, for the log. */
export function payloadSize(extensionPath: string): number {
  return bundledScript(extensionPath)?.length ?? 0;
}

/**
 * `ssh … python3 -` and the source to write to its stdin, or undefined with no
 * bundled copy. Flags appended to the argv reach the exporter as `sys.argv[1:]`.
 *
 * `python` as well as `python3`, because a cluster old enough to have only the
 * one is exactly the kind that will not have the package either.
 */
export function pushedCollector(
  sshPrefix: string[],
  extensionPath: string,
  interpreter: string
): { argv: string[]; stdin: string } | undefined {
  const source = bundledScript(extensionPath);
  if (!source) {
    return undefined;
  }
  return { argv: [...sshPrefix, interpreter, '-'], stdin: source };
}

/**
 * Hand a child its stdin, if it has any to get.
 *
 * The write can fail -- the command was not found, ssh gave up before reading
 * -- and an unhandled 'error' on the pipe would take the extension host down
 * with it. The child's own exit says what went wrong, so the pipe's is dropped.
 */
export function feedStdin(child: ChildProcess, text: string | undefined): void {
  if (text === undefined || !child.stdin) {
    return;
  }
  child.stdin.on('error', () => undefined);
  child.stdin.end(text);
}

/**
 * An argv fit for a log line.
 *
 * A configured command can carry an argument of any length -- an inline
 * script, a long `--wrap` -- and one such would bury every other line in the
 * channel.
 */
export function shortArgv(argv: string[]): string {
  return argv
    .map((arg) => (arg.length > 60 ? `<${arg.length} bytes>` : arg))
    .join(' ');
}
