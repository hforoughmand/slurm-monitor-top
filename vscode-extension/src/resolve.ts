import { spawn } from 'child_process';
import * as path from 'path';
import * as vscode from 'vscode';

import { payloadSize, pushedCollector, shortArgv } from './remote';
import { ServerSpec } from './servers';

/** An argv prefix that prints slurm-top JSON when data-mode flags are appended. */
export interface Collector {
  argv: string[];
  /** Extra environment the command needs (PYTHONPATH for the bundled copy). */
  env?: Record<string, string>;
  /** Human-readable origin, for the log and error messages. */
  origin: string;
}

/**
 * A string that only the JSON exporter's `--help` prints. An older
 * slurm-monitor-top ignores `--json` and starts the TUI instead, so a probe
 * that trusted the exit code alone could adopt a command that never emits JSON.
 */
const HELP_MARKER = '--watch';

/** Candidates in priority order: explicit override, installed CLI, bundled copy. */
function candidates(spec: ServerSpec, extensionPath: string): Collector[] {
  if (spec.ssh) {
    // An address that named a machine rather than a command: try the collector
    // it may have, then the module it may have, and failing both send it one.
    // Ordered by what each costs -- an installed collector needs no payload,
    // and the pushed copy needs nothing of the cluster at all.
    const where = spec.ssh[spec.ssh.length - 1];
    const list: Collector[] = [
      { argv: spec.command, origin: `slurm-top on ${where}` },
      {
        argv: [...spec.ssh, 'python3', '-m', 'slurm_top.export'],
        origin: `python3 -m slurm_top.export on ${where}`,
      },
    ];
    for (const interpreter of ['python3', 'python']) {
      const argv = pushedCollector(spec.ssh, extensionPath, interpreter);
      if (argv) {
        list.push({
          argv,
          origin:
            `the copy bundled in the extension, sent to ${where} for ${interpreter} ` +
            `(${payloadSize(extensionPath)} bytes, nothing installed there)`,
        });
      }
    }
    return list;
  }

  if (spec.command.length > 0) {
    // A command that was written down -- a container, a wrapper, an ssh line
    // that says what to run -- is the whole answer: nothing can be substituted
    // into it.
    return [{ argv: spec.command, origin: `configured command for ${spec.name || spec.id}` }];
  }

  const configured = spec.pythonPath.trim();
  const interpreters = configured ? [configured] : ['python3', 'python'];
  const bundled = path.join(extensionPath, 'python');

  // The module form goes first: on an installation too old to have the
  // exporter it fails immediately, whereas `slurm-top --json` there would start
  // the TUI and have to be timed out.
  const list: Collector[] = [];
  for (const python of interpreters) {
    list.push({
      argv: [python, '-m', 'slurm_top.export'],
      origin: `${python} -m slurm_top.export (installed package)`,
    });
  }
  list.push({ argv: ['slurm-top', '--json'], origin: 'slurm-top on PATH' });
  for (const python of interpreters) {
    list.push({
      argv: [python, '-m', 'slurm_top.export'],
      env: { PYTHONPATH: bundled },
      origin: `${python} with the copy bundled in the extension`,
    });
  }
  return list;
}

/** Why a candidate was passed over, in the words of the command itself. */
interface ProbeResult {
  ok: boolean;
  reason?: string;
}

/** The last few lines of whatever the command complained about. */
function lastLines(text: string, count = 3): string {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-count)
    .join('; ');
}

function probe(candidate: Collector, timeoutMs: number): Promise<ProbeResult> {
  return new Promise((resolve) => {
    const [command, ...args] = candidate.argv;
    let child;
    try {
      child = spawn(command, [...args, '--help'], {
        env: { ...process.env, ...(candidate.env ?? {}) },
      });
    } catch (err) {
      resolve({ ok: false, reason: String(err) });
      return;
    }
    let output = '';
    let errors = '';
    let settled = false;
    const finish = (result: ProbeResult) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (child.exitCode === null && child.signalCode === null) {
        child.kill();
      }
      resolve(result);
    };
    const timer = setTimeout(
      () => finish({ ok: false, reason: `no answer within ${Math.round(timeoutMs / 1000)}s` }),
      timeoutMs
    );
    child.on('error', (err) => finish({ ok: false, reason: String(err) }));
    child.on('close', (code) => {
      if (code === 0 && output.includes(HELP_MARKER)) {
        finish({ ok: true });
        return;
      }
      // What the command said is the whole diagnosis: `command not found`,
      // `Permission denied (publickey)`, a Python traceback. Dropping it -- as
      // this used to -- leaves "did not respond" and nothing to act on.
      const said = lastLines(errors) || lastLines(output);
      finish({
        ok: false,
        reason:
          code === 0
            ? `answered, but not with slurm-top JSON help${said ? `: ${said}` : ''}`
            : `exit ${code}${said ? `: ${said}` : ''}`,
      });
    });
    child.stdout?.setEncoding('utf8');
    // Keep only the head: enough to spot the marker, and an unexpected command
    // that streams forever cannot fill memory before the timeout fires.
    child.stdout?.on('data', (chunk: string) => {
      if (output.length < 8192) {
        output += chunk;
      }
    });
    child.stderr?.setEncoding('utf8');
    child.stderr?.on('data', (chunk: string) => {
      if (errors.length < 8192) {
        errors += chunk;
      }
    });
  });
}

/**
 * Pick the first candidate that answers `--help` successfully.
 *
 * `--help` rather than a real snapshot: it exits immediately even on a cluster
 * whose `squeue` is wedged, so a slow controller cannot look like a missing
 * install.
 */
export async function resolveCollector(
  spec: ServerSpec,
  extensionPath: string,
  log: vscode.OutputChannel
): Promise<Collector | undefined> {
  const list = candidates(spec, extensionPath);
  // ssh has to open a connection before anything runs, which takes longer than
  // starting a local interpreter ever would.
  const timeout = spec.command.length ? 25000 : 8000;
  for (const candidate of list) {
    const { ok, reason } = await probe(candidate, timeout);
    if (ok) {
      log.appendLine(`collector: using ${candidate.origin} -> ${shortArgv(candidate.argv)}`);
      return candidate;
    }
    log.appendLine(
      `collector: ${shortArgv(candidate.argv)} failed (${reason ?? 'no reason given'})` +
        (list.length > 1 ? ', trying next' : '')
    );
    lastFailure = reason;
  }
  return undefined;
}

/**
 * Why the most recent resolution failed.
 *
 * Module state rather than a return value so that `resolveCollector` keeps its
 * "the collector, or nothing" shape; the client reads this only to put a
 * reason in the message it shows.
 */
let lastFailure: string | undefined;

export function lastResolutionFailure(): string | undefined {
  return lastFailure;
}
