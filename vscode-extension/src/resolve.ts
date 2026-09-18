import { spawn } from 'child_process';
import * as path from 'path';
import * as vscode from 'vscode';

/** An argv prefix that prints slurm-top JSON when data-mode flags are appended. */
export interface Collector {
  argv: string[];
  /** Extra environment the command needs (PYTHONPATH for the bundled copy). */
  env?: Record<string, string>;
  /** Human-readable origin, for the log and error messages. */
  origin: string;
}

function config<T>(key: string, fallback: T): T {
  return vscode.workspace.getConfiguration('slurmTop').get<T>(key, fallback);
}

/**
 * A string that only the JSON exporter's `--help` prints. An older
 * slurm-monitor-top ignores `--json` and starts the TUI instead, so a probe
 * that trusted the exit code alone could adopt a command that never emits JSON.
 */
const HELP_MARKER = '--watch';

/** Candidates in priority order: explicit override, installed CLI, bundled copy. */
function candidates(extensionPath: string): Collector[] {
  const override = config<string[]>('command', []);
  if (override.length > 0) {
    return [{ argv: override, origin: 'slurmTop.command setting' }];
  }

  const configured = config<string>('pythonPath', '').trim();
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

function probe(candidate: Collector, timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    const [command, ...args] = candidate.argv;
    let child;
    try {
      child = spawn(command, [...args, '--help'], {
        env: { ...process.env, ...(candidate.env ?? {}) },
      });
    } catch {
      resolve(false);
      return;
    }
    let output = '';
    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      if (child.exitCode === null && child.signalCode === null) {
        child.kill();
      }
      resolve(ok);
    };
    const timer = setTimeout(() => finish(false), timeoutMs);
    child.on('error', () => finish(false));
    child.on('close', (code) => finish(code === 0 && output.includes(HELP_MARKER)));
    child.stdout?.setEncoding('utf8');
    // Keep only the head: enough to spot the marker, and an unexpected command
    // that streams forever cannot fill memory before the timeout fires.
    child.stdout?.on('data', (chunk: string) => {
      if (output.length < 8192) {
        output += chunk;
      }
    });
    child.stderr?.resume();
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
  extensionPath: string,
  log: vscode.OutputChannel
): Promise<Collector | undefined> {
  for (const candidate of candidates(extensionPath)) {
    if (await probe(candidate, 8000)) {
      log.appendLine(`collector: using ${candidate.origin} -> ${candidate.argv.join(' ')}`);
      return candidate;
    }
    log.appendLine(`collector: ${candidate.origin} did not respond, trying next`);
  }
  return undefined;
}
