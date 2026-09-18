import { ChildProcessWithoutNullStreams, execFile, spawn } from 'child_process';
import * as vscode from 'vscode';

import { Collector, resolveCollector } from './resolve';
import { JobDetail, NodeDetail, Snapshot, SUPPORTED_SCHEMA } from './types';

export type ClientState = 'idle' | 'starting' | 'running' | 'error';

/**
 * Owns one long-lived `slurm-top --json --watch N` process and turns its
 * newline-delimited output into snapshot events.
 *
 * A streaming child rather than one spawn per refresh: on a login node an
 * interpreter start costs more than the squeue call itself, and a 3-second
 * poll that pays it every tick is noticeable to everyone else on the node.
 */
export class SlurmClient implements vscode.Disposable {
  private readonly snapshotEmitter = new vscode.EventEmitter<Snapshot>();
  private readonly stateEmitter = new vscode.EventEmitter<{ state: ClientState; message?: string }>();

  readonly onSnapshot = this.snapshotEmitter.event;
  readonly onState = this.stateEmitter.event;

  private child?: ChildProcessWithoutNullStreams;
  private collector?: Collector;
  private buffer = '';
  private restartTimer?: NodeJS.Timeout;
  private restartDelay = 2000;
  private disposed = false;
  private wanted = false;
  private state: ClientState = 'idle';
  private lastSnapshot?: Snapshot;
  private schemaWarned = false;

  constructor(
    private readonly extensionPath: string,
    private readonly log: vscode.OutputChannel
  ) {}

  get latest(): Snapshot | undefined {
    return this.lastSnapshot;
  }

  get currentState(): { state: ClientState; message?: string } {
    return { state: this.state, message: this.lastError };
  }

  private lastError?: string;

  private setState(state: ClientState, message?: string): void {
    this.state = state;
    this.lastError = message;
    this.stateEmitter.fire({ state, message });
  }

  /** Start streaming (idempotent). */
  async start(): Promise<void> {
    this.wanted = true;
    if (this.child || this.disposed) {
      return;
    }
    this.setState('starting');

    if (!this.collector) {
      this.collector = await resolveCollector(this.extensionPath, this.log);
      if (!this.collector) {
        this.setState(
          'error',
          'Could not find a way to run slurm-top. Install it with `pip install slurm-monitor-top`, ' +
            'set slurmTop.pythonPath, or set slurmTop.command.'
        );
        return;
      }
    }
    if (!this.wanted || this.disposed) {
      return;
    }

    const interval = vscode.workspace.getConfiguration('slurmTop').get<number>('refreshInterval', 3);
    const [command, ...args] = this.collector.argv;
    const argv = [...args, '--watch', String(interval)];
    this.log.appendLine(`spawn: ${command} ${argv.join(' ')}`);

    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawn(command, argv, {
        env: { ...process.env, ...(this.collector.env ?? {}) },
      });
    } catch (err) {
      this.setState('error', `Failed to start collector: ${String(err)}`);
      return;
    }

    this.child = child;
    this.buffer = '';
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => this.consume(chunk));
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk: string) => this.log.appendLine(`collector stderr: ${chunk.trim()}`));
    child.on('error', (err) => {
      this.log.appendLine(`collector error: ${String(err)}`);
      this.setState('error', String(err));
    });
    child.on('close', (code, signal) => {
      this.child = undefined;
      this.log.appendLine(`collector exited (code=${code} signal=${signal})`);
      if (this.disposed || !this.wanted) {
        return;
      }
      this.setState('error', `Collector exited (code ${code ?? signal}). Retrying...`);
      this.scheduleRestart();
    });
  }

  /** Stop the child but keep the resolved command for the next start. */
  stop(): void {
    this.wanted = false;
    this.clearRestart();
    if (this.child) {
      this.log.appendLine('stopping collector');
      this.child.kill();
      this.child = undefined;
    }
    if (this.state !== 'error') {
      this.setState('idle');
    }
  }

  /** Drop the cached command too, so a config change is picked up. */
  async restart(): Promise<void> {
    this.collector = undefined;
    this.restartDelay = 2000;
    const wasWanted = this.wanted || this.child !== undefined;
    this.stop();
    this.setState('idle');
    if (wasWanted) {
      await this.start();
    }
  }

  private scheduleRestart(): void {
    this.clearRestart();
    const delay = this.restartDelay;
    // Back off so a permanently broken command does not respawn in a tight loop.
    this.restartDelay = Math.min(this.restartDelay * 2, 60000);
    this.restartTimer = setTimeout(() => {
      this.restartTimer = undefined;
      void this.start();
    }, delay);
  }

  private clearRestart(): void {
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = undefined;
    }
  }

  private consume(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf('\n');
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      newline = this.buffer.indexOf('\n');
      if (line) {
        this.handleLine(line);
      }
    }
    // A collector that somehow stops emitting newlines must not grow the buffer
    // without bound.
    if (this.buffer.length > 8 * 1024 * 1024) {
      this.log.appendLine('collector: discarding oversized partial line');
      this.buffer = '';
    }
  }

  private handleLine(line: string): void {
    let parsed: Snapshot;
    try {
      parsed = JSON.parse(line) as Snapshot;
    } catch (err) {
      this.log.appendLine(`collector: unparseable line (${String(err)}): ${line.slice(0, 200)}`);
      return;
    }
    if (parsed.error) {
      this.setState('error', parsed.error);
      return;
    }
    if (parsed.schema !== SUPPORTED_SCHEMA && !this.schemaWarned) {
      this.schemaWarned = true;
      this.log.appendLine(
        `collector: schema ${parsed.schema} but this extension expects ${SUPPORTED_SCHEMA}; ` +
          'some fields may be missing. Update slurm-monitor-top or the extension.'
      );
    }
    this.restartDelay = 2000;
    this.lastSnapshot = parsed;
    this.setState('running');
    this.snapshotEmitter.fire(parsed);
  }

  /** One-shot detail lookup for the job/node popups. */
  async fetchDetail(kind: 'job' | 'node', id: string): Promise<JobDetail | NodeDetail> {
    if (!this.collector) {
      this.collector = await resolveCollector(this.extensionPath, this.log);
    }
    const collector = this.collector;
    if (!collector) {
      throw new Error('No slurm-top collector available.');
    }
    const [command, ...args] = collector.argv;
    const argv = [...args, kind === 'job' ? '--job' : '--node', id];
    const stdout = await new Promise<string>((resolve, reject) => {
      execFile(
        command,
        argv,
        { env: { ...process.env, ...(collector.env ?? {}) }, timeout: 30000, maxBuffer: 16 * 1024 * 1024 },
        (err, out) => (err ? reject(err) : resolve(out))
      );
    });
    return JSON.parse(stdout) as JobDetail | NodeDetail;
  }

  dispose(): void {
    this.disposed = true;
    this.stop();
    this.snapshotEmitter.dispose();
    this.stateEmitter.dispose();
  }
}
