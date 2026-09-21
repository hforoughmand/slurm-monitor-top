import { ChildProcessWithoutNullStreams, execFile, spawn } from 'child_process';
import * as vscode from 'vscode';

import { shortArgv } from './remote';
import { Collector, lastResolutionFailure, resolveCollector } from './resolve';
import { ServerSpec } from './servers';
import { JobDetail, JobOutput, NodeDetail, Snapshot, SUPPORTED_SCHEMA } from './types';

export type ClientState = 'idle' | 'starting' | 'running' | 'error';

/**
 * Owns one long-lived `slurm-top --json --watch N` process and turns its
 * newline-delimited output into snapshot events.
 *
 * A streaming child rather than one spawn per refresh: on a login node an
 * interpreter start costs more than the squeue call itself, and a 3-second
 * poll that pays it every tick is noticeable to everyone else on the node.
 *
 * One of these per watched server. It knows nothing about the others: the
 * merging, and the routing of a click back to the right one, belong to
 * `ClusterClient` above it.
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
    readonly spec: ServerSpec,
    private readonly extensionPath: string,
    private readonly log: vscode.OutputChannel
  ) {}

  get id(): string {
    return this.spec.id;
  }

  /** What to call this server in a message: its label, else its hostname. */
  get label(): string {
    return this.spec.name || this.lastSnapshot?.host || this.spec.id;
  }

  /** Take a new label from the settings without restarting the collector. */
  rename(name: string): void {
    this.spec.name = name;
  }

  /** Prefix log lines with the server, so one channel can carry several. */
  private note(line: string): void {
    this.log.appendLine(`[${this.label}] ${line}`);
  }

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
      this.collector = await resolveCollector(this.spec, this.extensionPath, this.log);
      if (!this.collector) {
        const why = lastResolutionFailure();
        this.setState(
          'error',
          this.spec.command.length
            ? `${this.label}: \`${shortArgv(this.spec.command)}\` did not answer` +
              `${why ? ` — ${why}` : ''}. Over ssh, slurm-monitor-top has to be installed there and on ` +
              'the PATH of a non-interactive login.'
            : 'Could not find a way to run slurm-top. Install it with `pip install slurm-monitor-top`, ' +
              `set slurmTop.pythonPath, or set slurmTop.command.${why ? ` (${why})` : ''}`
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
    this.note(`spawn: ${shortArgv([command, ...argv])}`);

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
    child.stderr.on('data', (chunk: string) => this.note(`collector stderr: ${chunk.trim()}`));
    child.on('error', (err) => {
      this.note(`collector error: ${String(err)}`);
      this.setState('error', String(err));
    });
    child.on('close', (code, signal) => {
      this.child = undefined;
      this.note(`collector exited (code=${code} signal=${signal})`);
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
      this.note('stopping collector');
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
      this.note('collector: discarding oversized partial line');
      this.buffer = '';
    }
  }

  private handleLine(line: string): void {
    let parsed: Snapshot;
    try {
      parsed = JSON.parse(line) as Snapshot;
    } catch (err) {
      this.note(`collector: unparseable line (${String(err)}): ${line.slice(0, 200)}`);
      return;
    }
    if (parsed.error) {
      this.setState('error', parsed.error);
      return;
    }
    if (parsed.schema !== SUPPORTED_SCHEMA && !this.schemaWarned) {
      this.schemaWarned = true;
      this.note(
        `collector: schema ${parsed.schema} but this extension expects ${SUPPORTED_SCHEMA}; ` +
          'some fields may be missing. Update slurm-monitor-top or the extension.'
      );
    }
    this.restartDelay = 2000;
    this.lastSnapshot = parsed;
    this.setState('running');
    this.snapshotEmitter.fire(parsed);
  }

  /**
   * Run the collector once and parse its single JSON object.
   *
   * Separate from the streaming child: these are the requests that answer a
   * click (open this node, pin this job) and must not wait for the next tick.
   */
  private async runOnce(extra: string[], timeout = 30000): Promise<unknown> {
    if (!this.collector) {
      this.collector = await resolveCollector(this.spec, this.extensionPath, this.log);
    }
    const collector = this.collector;
    if (!collector) {
      throw new Error(`No slurm-top collector available for ${this.label}.`);
    }
    const [command, ...args] = collector.argv;
    const argv = [...args, ...extra];
    const stdout = await new Promise<string>((resolve, reject) => {
      execFile(
        command,
        argv,
        { env: { ...process.env, ...(collector.env ?? {}) }, timeout, maxBuffer: 16 * 1024 * 1024 },
        (err, out) => (err ? reject(err) : resolve(out))
      );
    });
    return JSON.parse(stdout);
  }

  /** One-shot detail lookup for the job/node popups. */
  async fetchDetail(
    kind: 'job' | 'node',
    id: string,
    options: { probeCpu?: boolean } = {}
  ): Promise<JobDetail | NodeDetail> {
    const extra = [kind === 'job' ? '--job' : '--node', id];
    if (kind === 'node' && options.probeCpu) {
      extra.push('--probe-cpu');
      if (!vscode.workspace.getConfiguration('slurmTop').get<boolean>('cpuProbeUsesSrun', true)) {
        extra.push('--no-srun');
      }
    }
    // A probe may sit in the queue for its one-second job, so it gets longer
    // than an ordinary lookup before we give up on it.
    return (await this.runOnce(extra, options.probeCpu ? 90000 : 30000)) as JobDetail | NodeDetail;
  }

  /**
   * Read the tail of one of a job's output files.
   *
   * Goes through the collector rather than through `vscode.workspace.fs` so it
   * still works when the collector is a remote command: whoever can run
   * `squeue` can read the file, and that is not always this machine.
   */
  async fetchJobOutput(
    jobId: string,
    stream: 'stdout' | 'stderr',
    lines: number
  ): Promise<JobOutput> {
    return (await this.runOnce(
      ['--job-output', jobId, '--stream', stream, '--lines', String(lines)],
      30000
    )) as JobOutput;
  }

  /**
   * Flip a job's pin and return the new list.
   *
   * The pins live in the collector's config file rather than in the editor, so
   * the terminal UI shows the same jobs on top.
   */
  async togglePin(jobId: string): Promise<string[]> {
    const result = (await this.runOnce(['--toggle-pin', jobId], 15000)) as { pinned?: string[] };
    return result.pinned ?? [];
  }

  dispose(): void {
    this.disposed = true;
    this.stop();
    this.snapshotEmitter.dispose();
    this.stateEmitter.dispose();
  }
}
