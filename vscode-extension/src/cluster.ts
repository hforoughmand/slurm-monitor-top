import * as vscode from 'vscode';

import { SlurmClient } from './client';
import { ServerEntry, aggregateState, mergeSnapshots } from './merge';
import { ServerSpec, readServers, specSignature } from './servers';
import { DetailTarget, JobDetail, JobOutput, MergedSnapshot, NodeDetail, ServerState } from './types';

/**
 * Every watched cluster behind one client.
 *
 * The views never talk to a collector directly: they get one merged snapshot
 * and send back requests tagged with a server id, and this routes each one to
 * the process that can answer it. That keeps "which cluster is this row from"
 * in exactly one place -- everywhere else a row simply carries its `server`.
 */
export class ClusterClient implements vscode.Disposable {
  private readonly snapshotEmitter = new vscode.EventEmitter<MergedSnapshot>();
  private readonly stateEmitter = new vscode.EventEmitter<{ state: ServerState; message?: string }>();

  readonly onSnapshot = this.snapshotEmitter.event;
  readonly onState = this.stateEmitter.event;

  private clients: SlurmClient[] = [];
  private readonly listeners = new Map<string, vscode.Disposable[]>();
  private readonly states = new Map<string, { state: ServerState; message?: string }>();
  private merged?: MergedSnapshot;
  private wanted = false;
  private disposed = false;

  constructor(
    private readonly extensionPath: string,
    private readonly log: vscode.OutputChannel
  ) {
    this.clients = readServers().map((spec) => this.create(spec));
  }

  get servers(): ServerSpec[] {
    return this.clients.map((client) => client.spec);
  }

  get latest(): MergedSnapshot | undefined {
    return this.merged;
  }

  get currentState(): { state: ServerState; message?: string } {
    return aggregateState(this.entries());
  }

  /** The label to put on a window title or a status line. */
  nameOf(server?: string): string {
    const client = this.find(server);
    return client ? client.label : String(server ?? '');
  }

  /** The pins the collector on that server currently holds. */
  pinnedFor(server?: string): string[] {
    return this.find(server)?.latest?.pinned ?? [];
  }

  private entries(): ServerEntry[] {
    return this.clients.map((client) => {
      const state = this.states.get(client.id) ?? { state: 'starting' as ServerState };
      return {
        id: client.id,
        name: client.spec.name,
        state: state.state,
        message: state.message,
        snapshot: client.latest,
      };
    });
  }

  /**
   * The client for a request.
   *
   * An unknown or missing id falls back to the only server there is, which is
   * what makes a message from an older view -- or from a detail window opened
   * before a server was renamed -- still land somewhere sensible.
   */
  private find(server?: string): SlurmClient | undefined {
    if (server) {
      const match = this.clients.find((client) => client.id === server);
      if (match) {
        return match;
      }
    }
    return this.clients.length === 1 ? this.clients[0] : undefined;
  }

  private require(server?: string): SlurmClient {
    const client = this.find(server);
    if (!client) {
      throw new Error(`No server called ${server || '(unnamed)'} is being watched.`);
    }
    return client;
  }

  /** Make a client for one server and subscribe to it. */
  private create(spec: ServerSpec): SlurmClient {
    const client = new SlurmClient(spec, this.extensionPath, this.log);
    this.states.set(spec.id, { state: 'paused' });
    this.listeners.set(spec.id, [
      client.onSnapshot(() => this.republish()),
      client.onState(({ state, message }) => {
        this.states.set(spec.id, { state: state === 'idle' ? 'paused' : state, message });
        // A server going quiet or coming back changes what the boxes say, so
        // the snapshot is re-sent as well as the status line.
        this.republish();
      }),
    ]);
    return client;
  }

  private drop(client: SlurmClient): void {
    for (const listener of this.listeners.get(client.id) ?? []) {
      listener.dispose();
    }
    this.listeners.delete(client.id);
    this.states.delete(client.id);
    client.dispose();
  }

  private republish(): void {
    if (this.disposed) {
      return;
    }
    const entries = this.entries();
    this.merged = mergeSnapshots(entries);
    this.snapshotEmitter.fire(this.merged);
    const { state, message } = aggregateState(entries);
    this.stateEmitter.fire({ state, message });
  }

  /**
   * Rebuild the client set from the settings.
   *
   * A server whose command is unchanged keeps its process: editing one entry in
   * a list of four must not restart the other three and blank their panels. A
   * renamed one keeps it too -- the label is ours, not the collector's.
   *
   * Returns false when nothing changed, so the caller can fall back to an
   * ordinary restart.
   */
  async reload(): Promise<boolean> {
    const specs = readServers();
    const shape = (list: ServerSpec[]) =>
      list.map((spec) => `${specSignature(spec)}::${spec.name}`).join('|');
    if (shape(this.servers) === shape(specs)) {
      return false;
    }

    const previous = this.clients;
    const reused = new Set<SlurmClient>();
    const next: SlurmClient[] = [];
    for (const spec of specs) {
      const match = previous.find(
        (client) => !reused.has(client) && specSignature(client.spec) === specSignature(spec)
      );
      if (match) {
        reused.add(match);
        match.rename(spec.name);
        next.push(match);
      } else {
        next.push(this.create(spec));
      }
    }
    for (const client of previous) {
      if (!reused.has(client)) {
        this.drop(client);
      }
    }
    this.clients = next;

    this.log.appendLine(
      `servers: watching ${this.clients.map((client) => client.label).join(', ') || '(none)'}`
    );
    if (this.wanted) {
      await Promise.all(this.clients.map((client) => client.start()));
    }
    this.republish();
    return true;
  }

  async start(): Promise<void> {
    this.wanted = true;
    await Promise.all(this.clients.map((client) => client.start()));
  }

  stop(): void {
    this.wanted = false;
    for (const client of this.clients) {
      client.stop();
    }
  }

  /** Re-resolve every collector; what the refresh command does. */
  async restart(): Promise<void> {
    if (await this.reload()) {
      return;
    }
    await Promise.all(this.clients.map((client) => client.restart()));
  }

  async fetchDetail(
    target: DetailTarget,
    options: { probeCpu?: boolean } = {}
  ): Promise<JobDetail | NodeDetail> {
    return this.require(target.server).fetchDetail(target.kind, target.id, options);
  }

  async fetchJobOutput(
    server: string | undefined,
    jobId: string,
    stream: 'stdout' | 'stderr',
    lines: number
  ): Promise<JobOutput> {
    return this.require(server).fetchJobOutput(jobId, stream, lines);
  }

  async togglePin(server: string | undefined, jobId: string): Promise<string[]> {
    return this.require(server).togglePin(jobId);
  }

  dispose(): void {
    this.disposed = true;
    for (const client of this.clients.slice()) {
      this.drop(client);
    }
    this.clients = [];
    this.snapshotEmitter.dispose();
    this.stateEmitter.dispose();
  }
}
