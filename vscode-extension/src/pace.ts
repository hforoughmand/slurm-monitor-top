import { sshDestination } from './servers';

/**
 * How often the extension reaches out to another machine.
 *
 * A cluster's login node sees every ssh connection and every `squeue` this
 * extension makes, and a burst of them -- a probe per way of running the
 * collector, a restart after a dropped session, a refresh every few seconds --
 * reads as a script hammering the node. So a remote cluster is contacted at
 * most once per `REMOTE_GAP_MS`: a new connection waits its turn, and the
 * collector streaming over an open one is not asked to refresh more often.
 * Collectors that run on this machine are not paced at all.
 */

/** The least time between two connections to one remote machine. */
export const REMOTE_GAP_MS = 30000;

/** The least time, in seconds, between two refreshes of a remote cluster. */
export const REMOTE_MIN_REFRESH_S = REMOTE_GAP_MS / 1000;

/** Per destination: the earliest moment the next connection may start. */
const nextSlot = new Map<string, number>();

/** Whether an argv reaches another machine, and so is paced. */
export function isRemote(argv: string[]): boolean {
  return sshDestination(argv) !== undefined;
}

/** Seconds between refreshes for a collector: the setting, floored for a remote one. */
export function refreshSeconds(argv: string[], configured: number): number {
  const seconds = Math.max(1, configured);
  return isRemote(argv) ? Math.max(seconds, REMOTE_MIN_REFRESH_S) : seconds;
}

/**
 * Wait until this argv may open a connection, and take that turn.
 *
 * The slot is reserved before waiting, so callers that arrive together queue
 * up one gap apart rather than all firing when the first gap ends. A caller
 * that gives up while waiting still uses its slot: the cost is a later start,
 * never a closer one.
 */
export async function paceRemote(argv: string[], note?: (line: string) => void): Promise<void> {
  const host = sshDestination(argv);
  if (!host) {
    return;
  }
  const now = Date.now();
  const slot = Math.max(now, nextSlot.get(host) ?? 0);
  nextSlot.set(host, slot + REMOTE_GAP_MS);
  const wait = slot - now;
  if (wait > 0) {
    note?.(`waiting ${Math.ceil(wait / 1000)}s before contacting ${host} again (calls are ${REMOTE_MIN_REFRESH_S}s apart)`);
    await new Promise((resolve) => setTimeout(resolve, wait));
  }
}
