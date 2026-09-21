import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';

/**
 * Running the collector on a machine that does not have it.
 *
 * The data layer is pure standard library and the extension already carries a
 * copy of it, so a cluster reachable by ssh does not have to have
 * slurm-monitor-top installed: the two modules are compressed, handed to the
 * remote `python3` as an argument, rebuilt in memory there and run. Nothing is
 * written to that machine and nothing is left behind.
 *
 * It is the last thing tried, not the first -- an installed collector needs no
 * payload at all -- but it is what makes "watch that cluster too" a matter of
 * typing its hostname.
 */

/** The modules the exporter needs; `__init__` only carries a version string. */
const MODULES = ['data', 'export'];

/**
 * What runs on the far side: rebuild `slurm_top` from the blob in argv[1], then
 * hand argv[2:] to the exporter exactly as an installed one would get them.
 *
 * `exec` into synthetic modules rather than a temporary file or a zipimport:
 * `slurm_top.export` does `from .data import ...`, which is satisfied by having
 * both in `sys.modules` first, and nothing touches the remote filesystem.
 */
const BOOTSTRAP = [
  'import sys,base64,zlib,json,types',
  'src=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])).decode("utf-8"))',
  'pkg=types.ModuleType("slurm_top");pkg.__path__=[];sys.modules["slurm_top"]=pkg',
  'for name in ("data","export"):',
  '    mod=types.ModuleType("slurm_top."+name);mod.__package__="slurm_top"',
  '    sys.modules["slurm_top."+name]=mod;setattr(pkg,name,mod)',
  '    exec(compile(src[name],"slurm_top/"+name+".py","exec"),mod.__dict__)',
  'sys.exit(sys.modules["slurm_top.export"].main(sys.argv[2:]))',
].join('\n');

/**
 * Quote one word for the shell on the other end of ssh.
 *
 * ssh joins its arguments with spaces and hands the result to a shell there, so
 * anything with a space or a quote in it has to arrive already quoted.
 */
export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

let payload: string | null | undefined;

/** The collector, compressed and base64'd; null when the copy is not bundled. */
function bundledPayload(extensionPath: string): string | null {
  if (payload !== undefined) {
    return payload;
  }
  try {
    const source: Record<string, string> = {};
    for (const name of MODULES) {
      source[name] = fs.readFileSync(path.join(extensionPath, 'python', 'slurm_top', `${name}.py`), 'utf8');
    }
    payload = zlib.deflateSync(Buffer.from(JSON.stringify(source), 'utf8'), { level: 9 }).toString('base64');
  } catch {
    payload = null;
  }
  return payload;
}

/** How big the pushed copy is, for the log. */
export function payloadSize(extensionPath: string): number {
  return bundledPayload(extensionPath)?.length ?? 0;
}

/**
 * `ssh … python3 -c <bootstrap> <blob>`, or undefined with no bundled copy.
 *
 * `python` as well as `python3`, because a cluster old enough to have only the
 * one is exactly the kind that will not have the package either.
 */
export function pushedCollector(
  sshPrefix: string[],
  extensionPath: string,
  interpreter: string
): string[] | undefined {
  const blob = bundledPayload(extensionPath);
  if (!blob) {
    return undefined;
  }
  return [...sshPrefix, interpreter, '-c', shellQuote(BOOTSTRAP), shellQuote(blob)];
}

/**
 * An argv fit for a log line.
 *
 * The pushed collector carries the whole data layer in one argument; printing
 * it would bury every other line in the channel under 26KB of base64.
 */
export function shortArgv(argv: string[]): string {
  return argv
    .map((arg) => (arg.length > 60 ? `<${arg.length} bytes>` : arg))
    .join(' ');
}
