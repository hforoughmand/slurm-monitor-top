# Screenshot tooling

Every image in the project READMEs is generated from this directory, against a
cluster that does not exist. Two reasons: a screenshot of a real cluster leaks
user names, project names and paths, and a hand-made screenshot goes stale the
moment a panel changes.

## Files

| File | What it is |
| --- | --- |
| `fake_cluster.py` | The synthetic cluster, plus fake `squeue`, `sinfo`, `scontrol`, `sstat`, `df`, `ssh` and `srun` commands that serve it |
| `capture_tui.py` | Drives the Textual app headlessly, one scene per feature, and writes `assets/tui-*.png` |
| `capture_webview.py` | Renders the VS Code extension's webview in headless Chrome and writes `vscode-extension/assets/ext-*.png` |

## Regenerating

```bash
python tools/screenshots/capture_tui.py                  # every TUI scene
python tools/screenshots/capture_tui.py overview          # just one
python tools/screenshots/capture_webview.py               # every extension scene
python tools/screenshots/capture_webview.py ext-sidebar   # just one
```

By default the TUI scenes go to `assets/` and the extension scenes to
`vscode-extension/assets/` — beside the README that links to them, so both
READMEs can use relative paths and `vsce` can rewrite those links to a URL that
resolves (it needs the `--baseContentUrl` in the extension's `package` script to
do so). `--out DIR` sends them elsewhere.
`capture_tui.py` also takes `--svg-only`, which stops after Textual's SVG export
and skips the browser.

Run `capture_tui.py` with an interpreter that has `textual` installed (the
project's virtualenv). `capture_webview.py` only needs the standard library,
because the collector it calls does.

Rendering a whole run at once is memory-hungry (a headless Chrome plus an
ImageMagick pass per scene); if the process gets killed, capture the scenes in
smaller batches.

## Requirements

- `textual` — for `capture_tui.py` only
- A Chrome or Chromium binary — the SVG and the webview are both rendered by it
- ImageMagick's `convert` — trims the browser's margin and reduces each PNG to a
  256-colour palette, which is visually identical for a terminal frame and about
  a third of the bytes. Without it the images are still written, just larger.

## Scenes

TUI (`assets/tui-<name>.png`):

| Scene | Shows |
| --- | --- |
| `overview` | The default four-panel layout |
| `job-search` | The `/` search popup mid-typing |
| `search-applied` | The jobs panel with the search in its border |
| `job-details` | `Enter` on a job: scontrol fields, usage bars, actions |
| `job-usage` | The asked-for-versus-used bars on a job wasting its cores |
| `job-output` | `o` on a job: stdout and stderr tailed side by side |
| `node-details` | `Enter` on a machine, and the jobs placed on it |
| `gpu-jobs` | `Enter` on a GPU type: jobs using and reserving it |
| `copy-popup` | `c`: the selected row, field by field |
| `sort-picker` | `s`: the sort column picker |
| `owner-filter` | `f`: the view narrowed to your own jobs |
| `job-pin` | `p`: pinned jobs held at the top of the table |
| `node-cpu` | `p` in the machine popup: the CPU model read over (fake) ssh |
| `panel-resize` | `Alt+→`: the machine panel with all its columns |

Extension (`vscode-extension/assets/ext-<name>.png`):

| Scene | Shows |
| --- | --- |
| `ext-sidebar` | The compact activity-bar view |
| `ext-dashboard` | The full editor tab |
| `ext-job-detail` | Job details as their own view |
| `ext-node-detail` | Machine details as an overlay on the dashboard |

## How they work

`capture_tui.py` installs `fake_cluster.py` into a temporary directory as one
symlink per command `slurm_top.data` shells out to, puts that directory first on
`PATH`, and runs the real `SlurmHtop` app under Textual's own test pilot. Each
scene presses the keys a user would and exports the frame as SVG, which headless
Chrome turns into a PNG. Every scene also gets a `HOME` of its own, because the
app persists pins and CPU readings there and one scene's state must not show up
in the next one's picture.

`ssh` and `srun` are stubbed for a sharper reason than tidiness: the CPU probe
behind `p` really does open an ssh connection, and really does submit a job, to
whatever answers to the name of the node. Without the stubs a capture would
reach out to a stranger's machine, or queue work on the actual cluster.

One wrinkle worth knowing about: Rich's SVG export writes one clip path per
terminal row but stops one row short, while still pointing the last row's text
at the missing one. A clip path that resolves to nothing means "draw nothing",
so the footer disappears. `patch_svg()` synthesises the missing clip paths and
grows the viewBox to match; without it every screenshot loses its keybinding
bar.

`capture_webview.py` needs no editor. The webview is plain HTML driven by
messages from the extension host, so the script builds a page around the real
`media/main.css` and `media/main.js`, stubs `acquireVsCodeApi`, supplies VS
Code's Dark Modern theme variables, and posts the same `config`, `snapshot` and
`detail` messages the host would — with the snapshot coming from
`slurm-top --json` run against the same fake cluster. What it cannot show is VS
Code itself: the activity bar, tab strip and title bar belong to the editor, not
the extension, and are not faked.

## Changing the cluster

Edit the `JOBS`, `NODES` and `DISKS` tables at the top of `fake_cluster.py` and
re-run the capture scripts. Keep the invented names invented: the point of this
directory is that no screenshot ever has to be blurred.
