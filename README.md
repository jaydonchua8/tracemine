# tracemine

Mine coding agent traces for repeated workflows and for failures that keep getting fixed the same way, and print the findings as a diff against a skill file.

Coding agents leave long session logs that nobody reads. Two things in them are worth pulling out. One is a sequence of steps that many sessions repeat, which is a candidate for a skill. The other is a failure that many sessions hit and then get past in the same way, which is a line that should already be in a skill. I read an internship posting that described this problem (measuring skill invocations, mining agent traces for failures and repeated workflows, and proposing reviewable edits back to a team's skills) and wanted to see how far a small, plain version could get.

It is one file, about 390 lines of Python, with no dependencies.

## Results

`python bench.py` generates synthetic traces: 400 sessions per seed, 5 seeds, 73,468 tool calls in total. Each session is background noise drawn from 36 kinds of call with a skewed frequency, and I plant 6 workflows and 5 failure/fix habits in a fraction of sessions (5% to 15% each), plus two recurring failures that have no consistent fix and one-off failures on 2% of shell calls.

| workflow mining | planted | recovered | reported | precision | partial overlaps |
|---|---|---|---|---|---|
| support only | 30 | 30 | 666 | 4.5% | 15 |
| with chance test | 30 | 30 | 34 | 88.2% | 0 |

| failure rules | planted | recovered | reported | wrong rules |
|---|---|---|---|---|
| | 25 | 25 | 25 | 0 |

"Support only" is the baseline: report every sequence that shows up in at least 2% of sessions. It finds everything and buries it in 636 other sequences. The chance test described below brings that down to 34 reported, of which 30 are the planted ones.

The 4 unplanted workflows are all three-step sequences made of the most common background steps (`Read:.py`, `Edit:.py`, `git status`, `Grep`), for example `Read:.py -> Edit:.py -> Read:.py`. Some of that is a side effect of a planted failure habit (a failed edit, a read, then a successful edit), so they are not pure noise, but they are not useful as skills either.

The two failures planted with no consistent fix were never turned into rules. They are listed separately as recurring failures without a fix, along with the one-off failures that happened to recur.

Mining took 3.6 seconds for the 73,468 calls on the last run I did, about 20,000 calls per second. It varied between about 17,000 and 22,000 across runs on the same machine.

## How it works

**Normalising.** Each tool call becomes a short key with the arguments removed: `cd /repo && git commit -m "x"` becomes `sh:git commit`, an edit to `src/app.py` becomes `Edit:.py`. Runs of the same key are collapsed to one.

**Workflows.** I count, for every contiguous sequence of 3 to 6 keys, how many sessions contain it. A sequence is kept only if both of its ends are surprising. For the last step: given how many sessions contain the rest of the sequence and how common that step is overall, I compute the Poisson probability of seeing it follow this often by chance, and require it to be under 0.0001. The same test is applied to the first step. This is what rejects "real workflow plus whatever common step happened to come next". After that, a sequence is dropped if a longer kept sequence contains it and has at least 70% of its support. Sequences that include a failed call are skipped, because the failure miner covers those.

**Failures.** Failed calls are grouped by key and by a normalised error line (quoted strings, paths and numbers masked, and signatures that differ in one word merged). For each failure I look at the next 4 calls for a successful call with the same intent: the same file for file tools, and for shell commands the same key, the same arguments, or mostly the same tokens. The fix is described either as a change to the command (`prefix the command with python -m`, `replace python with python3`) or as a step that was run in between (`first run npm install, then retry`). A token in the fix that appears in the error message is written as `<from-error>`, so `git push -u origin fix-auth` and `git push -u origin bump-deps` count as the same fix. A rule is reported only if one fix follows at least 60% of all occurrences.

**Skill usage.** For each skill invocation it counts invocations and sessions, and the tool error rate in the next 10 calls next to the overall error rate.

**Proposal.** `propose()` returns a unified diff that appends the top rules and workflows to a skill file, each with its session count and example session ids. It never writes the file. A person reads the diff and decides.

```
python bench.py --dump traces.jsonl
python tracemine.py report traces.jsonl --skill SKILL.md
```

prints lines such as

```
+- When `sh:git push` fails with `fatal: The current branch <*> has no upstream branch.`: add `-u origin <from-error>`. Seen in 22 sessions, this fix followed 95% of the time (e.g. ...).
```

## What I tried that did not work

My first filter was a plain lift ratio (observed support divided by expected, at least 3). It left 63 reported workflows for the 30 planted, a precision of 47.6%, because a sequence seen 8 or 9 times with 3 expected passes a ratio test fairly often when thousands of sequences are tested. Switching to the Poisson probability and leaving failed steps out of workflow mining got it to 34.

My first version of fix matching used only token overlap between the failed and the fixed command. It recovered 10 of the 25 planted rules. It missed `python` to `python3` (overlap too low) and `git push` to `git push -u origin <branch>` (the branch name in the error split the failures into separate groups). Matching on key or arguments and merging error signatures fixed both.

## Limits

- All numbers are from synthetic traces that I generated, and I tuned the thresholds on the same generator. Real sessions will be messier, and I expect lower precision there.
- It is not integrated with any product. The loader accepts a flat JSONL format and a transcript shape with `tool_use` and `tool_result` blocks, which I only tested on a hand-written fixture (`sample_session.jsonl`), not on real exported sessions.
- Workflows are contiguous sequences only. A workflow with a different unrelated step in the middle each time is missed.
- The command normaliser is simple. `git -C dir status` becomes `sh:git dir`, and only the first command of a pipeline or chain is looked at.
- The skill usage numbers are counts and a raw error rate. They do not show whether a skill caused anything.
- The proposed edit is appended text. It does not rewrite existing skill instructions or check them for conflicts.

## Run it

```
python test_tracemine.py
python bench.py
python tracemine.py report sample_session.jsonl
```

Python 3.8 or newer. Nothing to install.
