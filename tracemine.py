"""tracemine: mine coding-agent traces for repeated workflows and failure/fix pairs,
and turn them into a reviewable diff against a skill file.

Standard library only. See README.md.
"""
import difflib
import json
import math
import os
import re
import shlex
import sys
from collections import Counter, defaultdict

import formats

# ---------------------------------------------------------------- loading

def load(lines):
    """Return {session_id: [call, ...]} from JSONL lines.

    Claude Code transcripts, Codex rollouts and the flat benchmark shape are all accepted,
    see formats.py. A call is a dict with keys tool, input, ok, error (plus done, agent).
    """
    return formats.calls_only(formats.parse(lines))

# ---------------------------------------------------------------- normalising

SUBCOMMANDS = {"git", "npm", "pnpm", "yarn", "cargo", "go", "docker", "kubectl", "gh",
               "pip", "pip3", "uv", "brew", "apt", "terraform", "bun", "npx"}
PATH_KEYS = ("file_path", "path", "notebook_path")


def tokens(command):
    """Tokens of the first real command in a shell line (skips `cd x &&` and VAR=val)."""
    for seg in re.split(r"&&|;", command):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        while toks and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[0]):
            toks = toks[1:]
        if toks and toks[0] != "cd":
            return toks
    return []


def step(call):
    """A short key naming what kind of call this is, with arguments stripped."""
    tool, inp = call["tool"], call["input"]
    if tool == "Bash":
        t = tokens(inp.get("command", ""))
        if not t:
            return "sh:?"
        prog = os.path.basename(t[0])
        rest = t[1:]
        if prog in ("python", "python3") and len(rest) >= 2 and rest[0] == "-m":
            return "sh:python -m " + rest[1]
        if prog in SUBCOMMANDS:
            sub = [x for x in rest if not x.startswith("-")][:2 if prog in ("docker", "gh") and
                   len(rest) > 1 and rest[0] in ("compose", "pr", "issue", "run") else 1]
            return "sh:" + " ".join([prog] + sub)
        return "sh:" + prog
    if tool == "Skill":
        return "Skill:" + str(inp.get("skill", "?"))
    for k in PATH_KEYS:
        if k in inp:
            return tool + ":" + (os.path.splitext(str(inp[k]))[1] or "(none)")
    return tool


def error_signature(error):
    """One normalised line from an error message, so the same failure groups together."""
    lines = [l.strip() for l in error.splitlines() if l.strip()]
    if not lines:
        return "(no message)"
    pick = next((l for l in lines if re.search(
        r"error|not found|no such|denied|failed|cannot|fatal", l, re.I)), lines[0])
    pick = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`", "<s>", pick)
    pick = re.sub(r"(?<![\w<])[\w.\-]*/[\w./\-]*", "<path>", pick)
    pick = re.sub(r"\b[0-9a-f]{7,}\b", "<hex>", pick)
    pick = re.sub(r"\d+", "<n>", pick)
    return pick[:100]

# ---------------------------------------------------------------- workflows

def collapse(keys):
    out = []
    for k in keys:
        if not out or out[-1] != k:
            out.append(k)
    return out


def poisson_tail(n, lam):
    """P(X >= n) for X ~ Poisson(lam)."""
    term = total = math.exp(-lam)
    for k in range(1, n):
        term *= lam / k
        total += term
    return max(0.0, 1.0 - total)


def mine_workflows(sessions, min_support=None, max_len=6, max_p=1e-4, keep=0.7):
    """Contiguous step sequences (length 3..max_len) that recur across sessions.

    support = number of distinct sessions containing the sequence.
    Sequences that include a failed call are skipped (mine_failures covers those).
    A sequence is kept only if its last step follows the rest, and its first step precedes
    the rest, more often than chance: given how common that step is overall, the Poisson
    probability of seeing this much support must be under max_p on both ends.
    Then sequences contained in a longer kept sequence with >= keep of their support
    are dropped. max_p=1 gives the plain support-threshold baseline.
    """
    seqs = {sid: collapse([step(c) + ("" if c["ok"] else "!") for c in calls])
            for sid, calls in sessions.items()}
    if min_support is None:
        min_support = max(3, int(0.02 * len(seqs)))
    uni = Counter(k for s in seqs.values() for k in s)
    total = sum(uni.values()) or 1
    sup = Counter()
    where = defaultdict(list)
    for sid, s in seqs.items():
        seen = set()
        for n in range(2, max_len + 1):
            for i in range(len(s) - n + 1):
                seen.add(tuple(s[i:i + n]))
        sup.update(seen)
        for g in seen:
            if len(g) >= 3:
                where[g].append(sid)
    kept = []
    for g, n in sup.items():
        if len(g) < 3 or n < min_support or any(k.endswith("!") for k in g):
            continue
        exp_r = sup[g[:-1]] * uni[g[-1]] / total
        exp_l = sup[g[1:]] * uni[g[0]] / total
        lift = n / max(exp_l, exp_r)
        if max_p >= 1 or max(poisson_tail(n, exp_l), poisson_tail(n, exp_r)) < max_p:
            kept.append((g, n, lift))
    kept.sort(key=lambda x: (-len(x[0]), -x[1], x[0]))
    out = []
    for g, n, lift in kept:
        if any(len(h) > len(g) and hn >= keep * n and _contains(h, g) for h, hn, _ in out):
            continue
        out.append((g, n, lift))
    out.sort(key=lambda x: (-x[1] * (len(x[0]) - 1), x[0]))
    return [{"steps": list(g), "sessions": n, "lift": round(lift, 1),
             "examples": sorted(where[g])[:3]} for g, n, lift in out]


def _contains(h, g):
    return any(h[i:i + len(g)] == g for i in range(len(h) - len(g) + 1))

# ---------------------------------------------------------------- failures

def _target(call):
    inp = call["input"]
    if call["tool"] == "Bash":
        return tokens(inp.get("command", ""))
    return [str(inp[k]) for k in PATH_KEYS if k in inp]


def _same_intent(a, b):
    """Is successful call b plausibly the retry of failed call a?"""
    if a["tool"] != b["tool"]:
        return False
    ta, tb = _target(a), _target(b)
    if a["tool"] != "Bash":
        return ta == tb
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return False
    if step(a) == step(b) or (ta[1:] and ta[1:] == tb[1:]):
        return True
    return len(sa & sb) / len(sa | sb) >= 0.5


def _fix_label(failed, fixed, between):
    ta, tb = _target(failed), _target(fixed)
    if failed["tool"] == "Bash" and ta != tb:
        err = failed["error"]
        removed = [t for t in ta if t not in tb]
        added = ["<from-error>" if len(t) > 2 and t in err else t for t in tb if t not in ta]
        if removed and added:
            return "replace `%s` with `%s`" % (" ".join(removed), " ".join(added))
        if added and tb[len(tb) - len(ta):] == ta:
            return "prefix the command with `%s`" % " ".join(added)
        if added:
            return "add `%s`" % " ".join(added)
        return "drop `%s`" % " ".join(removed)
    mid = collapse([step(c) for c in between])
    if 1 <= len(mid) <= 2:
        return "first run " + " then ".join("`%s`" % m for m in mid) + ", then retry"
    return "retry unchanged" if not mid else "retry later"


def merge_signatures(sigs):
    """Map each signature to a template. Signatures with the same word count that differ
    in one word position are merged, with that word replaced by <*>."""
    out, buckets = {}, defaultdict(list)
    for s in sorted(sigs):
        words = s.split()
        for tpl in buckets[len(words)]:
            diff = [i for i, (a, b) in enumerate(zip(tpl, words)) if a != b and a != "<*>"]
            if len(diff) <= 1 and len(words) > 2:
                for i in diff:
                    tpl[i] = "<*>"
                out[s] = tpl
                break
        else:
            tpl = list(words)
            buckets[len(words)].append(tpl)
            out[s] = tpl
    return {s: " ".join(t) for s, t in out.items()}


def mine_failures(sessions, min_support=None, window=4, min_agree=0.6):
    """Group failed calls by (step, error signature) and find the fix that usually follows.

    A fix is the next successful call within `window` calls that has the same intent.
    A rule is reported when the failure shows up in >= min_support sessions and one fix
    accounts for >= min_agree of all occurrences (recovered or not).
    """
    if min_support is None:
        min_support = max(3, int(0.02 * len(sessions)))
    groups = defaultdict(lambda: {"n": 0, "sessions": set(), "fixes": Counter()})
    by_step = defaultdict(set)
    for calls in sessions.values():
        for c in calls:
            if not c["ok"]:
                by_step[step(c)].add(error_signature(c["error"]))
    template = {k: merge_signatures(v) for k, v in by_step.items()}
    for sid, calls in sessions.items():
        for i, c in enumerate(calls):
            if c["ok"]:
                continue
            key = step(c)
            g = groups[(key, template[key][error_signature(c["error"])])]
            g["n"] += 1
            g["sessions"].add(sid)
            for j in range(i + 1, min(len(calls), i + 1 + window)):
                d = calls[j]
                if d["ok"] and _same_intent(c, d):
                    g["fixes"][_fix_label(c, d, calls[i + 1:j])] += 1
                    break
    rules, unresolved = [], []
    for (key, sig), g in groups.items():
        if len(g["sessions"]) < min_support:
            continue
        fix, count = g["fixes"].most_common(1)[0] if g["fixes"] else ("", 0)
        row = {"step": key, "error": sig, "sessions": len(g["sessions"]),
               "occurrences": g["n"], "recovered": sum(g["fixes"].values()),
               "examples": sorted(g["sessions"])[:3]}
        if count / g["n"] >= min_agree and not fix.startswith("retry"):
            row.update(fix=fix, agreement=round(count / g["n"], 2))
            rules.append(row)
        else:
            unresolved.append(row)
    rules.sort(key=lambda r: (-r["sessions"], r["step"]))
    unresolved.sort(key=lambda r: (-r["sessions"], r["step"]))
    return rules, unresolved

# ---------------------------------------------------------------- skill usage

def skill_stats(sessions, window=10):
    """Per skill: invocations, sessions, and tool error rate in the next `window` calls."""
    stats = defaultdict(lambda: {"invocations": 0, "sessions": set(), "after": 0, "errors": 0})
    all_calls = all_err = 0
    for sid, calls in sessions.items():
        for i, c in enumerate(calls):
            all_calls += 1
            all_err += not c["ok"]
            if c["tool"] != "Skill":
                continue
            s = stats[str(c["input"].get("skill", "?"))]
            s["invocations"] += 1
            s["sessions"].add(sid)
            nxt = calls[i + 1:i + 1 + window]
            s["after"] += len(nxt)
            s["errors"] += sum(not d["ok"] for d in nxt)
    base = all_err / all_calls if all_calls else 0.0
    rows = [{"skill": k, "invocations": s["invocations"], "sessions": len(s["sessions"]),
             "error_rate_after": round(s["errors"] / s["after"], 3) if s["after"] else None}
            for k, s in stats.items()]
    rows.sort(key=lambda r: -r["invocations"])
    return rows, round(base, 3)

# ---------------------------------------------------------------- proposals

def propose(skill_text, workflows, rules, name="SKILL.md", top=5):
    """Unified diff that appends mined sections to a skill file. Nothing is written."""
    add = []
    if rules:
        add += ["", "## Known failures (mined from traces, review before merging)", ""]
        for r in rules[:top]:
            add.append("- When `%s` fails with `%s`: %s. Seen in %d sessions, this fix followed "
                       "%d%% of the time (e.g. %s)." % (
                           r["step"], r["error"], r["fix"], r["sessions"],
                           round(100 * r["agreement"]), ", ".join(r["examples"])))
    if workflows:
        add += ["", "## Repeated workflows (mined from traces, review before merging)", ""]
        for w in workflows[:top]:
            add.append("- %s. Seen in %d sessions (e.g. %s)." % (
                " -> ".join("`%s`" % s for s in w["steps"]), w["sessions"],
                ", ".join(w["examples"])))
    old = skill_text.splitlines()
    return "\n".join(difflib.unified_diff(old, old + add, "a/" + name, "b/" + name, lineterm=""))


def skills_main(paths, as_json):
    """Per skill verdicts and analytics. Each path is parsed on its own, because a Codex
    rollout file names its session only once, on its first line."""
    import skilleval
    sessions = {}
    for path in paths:
        with open(path) as f:
            sessions.update(formats.parse(f))
    rep = skilleval.report(sessions)
    if as_json:
        rows = [{k: r[k] for k in ("session", "agent", "skill", "how", "verdict", "reason")}
                for r in rep["rows"]]
        print(json.dumps({"invocations": rows, "analytics": rep["analytics"],
                          "diffs": rep["diffs"]}, indent=1))
        return 0
    print("%d sessions, %d skill invocations" % (len(sessions), len(rep["rows"])))
    print("\n%-18s %5s %5s %8s  %s" % ("skill", "uses", "eval", "success", "verdicts"))
    for a in rep["analytics"]:
        rate = "-" if a["success_rate"] is None else "%.0f%%" % (100 * a["success_rate"])
        print("%-18s %5d %5d %8s  %s" % (a["skill"], a["invocations"], a["evaluated"], rate,
              ", ".join("%s %d" % (v, n) for v, n in a["verdicts"].items() if n)))
    print("\ninvocations that were not a success")
    for r in rep["rows"]:
        if r["verdict"] != "success":
            print("  %s  %s  %s: %s" % (r["session"], r["skill"], r["verdict"], r["reason"]))
    for skill, diff in sorted(rep["diffs"].items()):
        print("\nproposed edit for %s\n%s" % (skill, diff))
    return 0


def main(argv):
    if len(argv) >= 2 and argv[0] == "skills":
        return skills_main([a for a in argv[1:] if not a.startswith("--")], "--json" in argv)
    if len(argv) < 2 or argv[0] != "report":
        print("usage: python tracemine.py report traces.jsonl [--skill SKILL.md] [--json]\n"
              "       python tracemine.py skills session.jsonl [more.jsonl ...] [--json]")
        return 2
    with open(argv[1]) as f:
        sessions = load(f)
    workflows = mine_workflows(sessions)
    rules, unresolved = mine_failures(sessions)
    skills, base = skill_stats(sessions)
    if "--json" in argv:
        print(json.dumps({"workflows": workflows, "rules": rules, "unresolved": unresolved,
                          "skills": skills, "base_error_rate": base}, indent=1))
        return 0
    print("%d sessions, %d calls, base tool error rate %.3f" % (
        len(sessions), sum(len(c) for c in sessions.values()), base))
    print("\nrepeated workflows")
    for w in workflows:
        print("  %3d sessions  %s" % (w["sessions"], " -> ".join(w["steps"])))
    print("\nfailures with a consistent fix")
    for r in rules:
        print("  %3d sessions  %s | %s | %s (%d%%)" % (
            r["sessions"], r["step"], r["error"], r["fix"], round(100 * r["agreement"])))
    print("\nrecurring failures with no consistent fix")
    for r in unresolved:
        print("  %3d sessions  %s | %s | recovered %d of %d" % (
            r["sessions"], r["step"], r["error"], r["recovered"], r["occurrences"]))
    print("\nskills")
    for s in skills:
        print("  %-20s %3d invocations in %3d sessions, error rate in next 10 calls %s" % (
            s["skill"], s["invocations"], s["sessions"], s["error_rate_after"]))
    if "--skill" in argv:
        path = argv[argv.index("--skill") + 1]
        with open(path) as f:
            print("\nproposed edit\n" + propose(f.read(), workflows, rules, os.path.basename(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
