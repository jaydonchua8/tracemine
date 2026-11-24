"""skilleval: find skill invocations in agent sessions, give each one a verdict, roll the
verdicts up per skill, and mine failure/fix rules per skill so an edit can be proposed
against the right SKILL.md.

The verdict names are the result categories listed in Nessie's public "Skill evals" doc
page (success, skill error, unexpected behavior, other error, unknown). How Nessie decides
them is not public. What is here is my own rule based baseline that runs offline.

Standard library only.
"""
import re
from collections import Counter, defaultdict

import tracemine as tm

VERDICTS = ["success", "skill_error", "unexpected_behavior", "other_error", "unknown"]

SKILL_MD = re.compile(r"skills/([A-Za-z0-9_.\-]+)/SKILL\.md\b")
SLASH = re.compile(r"<command-name>\s*/?([A-Za-z0-9_.:\-]+)\s*</command-name>")
SKILL_TAG = re.compile(r"<skill>\s*<name>\s*([A-Za-z0-9_.:\-]+)\s*</name>")
MENTION = re.compile(r"(?<![\w$])\$([a-z][a-z0-9\-]{2,})\b")
BUILTIN_COMMANDS = {"clear", "help", "compact", "model", "config", "cost", "login", "logout",
                    "resume", "status", "init", "memory", "permissions", "exit", "mcp",
                    "agents", "review", "doctor", "context", "plugin", "hooks"}
BENIGN_EXIT1 = {"grep", "rg", "egrep", "fgrep", "diff", "cmp", "test", "[", "pgrep"}
BARE_EXIT = re.compile(r"^(?:exit code|process exited with code)[: ]+1\.?$", re.I)

CORRECTION = [re.compile(p, re.I) for p in (
    r"^\s*(?:no|nope)\b(?!\s+(?:rush|worries|problem|need|changes)\b)",
    r"^\s*(?:wrong|stop|wait)\b",
    r"\bnot what i (?:asked|wanted|meant)\b",
    r"\b(?:undo|revert) (?:that|this|it|the|your)\b",
    r"\bthat(?:'s| is| was) (?:not right|wrong|incorrect|not correct)\b",
    r"\bi (?:didn'?t|did not|never) ask\b",
    r"\bwhy did you\b",
    r"\byou (?:skipped|forgot|missed|ignored)\b",
    r"\bshould(?:n'?t| not) have\b",
    r"\b(?:try|do|run) (?:it |that |the skill )?again\b",
)]

DEFAULTS = {"tail": 2, "min_coverage": 0.5, "max_calls": 40, "min_profile": 5}

# ---------------------------------------------------------------- invocations

def _skill_from_call(call):
    """(skill name, how) if this call invokes a skill, else None."""
    inp = call["input"]
    if call["tool"] == "Skill":
        return str(inp.get("skill") or inp.get("command") or "?"), "tool"
    if call["tool"] == "Read":
        m = SKILL_MD.search(str(inp.get("file_path", "")))
        if m:
            return m.group(1), "read"
    return None


def known_skills(sessions):
    """Skill names seen as a Skill tool call or a SKILL.md read anywhere in the corpus."""
    names = set()
    for s in sessions.values():
        for e in s["events"]:
            if e["kind"] == "call":
                hit = _skill_from_call(e)
                if hit:
                    names.add(hit[0])
    return names


def invocations(session, known=()):
    """Skill invocations in one session, in order.

    Signals: a Skill tool call (Claude Code), a slash command tag in a user message
    (Claude Code), a read of skills/<name>/SKILL.md (Codex and others), a <skill><name>
    block in a user message, and a $name mention of a known skill (Codex). Several signals
    for the same skill inside one turn count as one invocation, and the first tool call
    among them is kept as the invoking call.
    """
    out, turn = [], {}
    for i, e in enumerate(session["events"]):
        hits = []
        if e["kind"] == "user":
            turn = {}
            text = e["text"]
            hits += [(n, "slash") for n in SLASH.findall(text) if n not in BUILTIN_COMMANDS]
            hits += [(n, "tag") for n in SKILL_TAG.findall(text)]
            hits += [(n, "mention") for n in MENTION.findall(text) if n in known]
        else:
            hit = _skill_from_call(e)
            if hit:
                hits.append(hit)
        for name, how in hits:
            if name in turn:
                inv = turn[name]
                if inv["call"] is None and e["kind"] == "call":
                    inv["call"], inv["index"] = e, i
                continue
            inv = {"skill": name, "how": how, "index": i,
                   "call": e if e["kind"] == "call" else None}
            turn[name] = inv
            out.append(inv)
    return out


def span(session, inv, stop_at=None, max_calls=DEFAULTS["max_calls"]):
    """Calls that belong to an invocation: everything after it up to the next user message,
    the next invocation, the end of the session, or max_calls.

    Returns (calls, next_user_text or None, how_it_ended).
    """
    events, calls = session["events"], []
    for j in range(inv["index"] + 1, len(events)):
        e = events[j]
        if stop_at is not None and j >= stop_at:
            return calls, None, "skill"
        if e["kind"] == "user":
            return calls, e["text"], "user"
        if len(calls) >= max_calls:
            return calls, None, "cap"
        calls.append(e)
    return calls, None, "eof"

# ---------------------------------------------------------------- verdicts

def is_failure(call):
    """A failed call, not counting exit status 1 from programs that use it to say "no"."""
    if call["ok"]:
        return False
    if call["tool"] == "Bash":
        t = tm.tokens(call["input"].get("command", ""))
        err = call["error"].strip()
        if t and t[0].rsplit("/", 1)[-1] in BENIGN_EXIT1 and call.get("exit", 1) == 1 and \
                (not err or BARE_EXIT.match(err)):
            return False
    return True


def _touches_skill(call, skill):
    inp = call["input"]
    text = str(inp.get("command", "")) + " " + " ".join(str(inp[k]) for k in tm.PATH_KEYS if k in inp)
    return ("skills/%s/" % skill) in text


def unresolved_failures(calls):
    """Indexes of failed calls with no later successful call of the same intent."""
    out = []
    for i, c in enumerate(calls):
        if is_failure(c) and not any(d["ok"] and tm._same_intent(c, d) for d in calls[i + 1:]):
            out.append(i)
    return out


def is_correction(text):
    return bool(text) and any(p.search(text) for p in CORRECTION)


def profiles(spans_by_skill, min_profile=DEFAULTS["min_profile"]):
    """Per skill, the steps that at least half of its invocations run. Learned from the
    traces being evaluated, with no labels."""
    out = {}
    for skill, spans in spans_by_skill.items():
        if len(spans) < min_profile:
            continue
        df = Counter()
        for calls in spans:
            df.update({tm.step(c) for c in calls if c["ok"]})
        common = {k for k, n in df.items() if n >= 0.5 * len(spans)}
        if common:
            out[skill] = common
    return out


def verdict(inv, calls, next_user, ended, truncated, profile=None, tail=DEFAULTS["tail"],
            min_coverage=DEFAULTS["min_coverage"]):
    """Return (verdict, reason)."""
    skill, head = inv["skill"], inv["call"]
    if head is not None and not head["ok"]:
        return "skill_error", "the call that loads the skill failed"
    if ended == "eof" and (truncated or (head is not None and not head["done"]) or
                           any(not c["done"] for c in calls)):
        return "unknown", "session ends before the last call has a result"
    bad = unresolved_failures(calls)
    for i in bad:
        if _touches_skill(calls[i], skill):
            return "skill_error", "a file under skills/%s/ failed: %s" % (
                skill, tm.error_signature(calls[i]["error"]))
    late = [i for i in bad if i >= len(calls) - tail]
    if late:
        return "other_error", "ends on an unrecovered failure: %s | %s" % (
            tm.step(calls[late[-1]]), tm.error_signature(calls[late[-1]]["error"]))
    if is_correction(next_user):
        return "unexpected_behavior", "the next user message corrects the agent"
    if profile:
        done = {tm.step(c) for c in calls if c["ok"]}
        coverage = len(done & profile) / len(profile)
        if coverage < min_coverage:
            return "unexpected_behavior", "ran %d%% of the steps this skill usually runs" % round(
                100 * coverage)
    return "success", ""


def evaluate(sessions, **params):
    """Verdict for every skill invocation. Returns a list of rows with keys
    session, agent, skill, how, verdict, reason, calls (the span)."""
    p = dict(DEFAULTS, **params)
    known = known_skills(sessions)
    rows = []
    for sid, s in sessions.items():
        invs = invocations(s, known)
        for n, inv in enumerate(invs):
            stop = invs[n + 1]["index"] if n + 1 < len(invs) else None
            calls, nxt, ended = span(s, inv, stop, p["max_calls"])
            rows.append({"session": sid, "agent": s["agent"], "skill": inv["skill"],
                         "how": inv["how"], "n": n, "inv": inv, "calls": calls,
                         "next_user": nxt, "ended": ended, "truncated": s["truncated"]})
    by_skill = defaultdict(list)
    for r in rows:
        by_skill[r["skill"]].append(r["calls"])
    prof = profiles(by_skill, p["min_profile"])
    for r in rows:
        r["verdict"], r["reason"] = verdict(
            r["inv"], r["calls"], r["next_user"], r["ended"], r["truncated"],
            prof.get(r["skill"]), p["tail"], p["min_coverage"])
    return rows

# ---------------------------------------------------------------- analytics

def analytics(rows):
    """Per skill: invocations, sessions, verdict counts, and success rate over evaluated
    invocations (unknown ones are left out of the rate), split by agent."""
    acc = defaultdict(lambda: {"sessions": set(), "verdicts": Counter(), "agents": Counter()})
    for r in rows:
        a = acc[r["skill"]]
        a["sessions"].add(r["session"])
        a["verdicts"][r["verdict"]] += 1
        a["agents"][r["agent"]] += 1
    out = []
    for skill, a in acc.items():
        total = sum(a["verdicts"].values())
        evaluated = total - a["verdicts"]["unknown"]
        out.append({"skill": skill, "invocations": total, "sessions": len(a["sessions"]),
                    "evaluated": evaluated,
                    "success_rate": round(a["verdicts"]["success"] / evaluated, 3) if evaluated else None,
                    "verdicts": {v: a["verdicts"][v] for v in VERDICTS},
                    "agents": dict(a["agents"])})
    out.sort(key=lambda r: (-r["invocations"], r["skill"]))
    return out

# ---------------------------------------------------------------- per skill rules

def skill_rules(rows, min_support=3, min_share=0.6):
    """Failure/fix rules mined only from the calls inside each skill's invocations.

    A rule is attached to a skill when it has min_support invocations behind it there and
    at least min_share of all its occurrences across skills come from that skill. A fix
    that shows up equally under every skill is a general habit, not that skill's problem,
    and is returned under the key None.
    """
    per, totals = {}, Counter()
    for skill in {r["skill"] for r in rows}:
        sub = {"%s#%d" % (r["session"], r["n"]): r["calls"] for r in rows
               if r["skill"] == skill and r["calls"]}
        rules, unresolved = tm.mine_failures(sub, min_support=min_support)
        per[skill] = (rules, unresolved)
        for x in rules:
            totals[(x["step"], x["error"], x["fix"])] += x["occurrences"]
    out = defaultdict(list)
    for skill, (rules, _) in per.items():
        for x in rules:
            share = x["occurrences"] / totals[(x["step"], x["error"], x["fix"])]
            x["share"] = round(share, 2)
            out[skill if share >= min_share else None].append(x)
    general = {}
    for x in out.pop(None, []):
        general.setdefault((x["step"], x["error"], x["fix"]), x)
    result = {k: sorted(v, key=lambda r: -r["sessions"]) for k, v in out.items()}
    result[None] = list(general.values())
    return result, {k: v[1] for k, v in per.items()}


def report(sessions, skill_files=None, **params):
    """Everything in one dict: verdict rows, analytics, per skill rules and diffs.
    skill_files maps a skill name to the current text of its SKILL.md."""
    rows = evaluate(sessions, **params)
    rules, unresolved = skill_rules(rows)
    diffs = {}
    for skill, rr in rules.items():
        if skill is None or not rr:
            continue
        text = (skill_files or {}).get(skill, "")
        diffs[skill] = tm.propose(text, [], rr, name="skills/%s/SKILL.md" % skill)
    return {"rows": rows, "analytics": analytics(rows), "rules": rules,
            "unresolved": unresolved, "diffs": diffs}
