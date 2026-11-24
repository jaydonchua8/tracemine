"""Synthetic benchmark for skilleval.

Generates labelled sessions, writes them out in the Claude Code transcript shape and the
Codex rollout shape, parses them back through formats.py, and scores the verdicts.

Two generator settings are used. "tune" is what I picked the thresholds on. "held-out"
uses other seeds, four skills that do not appear in tune, more Codex sessions, more
harmless failures, longer wrap-up after a failure, and user follow-ups phrased in ways
the correction patterns were not written against.

python bench_skills.py              run both, print tables, update bench_results.json
python bench_skills.py --dump DIR   write tune seed 11 as DIR/claude_code.jsonl and DIR/codex.jsonl

All data here is synthetic.
"""
import json
import os
import random
import sys

import bench
import formats
import skilleval as se
import tracemine as tm


def call(tool, inp, ok=True, error="", code=1, done=True):
    return {"tool": tool, "input": inp, "ok": ok, "error": error, "code": code, "done": done}


def sh(cmd, ok=True, error="", code=1, done=True):
    return call("Bash", {"command": cmd}, ok, error, code, done)


F = bench.f
SKILL_DIR = {"claude_code": ".claude/skills", "codex": ".agents/skills"}

# name -> list of step makers (r, skill_dir) -> call
TUNE_SKILLS = {
    "release-check": [lambda r, d: sh("git fetch --tags"), lambda r, d: sh("git log -n 20 --oneline"),
                      lambda r, d: call("Read", {"file_path": "CHANGELOG.md"}),
                      lambda r, d: sh("bash %s/release-check/scripts/check.sh" % d),
                      lambda r, d: sh("npm run build")],
    "db-migrate": [lambda r, d: call("Read", {"file_path": "db/schema.sql"}),
                   lambda r, d: sh("alembic revision -m 'add %s'" % r.choice(bench.WORDS)),
                   lambda r, d: call("Edit", {"file_path": "migrations/versions/%04d.py" % r.randint(1, 99)}),
                   lambda r, d: sh("alembic upgrade head"),
                   lambda r, d: sh("pytest tests/test_db.py -q")],
    "pr-review": [lambda r, d: sh("gh pr view %d" % r.randint(10, 900)),
                  lambda r, d: sh("gh pr diff %d" % r.randint(10, 900)),
                  lambda r, d: call("Read", {"file_path": F(r) + ".py"}),
                  lambda r, d: call("Grep", {"pattern": r.choice(bench.WORDS)}),
                  lambda r, d: sh("gh pr comment %d --body-file /tmp/review.md" % r.randint(10, 900))],
    "add-endpoint": [lambda r, d: call("Grep", {"pattern": "router"}),
                     lambda r, d: call("Read", {"file_path": "api/routes.ts"}),
                     lambda r, d: call("Edit", {"file_path": "api/routes.ts"}),
                     lambda r, d: call("Write", {"file_path": "api/%s.test.ts" % r.choice(bench.WORDS)}),
                     lambda r, d: sh("npx tsc --noEmit"), lambda r, d: sh("npm test")],
    "bump-deps": [lambda r, d: sh("npm view %s version" % r.choice(bench.PKGS)),
                  lambda r, d: call("Edit", {"file_path": "package.json"}),
                  lambda r, d: sh("npm install"), lambda r, d: sh("npm test"),
                  lambda r, d: sh("git commit -am 'bump deps'")],
}
HELD_SKILLS = {
    "incident-notes": [lambda r, d: sh("kubectl logs deploy/api --since=1h"),
                       lambda r, d: sh("kubectl get events -n prod"),
                       lambda r, d: call("Grep", {"pattern": "ERROR"}),
                       lambda r, d: call("Write", {"file_path": "incidents/2026-%02d.md" % r.randint(1, 12)})],
    "py-lint-fix": [lambda r, d: sh("ruff check ."), lambda r, d: call("Edit", {"file_path": F(r) + ".py"}),
                    lambda r, d: sh("mypy src"), lambda r, d: sh("pytest -q")],
    "changelog": [lambda r, d: sh("git log -n 30 --oneline"),
                  lambda r, d: call("Read", {"file_path": "CHANGELOG.md"}),
                  lambda r, d: call("Edit", {"file_path": "CHANGELOG.md"})],
    "docker-rebuild": [lambda r, d: sh("docker compose build api"), lambda r, d: sh("docker compose up -d"),
                       lambda r, d: sh("curl -sf http://localhost:8080/health"),
                       lambda r, d: sh("docker logs api-1 --tail 50")],
}

# A failure habit tied to one skill: (step command, error, the calls that fix it, expected rule)
HABITS = {
    "db-migrate": ("alembic upgrade head",
                   "sqlalchemy.exc.OperationalError: connection to server at localhost port 5432 failed: Connection refused",
                   "docker compose up -d db", ("sh:alembic", "first run `sh:docker compose up`, then retry")),
    "py-lint-fix": ("mypy src",
                    "src/config.py:3: error: Library stubs not installed for \"yaml\"",
                    "pip install types-PyYAML", ("sh:mypy", "first run `sh:pip install`, then retry")),
}

PROBES = [("ls dist", "ls: cannot access 'dist': No such file or directory", 2),
          ("which pnpm", "", 1),
          ("git stash pop", "No stash entries found.", 1),
          ("node scripts/legacy-check.js", "Error: Cannot find module '/repo/scripts/legacy-check.js'", 1)]
HARD_FAILS = [("npm run build", "src/api.ts(41,7): error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.", 2),
              ("git push", " ! [rejected]        main -> main (fetch first)\nerror: failed to push some refs to 'origin'", 1),
              ("curl -sf http://localhost:8080/health", "curl: (7) Failed to connect to localhost port 8080: Connection refused", 7),
              ("pytest tests -q", "FAILED tests/test_api.py::test_auth - AssertionError\n1 failed, 12 passed", 1),
              ("docker compose up -d", "Error response from daemon: driver failed programming external connectivity: port is already allocated", 1),
              ("terraform plan", "Error: No valid credential sources found", 1)]
SCRIPT_ERRORS = [("check.sh: line 12: jq: command not found", 127),
                 ("Traceback (most recent call last):\n  File \"run.py\", line 8, in <module>\nKeyError: 'version'", 1),
                 ("check.sh: line 3: syntax error near unexpected token `fi'", 2)]

PROMPTS = ["the health check is flaky, take a look", "ship the fix for the token refresh bug",
           "add pagination to the users list", "clean up the retry helper",
           "get this branch ready", "the nightly job failed, find out why",
           "update the docs for the cache flag", "do the usual for this ticket"]
BENIGN = ["thanks, now update the README to mention it",
          "looks good. can you also add a test for the empty case",
          "no rush, but next rename the helper to parse_row",
          "ok, next ticket: the search endpoint is slow", "great. commit that",
          "no need to squash, just open the PR", "nice, now do the same for the staging config"]
BENIGN_HELD = ["wait for CI and then merge it",
               "no changes needed there. moving on, bump the base image",
               "stop the dev server when you are done and then summarize",
               "why did you pick port 8081? just curious, it is fine"]
CORRECT = ["no, that's not what I asked for. only touch the api package",
           "you skipped the changelog step, do it again",
           "wrong branch. revert that and use release/2.4",
           "why did you edit the lockfile by hand? undo that",
           "that's not right, the migration should be reversible",
           "I didn't ask you to push. revert the push",
           "nope, run the skill again and follow every step this time",
           "stop. you forgot to run the tests"]
CORRECT_HELD = ["hmm, this isn't it. the version in package.json is unchanged",
                "you were supposed to open a draft PR, not merge",
                "that broke staging, roll it back",
                "this missed half the files in the checklist",
                "not quite, the notes need the timeline section",
                "please redo this properly, the output is empty",
                "the skill says to ask before deleting, and you deleted them anyway",
                "uh, where are the tests it was meant to add?"]

TUNE = {"name": "tune", "skills": TUNE_SKILLS, "seeds": [11, 12, 13], "sessions": 300,
        "codex": 0.3, "probe": 0.15, "wrapup": 1, "held_phrases": 0.0, "noise": (0, 2),
        "mix": [("success", 0.60), ("skill_error", 0.08), ("other_error", 0.12),
                ("unexpected_behavior", 0.15), ("unknown", 0.05)]}
HELD = {"name": "held-out", "skills": HELD_SKILLS, "seeds": [101, 102], "sessions": 300,
        "codex": 0.6, "probe": 0.30, "wrapup": 3, "held_phrases": 0.5, "noise": (0, 4),
        "mix": [("success", 0.50), ("skill_error", 0.10), ("other_error", 0.15),
                ("unexpected_behavior", 0.20), ("unknown", 0.05)]}

# ---------------------------------------------------------------- generating

def noise(r, n):
    return [call(c["tool"], c["input"]) for c in bench.noise(r, n)]


def from_flat(calls):
    return [call(c["tool"], c["input"], c["ok"], c["error"]) for c in calls]


def script_body(r, cfg, name, d, skip=()):
    """The skill's steps in order with background calls in between."""
    out = []
    for i, mk in enumerate(cfg["skills"][name]):
        if i in skip:
            continue
        out.append(mk(r, d))
        out += noise(r, r.randint(*cfg["noise"]))
    return out


def success_span(r, cfg, name, d):
    steps = cfg["skills"][name]
    skip = {r.randrange(len(steps))} if r.random() < 0.15 else set()
    out = script_body(r, cfg, name, d, skip)
    habit = HABITS.get(name)
    if habit and r.random() < 0.5:
        cmd, err, fix, _ = habit
        idx = next((i for i, c in enumerate(out) if c["input"].get("command") == cmd), None)
        if idx is not None:
            out[idx:idx] = [sh(cmd, False, err), sh(fix)]
    if r.random() < 0.10:
        out[r.randrange(len(out) + 1):0] = from_flat(bench.fail_edit_stale(r))
    if r.random() < 0.10:
        out.insert(r.randrange(len(out) + 1), sh("grep -rn TODO %s" % F(r).split("/")[0], False, "", 1))
    if r.random() < cfg["probe"]:
        cmd, err, code = r.choice(PROBES)
        out.insert(r.randrange(len(out) + 1), sh(cmd, False, err, code))
    return out


def span_for(r, cfg, label, name, agent):
    """(calls after the invocation, invocation ok, follow-up kind)."""
    d = SKILL_DIR[agent]
    n = len(cfg["skills"][name])
    if label == "success":
        return success_span(r, cfg, name, d), True, "benign"
    if label == "skill_error":
        if r.random() < 0.4:
            return noise(r, r.randint(0, 3)), False, "benign"
        err, code = r.choice(SCRIPT_ERRORS)
        script = "%s/%s/scripts/%s" % (d, name, "check.sh" if "check.sh" in err else "run.py")
        cmd = ("bash " if script.endswith(".sh") else "python3 ") + script
        out = script_body(r, cfg, name, d, set(range(r.randint(1, n), n)))
        out += [sh(cmd, False, err, code)] * r.randint(1, 2)
        return out + noise(r, r.randint(0, 4)), True, "benign"
    if label == "other_error":
        cmd, err, code = r.choice(HARD_FAILS)
        out = script_body(r, cfg, name, d, set(range(r.randint(1, n), n)))
        out += [sh(cmd, False, err, code) for _ in range(r.randint(1, 3))]
        return out + noise(r, r.randint(0, cfg["wrapup"])), True, "benign"
    if label == "unexpected_behavior":
        kind = r.choices(["corrected", "skipped", "both"], [0.5, 0.3, 0.2])[0]
        if kind == "corrected":
            return success_span(r, cfg, name, d), True, "correct"
        keep = r.randrange(n) if r.random() < 0.5 else -1
        out = script_body(r, cfg, name, d, set(range(n)) - {keep}) + noise(r, r.randint(3, 6))
        return out, True, "correct" if kind == "both" else "benign"
    out = script_body(r, cfg, name, d, set(range(r.randint(1, n), n)))     # unknown
    out.append(sh("npm test", done=False))
    return out, True, "end"


def generate(seed, cfg):
    """Returns (sessions, truth). A session is {"agent", "items", "torn"} and an item is
    ("user", text) or ("call", call). truth[sid] is a list of (skill, label)."""
    r = random.Random(seed)
    names = sorted(cfg["skills"])
    labels, weights = zip(*cfg["mix"])
    sessions, truth = {}, {}
    for i in range(cfg["sessions"]):
        agent = "codex" if r.random() < cfg["codex"] else "claude_code"
        sid = "%s-%d-%03d" % (cfg["name"], seed, i)
        items, marks, follow, torn = [], [], "benign", False
        for _ in range(r.randint(2, 5)):
            if follow == "correct":
                pool = CORRECT_HELD if r.random() < cfg["held_phrases"] else CORRECT
            elif r.random() < 0.5:
                pool = PROMPTS
            else:
                pool = BENIGN_HELD if r.random() < cfg["held_phrases"] * 0.6 else BENIGN
            text, follow = r.choice(pool), "benign"
            if r.random() < 0.08:
                text += " (and check that $HOME/.npmrc is not committed)"
            if r.random() >= 0.55:
                items.append(("user", text))
                body = noise(r, r.randint(3, 10))
                if r.random() < 0.10:      # things that look like skill use but are not
                    body.insert(0, sh("ls %s" % SKILL_DIR[agent]))
                    body.insert(1, call("Read", {"file_path": "docs/skills/README.md"}))
                if r.random() < 0.15:
                    cmd, err, code = r.choice(HARD_FAILS)
                    body.append(sh(cmd, False, err, code))
                items += [("call", c) for c in body]
                continue
            name, label = r.choice(names), r.choices(labels, weights)[0]
            body, inv_ok, follow = span_for(r, cfg, label, name, agent)
            how = "tool"
            if agent == "claude_code" and inv_ok and r.random() < 0.25:
                how = "slash"
                text = "<command-name>/%s</command-name>\n<command-args>%s</command-args>" % (name, text)
            elif agent == "codex" and r.random() < 0.4:
                text = "$%s %s" % (name, text)
            items.append(("user", text))
            if how == "tool":
                if agent == "claude_code":
                    inv = call("Skill", {"skill": name}, inv_ok, "" if inv_ok else "Unknown skill: %s" % name)
                else:
                    path = "%s/%s/SKILL.md" % (SKILL_DIR[agent], name)
                    inv = sh("cat " + path, inv_ok, "" if inv_ok else "cat: %s: No such file or directory" % path)
                items.append(("call", inv))
            items += [("call", c) for c in body]
            marks.append((name, label))
            if follow == "end":
                torn = r.random() < 0.5
                break
        if follow == "correct":
            pool = CORRECT_HELD if r.random() < cfg["held_phrases"] else CORRECT
            items.append(("user", r.choice(pool)))
            items += [("call", c) for c in noise(r, r.randint(1, 4))]
        sessions[sid] = {"agent": agent, "items": items, "torn": torn}
        truth[sid] = marks
    return sessions, truth

# ---------------------------------------------------------------- rendering

def render_claude(sid, s):
    """Claude Code transcript lines for one session."""
    out, n = [], 0
    for kind, v in s["items"]:
        if kind == "user":
            out.append({"type": "user", "sessionId": sid, "isSidechain": False,
                        "message": {"role": "user", "content": v}})
            continue
        n += 1
        tid = "toolu_%04d" % n
        out.append({"type": "assistant", "sessionId": sid, "isSidechain": False,
                    "message": {"role": "assistant", "content": [
                        {"type": "text", "text": "Working on it."},
                        {"type": "tool_use", "id": tid, "name": v["tool"], "input": v["input"]}]}})
        if not v["done"]:
            continue
        block = {"type": "tool_result", "tool_use_id": tid, "content": "ok"}
        if not v["ok"]:
            body = v["error"]
            if v["tool"] == "Bash":
                body = ("Exit code %d\n%s" % (v["code"], body)).strip()
            block.update(is_error=True, content=body)
        out.append({"type": "user", "sessionId": sid, "isSidechain": False,
                    "message": {"role": "user", "content": [block]}})
    lines = [json.dumps(x) for x in out]
    return lines + ['{"type":"assistant","sessionId":"%s","mess' % sid] if s["torn"] else lines


def render_codex(sid, s):
    """Codex rollout lines for one session."""
    ts = "2026-11-01T00:00:00.000Z"

    def line(kind, payload):
        return {"timestamp": ts, "type": kind, "payload": payload}

    out = [line("session_meta", {"id": sid, "timestamp": ts, "cwd": "/repo",
                                 "originator": "codex_cli_rs", "cli_version": "0.0.0-synthetic"}),
           line("response_item", {"type": "message", "role": "user", "content": [
               {"type": "input_text", "text": "<environment_context>\n  <cwd>/repo</cwd>\n</environment_context>"}]})]
    n = 0
    for kind, v in s["items"]:
        if kind == "user":
            out.append(line("response_item", {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": v}]}))
            out.append(line("event_msg", {"type": "user_message", "message": v}))
            continue
        n += 1
        cid = "call_%04d" % n
        tool, inp = v["tool"], v["input"]
        out.append(line("response_item", {"type": "reasoning", "summary": [], "encrypted_content": "x"}))
        custom = tool in ("Edit", "Write")
        if custom:
            verb = "Add" if tool == "Write" else "Update"
            patch = "*** Begin Patch\n*** %s File: %s\n@@\n-old\n+new\n*** End Patch" % (verb, inp["file_path"])
            out.append(line("response_item", {"type": "custom_tool_call", "status": "completed",
                                              "call_id": cid, "name": "apply_patch", "input": patch}))
        else:
            if tool == "Bash":
                name, args = "shell", {"command": ["bash", "-lc", inp["command"]], "workdir": "/repo"}
            elif tool == "Read":
                name, args = "shell", {"command": ["bash", "-lc", "cat " + inp["file_path"]]}
            elif tool in ("Grep", "Glob"):
                name, args = "shell", {"command": ["bash", "-lc", "rg -n %s" % inp.get("pattern", "x")]}
            else:
                name, args = {"TodoWrite": "update_plan", "WebSearch": "web_search"}.get(tool, tool), {}
            out.append(line("response_item", {"type": "function_call", "name": name,
                                              "arguments": json.dumps(args), "call_id": cid}))
        if not v["done"]:
            continue
        body = json.dumps({"output": "" if v["ok"] else v["error"],
                           "metadata": {"exit_code": 0 if v["ok"] else v["code"], "duration_seconds": 0.4}})
        out.append(line("response_item", {
            "type": "custom_tool_call_output" if custom else "function_call_output",
            "call_id": cid, "output": body}))
    lines = [json.dumps(x) for x in out]
    return lines + ['{"timestamp":"%s","type":"response_item","payl' % ts] if s["torn"] else lines


def render(sessions):
    """All sessions as one list of lines per agent."""
    out = {"claude_code": [], "codex": []}
    for sid, s in sessions.items():
        out[s["agent"]] += (render_claude if s["agent"] == "claude_code" else render_codex)(sid, s)
    return out


def parse_all(sessions):
    parsed = {}
    for sid, s in sessions.items():       # one file per session, as both agents store them
        lines = (render_claude if s["agent"] == "claude_code" else render_codex)(sid, s)
        for k, v in formats.parse(lines).items():
            parsed[k] = v
    return parsed

# ---------------------------------------------------------------- scoring

def sid_of(parsed_sid):
    return parsed_sid[len("codex:"):] if parsed_sid.startswith("codex:") else parsed_sid


def score(rows, truth):
    """Confusion counts keyed (truth, predicted), plus detection counts."""
    pred = {}
    for row in rows:
        pred.setdefault(sid_of(row["session"]), []).append(row)
    conf, missed, extra = {}, 0, 0
    for sid, marks in truth.items():
        got = pred.get(sid, [])
        for i, (name, label) in enumerate(marks):
            if i < len(got) and got[i]["skill"] == name:
                conf[(label, got[i]["verdict"])] = conf.get((label, got[i]["verdict"]), 0) + 1
            else:
                missed += 1
        extra += max(0, len(got) - len(marks))
    return conf, missed, extra


def metrics(conf):
    total = sum(conf.values())
    acc = sum(n for (t, p), n in conf.items() if t == p) / total if total else 0.0
    f1s, per = [], {}
    for v in se.VERDICTS:
        tp = conf.get((v, v), 0)
        fp = sum(n for (t, p), n in conf.items() if p == v and t != v)
        fn = sum(n for (t, p), n in conf.items() if t == v and p != v)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[v] = {"n": tp + fn, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
        f1s.append(f1)
    return {"n": total, "accuracy": round(acc, 3), "macro_f1": round(sum(f1s) / len(f1s), 3), "per_class": per}


def baseline_conf(rows, truth, window=10):
    """The first version's signal, as a two way verdict: any failed call in the next
    `window` calls after the invocation means not success."""
    pred = {}
    for row in rows:
        pred.setdefault(sid_of(row["session"]), []).append(row)
    hit = total = 0
    for sid, marks in truth.items():
        got = pred.get(sid, [])
        for i, (name, label) in enumerate(marks):
            if label == "unknown" or i >= len(got) or got[i]["skill"] != name:
                continue
            head = got[i]["inv"]["call"]
            failed = (head is not None and not head["ok"]) or any(not c["ok"] for c in got[i]["calls"][:window])
            total += 1
            hit += (label == "success") == (not failed)
    return hit, total


def binary_acc(conf):
    hit = sum(n for (t, p), n in conf.items() if t != "unknown" and (t == "success") == (p == "success"))
    total = sum(n for (t, p), n in conf.items() if t != "unknown")
    return hit, total


def run_config(cfg, **params):
    conf, missed, extra, planted = {}, 0, 0, 0
    base_hit = base_n = calls = 0
    rule_hit = rule_wrong = rule_planted = 0
    for seed in cfg["seeds"]:
        sessions, truth = generate(seed, cfg)
        parsed = parse_all(sessions)
        calls += sum(1 for s in parsed.values() for e in s["events"] if e["kind"] == "call")
        rows = se.evaluate(parsed, **params)
        c, m, x = score(rows, truth)
        for k, v in c.items():
            conf[k] = conf.get(k, 0) + v
        missed, extra = missed + m, extra + x
        planted += sum(len(v) for v in truth.values())
        h, n = baseline_conf(rows, truth)
        base_hit, base_n = base_hit + h, base_n + n
        rules, _ = se.skill_rules(rows)
        want = {(k, v[3]) for k, v in HABITS.items() if k in cfg["skills"]}
        got = {(k, (x["step"], x["fix"])) for k, rr in rules.items() if k for x in rr}
        rule_planted += len(want)
        rule_hit += len(want & got)
        rule_wrong += len(got - want)
    return {"conf": conf, "metrics": metrics(conf), "planted": planted, "missed": missed,
            "extra": extra, "calls": calls, "baseline": (base_hit, base_n),
            "binary": binary_acc(conf),
            "rules": {"planted": rule_planted, "attached_to_right_skill": rule_hit,
                      "attached_wrongly": rule_wrong}}


def tune():
    """Grid search on the tune setting only, by macro F1."""
    best = None
    for tail in (1, 2, 3, 4):
        for cov in (0.0, 0.2, 0.34, 0.5, 0.67):
            m = run_config(TUNE, tail=tail, min_coverage=cov)["metrics"]["macro_f1"]
            if best is None or m > best[0]:
                best = (m, tail, cov)
    return {"tail": best[1], "min_coverage": best[2]}


def show(name, res):
    m = res["metrics"]
    print("\n%s: %d calls, %d planted invocations, %d missed, %d extra" % (
        name, res["calls"], res["planted"], res["missed"], res["extra"]))
    print("accuracy %.3f, macro F1 %.3f over %d scored invocations" % (m["accuracy"], m["macro_f1"], m["n"]))
    print("| verdict | n | precision | recall | F1 |\n|---|---|---|---|---|")
    for v in se.VERDICTS:
        p = m["per_class"][v]
        print("| %s | %d | %.3f | %.3f | %.3f |" % (v, p["n"], p["precision"], p["recall"], p["f1"]))
    print("confusion (rows are truth, columns are predicted: %s)" % ", ".join(se.VERDICTS))
    for t in se.VERDICTS:
        print("  %-20s %s" % (t, " ".join("%4d" % res["conf"].get((t, p), 0) for p in se.VERDICTS)))
    bh, bn = res["baseline"]
    h, n = res["binary"]
    print("success or not, unknown left out: this %d/%d (%.3f), first version's error-in-next-10 signal %d/%d (%.3f)"
          % (h, n, h / n, bh, bn, bh / bn))
    print("per skill failure rules: %(planted)d planted, %(attached_to_right_skill)d attached to the right skill, "
          "%(attached_wrongly)d attached wrongly" % res["rules"])


def pack(res):
    return {"calls": res["calls"], "planted_invocations": res["planted"], "missed": res["missed"],
            "extra": res["extra"], "accuracy": res["metrics"]["accuracy"],
            "macro_f1": res["metrics"]["macro_f1"], "per_class": res["metrics"]["per_class"],
            "confusion": {"%s>%s" % k: v for k, v in sorted(res["conf"].items())},
            "binary_this": list(res["binary"]), "binary_first_version_signal": list(res["baseline"]),
            "skill_rules": res["rules"]}


def run():
    params = tune()
    print("thresholds picked on the tune setting: %s" % params)
    out = {"params": params}
    for cfg in (TUNE, HELD):
        res = run_config(cfg, **params)
        show(cfg["name"], res)
        out[cfg["name"]] = pack(res)
    path = "bench_results.json"
    data = {}
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
    data["skill_verdicts"] = out
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        lines = render(generate(TUNE["seeds"][0], TUNE)[0])
        for agent, ls in lines.items():
            with open(os.path.join(sys.argv[2], agent + ".jsonl"), "w") as fh:
                fh.write("\n".join(ls) + "\n")
    else:
        run()
