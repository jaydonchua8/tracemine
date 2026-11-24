"""Generate synthetic agent traces with planted workflows and failure/fix habits,
then check how many of them tracemine recovers and how much noise it reports.

python bench.py            run the benchmark, print tables, write bench_results.json
python bench.py --dump F   write one generated trace set to F as JSONL
"""
import json
import os
import random
import sys
import time

import tracemine as tm

SESSIONS = 400
SEEDS = [1, 2, 3, 4, 5]

FILES = ["src/app", "src/db", "lib/util", "api/routes", "core/model", "tests/test_api", "cli/main"]
WORDS = ["retry", "token", "parse", "cache", "index", "user"]
BRANCHES = ["fix-auth", "feat-search", "bump-deps", "refactor-db", "ingest-v2"]
PKGS = ["yaml", "requests", "numpy", "attrs", "httpx"]


def f(r):
    return r.choice(FILES)


def bash(cmd):
    return ("Bash", {"command": cmd})


# Background calls, most common first. Weights fall off as 1 / rank^0.9.
NOISE = [
    lambda r: ("Read", {"file_path": f(r) + ".py"}),
    lambda r: ("Edit", {"file_path": f(r) + ".py"}),
    lambda r: bash("git status"),
    lambda r: ("Grep", {"pattern": r.choice(WORDS)}),
    lambda r: bash("pytest %s.py -q" % f(r)),
    lambda r: ("Read", {"file_path": f(r) + ".ts"}),
    lambda r: bash("git diff"),
    lambda r: ("Edit", {"file_path": f(r) + ".ts"}),
    lambda r: bash("ls %s" % f(r).split("/")[0]),
    lambda r: ("Glob", {"pattern": "**/*.py"}),
    lambda r: bash("npm test"),
    lambda r: bash("git add %s.py" % f(r)),
    lambda r: bash("git commit -m 'update %s'" % r.choice(WORDS)),
    lambda r: ("Read", {"file_path": f(r) + ".md"}),
    lambda r: bash("cat %s.json" % f(r)),
    lambda r: ("TodoWrite", {}),
    lambda r: bash("rg %s %s" % (r.choice(WORDS), f(r).split("/")[0])),
    lambda r: bash("git log -n %d" % r.randint(3, 30)),
    lambda r: bash("npm run build"),
    lambda r: ("Write", {"file_path": f(r) + ".py"}),
    lambda r: bash("python %s.py" % f(r)),
    lambda r: bash("ruff check ."),
    lambda r: bash("npx tsc --noEmit"),
    lambda r: bash("git push"),
    lambda r: bash("npm install"),
    lambda r: bash("cargo build"),
    lambda r: bash("cargo test"),
    lambda r: bash("gh pr view %d" % r.randint(10, 900)),
    lambda r: ("WebSearch", {"query": r.choice(WORDS)}),
    lambda r: bash("git checkout %s" % r.choice(BRANCHES)),
    lambda r: bash("mypy %s" % f(r).split("/")[0]),
    lambda r: bash("make %s" % r.choice(["lint", "fmt", "dev"])),
    lambda r: bash("docker ps"),
    lambda r: bash("curl -s http://localhost:%d/health" % r.randint(3000, 9000)),
    lambda r: bash("pip install %s" % r.choice(PKGS)),
    lambda r: bash("go test ./..."),
]
WEIGHTS = [1 / (i + 1) ** 0.9 for i in range(len(NOISE))]

# (probability a session contains it, steps, expected mined keys)
WORKFLOWS = [
    (0.12, [lambda r: bash("git checkout -b %s" % r.choice(BRANCHES)),
            lambda r: ("Edit", {"file_path": f(r) + ".py"}),
            lambda r: bash("ruff check ."),
            lambda r: bash("pytest %s.py -q" % f(r)),
            lambda r: bash("git commit -m 'update %s'" % r.choice(WORDS))],
     ["sh:git checkout", "Edit:.py", "sh:ruff", "sh:pytest", "sh:git commit"]),
    (0.15, [lambda r: bash("npm install"), lambda r: bash("npm run build"),
            lambda r: bash("npm test")],
     ["sh:npm install", "sh:npm run", "sh:npm test"]),
    (0.08, [lambda r: bash("gh pr view %d" % r.randint(10, 900)), lambda r: bash("git diff"),
            lambda r: ("Read", {"file_path": f(r) + ".md"})],
     ["sh:gh pr view", "sh:git diff", "Read:.md"]),
    (0.10, [lambda r: ("Grep", {"pattern": r.choice(WORDS)}),
            lambda r: ("Read", {"file_path": f(r) + ".ts"}),
            lambda r: ("Edit", {"file_path": f(r) + ".ts"}),
            lambda r: bash("npx tsc --noEmit")],
     ["Grep", "Read:.ts", "Edit:.ts", "sh:npx tsc"]),
    (0.06, [lambda r: bash("docker ps"), lambda r: bash("docker logs api-%d" % r.randint(1, 9)),
            lambda r: bash("docker restart api-%d" % r.randint(1, 9))],
     ["sh:docker ps", "sh:docker logs", "sh:docker restart"]),
    (0.05, [lambda r: bash("cargo build"), lambda r: bash("cargo test"),
            lambda r: bash("git add %s.py" % f(r)),
            lambda r: bash("git commit -m 'update %s'" % r.choice(WORDS)),
            lambda r: bash("git push")],
     ["sh:cargo build", "sh:cargo test", "sh:git add", "sh:git commit", "sh:git push"]),
]


def call(tool_input, ok=True, error=""):
    return {"tool": tool_input[0], "input": tool_input[1], "ok": ok, "error": error}


def noise(r, n):
    return [call(r.choices(NOISE, WEIGHTS)[0](r)) for _ in range(n)]


def fail_pytest_module(r):
    p, pkg = f(r), r.choice(PKGS)
    err = "Traceback (most recent call last):\n  File \"%s.py\", line %d\nModuleNotFoundError: No module named '%s'" % (p, r.randint(1, 300), pkg)
    return ([call(bash("pytest %s.py -q" % p), False, err)] + noise(r, r.randint(0, 2)) +
            [call(bash("python -m pytest %s.py -q" % p))])


def fail_npm_module(r):
    err = "Error: Cannot find module '%s'\nRequire stack:\n- /app/%s.js" % (r.choice(WORDS), f(r))
    return [call(bash("npm test"), False, err), call(bash("npm install")), call(bash("npm test"))]


def fail_edit_stale(r):
    p = f(r) + ".py"
    return [call(("Edit", {"file_path": p}), False, "String to replace not found in file."),
            call(("Read", {"file_path": p})), call(("Edit", {"file_path": p}))]


def fail_push_upstream(r):
    b = r.choice(BRANCHES)
    err = "fatal: The current branch %s has no upstream branch.\nTo push the current branch and set the remote as upstream, use" % b
    return ([call(bash("git push"), False, err)] + noise(r, r.randint(0, 1)) +
            [call(bash("git push -u origin %s" % b))])


def fail_python_missing(r):
    p = f(r)
    return ([call(bash("python %s.py" % p), False, "bash: python: command not found")] +
            noise(r, r.randint(0, 2)) + [call(bash("python3 %s.py" % p))])


def fail_mypy_inconsistent(r):
    """A recurring failure that people get past in four different ways. Should not become a rule."""
    d = f(r).split("/")[0]
    err = "error: Cannot find implementation or library stub for module named \"%s\"" % r.choice(PKGS)
    fix = r.choice([
        [call(bash("mypy %s --ignore-missing-imports" % d))],
        [call(bash("pip install types-%s" % r.choice(PKGS))), call(bash("mypy %s" % d))],
        [call(("Edit", {"file_path": "mypy.ini"})), call(bash("mypy %s" % d))],
        noise(r, 5)])
    return [call(bash("mypy %s" % d), False, err)] + fix


def fail_curl_unrecovered(r):
    err = "curl: (7) Failed to connect to localhost port %d: Connection refused" % r.randint(3000, 9000)
    return [call(bash("curl -s http://localhost:%d/health" % r.randint(3000, 9000)), False, err)]


# (probability, generator, expected (step, fix label) or None for distractors)
FAILURES = [
    (0.10, fail_pytest_module, ("sh:pytest", "prefix the command with `python -m`")),
    (0.08, fail_npm_module, ("sh:npm test", "first run `sh:npm install`, then retry")),
    (0.12, fail_edit_stale, ("Edit:.py", "first run `Read:.py`, then retry")),
    (0.06, fail_push_upstream, ("sh:git push", "add `-u origin <from-error>`")),
    (0.07, fail_python_missing, ("sh:python", "replace `python` with `python3`")),
    (0.10, fail_mypy_inconsistent, None),
    (0.08, fail_curl_unrecovered, None),
]


def generate(seed, n_sessions=SESSIONS):
    r = random.Random(seed)
    sessions = {}
    for i in range(n_sessions):
        chunks = [noise(r, r.randint(2, 8)) for _ in range(r.randint(4, 9))]
        for p, steps, _ in WORKFLOWS:
            if r.random() < p:
                chunks.insert(r.randrange(len(chunks) + 1), [call(s(r)) for s in steps])
        for p, gen, _ in FAILURES:
            if r.random() < p:
                chunks.insert(r.randrange(len(chunks) + 1), gen(r))
        calls = [c for ch in chunks for c in ch]
        for c in calls:  # one-off failures with nothing consistent after them
            if c["ok"] and c["tool"] == "Bash" and r.random() < 0.02:
                c["ok"], c["error"] = False, "exit code %d" % r.randint(1, 127)
        sessions["s%03d" % i] = calls
    return sessions


def score_workflows(found, planted):
    got = {tuple(w["steps"]) for w in found}
    hit = got & planted
    partial = {g for g in got - planted
               if any(tm._contains(p, g) or tm._contains(g, p) for p in planted)}
    return len(hit), len(got), len(partial)


def run():
    planted_w = {tuple(k) for _, _, k in WORKFLOWS}
    planted_f = {k for _, _, k in FAILURES if k}
    rows = {"support only": [0, 0, 0], "with chance test": [0, 0, 0]}
    f_hit = f_got = unresolved_n = calls_n = 0
    secs = 0.0
    for seed in SEEDS:
        sessions = generate(seed)
        calls_n += sum(len(c) for c in sessions.values())
        t = time.perf_counter()
        found = tm.mine_workflows(sessions)
        rules, unresolved = tm.mine_failures(sessions)
        secs += time.perf_counter() - t
        base = tm.mine_workflows(sessions, max_p=1)
        for name, res in (("support only", base), ("with chance test", found)):
            for i, v in enumerate(score_workflows(res, planted_w)):
                rows[name][i] += v
        got = {(x["step"], x["fix"]) for x in rules}
        f_hit += len(got & planted_f)
        f_got += len(got)
        for x in sorted(got - planted_f):
            print("  seed %d unplanted rule: %s | %s" % (seed, x[0], x[1]))
        for x in sorted(planted_f - got):
            print("  seed %d missed rule: %s | %s" % (seed, x[0], x[1]))
        unresolved_n += len(unresolved)
    n_w, n_f = len(planted_w) * len(SEEDS), len(planted_f) * len(SEEDS)
    out = {"seeds": len(SEEDS), "sessions_per_seed": SESSIONS, "calls": calls_n,
           "workflows": {}, "failures": {}}
    print("%d seeds x %d sessions, %d calls total" % (len(SEEDS), SESSIONS, calls_n))
    print("\n| workflow mining | planted | recovered | reported | precision | partial overlaps |")
    print("|---|---|---|---|---|---|")
    for name, (hit, got, partial) in rows.items():
        print("| %s | %d | %d | %d | %.1f%% | %d |" % (
            name, n_w, hit, got, 100 * hit / got if got else 0, partial))
        out["workflows"][name] = {"planted": n_w, "recovered": hit, "reported": got,
                                  "partial_overlaps": partial}
    print("\n| failure rules | planted | recovered | reported | wrong rules |")
    print("|---|---|---|---|---|")
    print("| | %d | %d | %d | %d |" % (n_f, f_hit, f_got, f_got - f_hit))
    print("\nrecurring failures listed as having no consistent fix: %d (2 planted per seed)"
          % unresolved_n)
    print("mining time: %.2f s for %d calls (%.0f calls/s)" % (secs, calls_n, calls_n / secs))
    out["failures"] = {"planted": n_f, "recovered": f_hit, "reported": f_got,
                       "unresolved_listed": unresolved_n}
    out["mining_seconds"] = round(secs, 2)
    out["calls_per_second"] = round(calls_n / secs)
    data = {}
    if os.path.exists("bench_results.json"):
        with open("bench_results.json") as fh:
            data = json.load(fh)
    data = {k: v for k, v in data.items() if k == "skill_verdicts"}   # kept from bench_skills.py
    data["workflow_mining"] = out
    with open("bench_results.json", "w") as fh:
        json.dump(data, fh, indent=1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        with open(sys.argv[2], "w") as fh:
            for sid, calls in generate(1).items():
                for c in calls:
                    fh.write(json.dumps(dict(c, session=sid)) + "\n")
    else:
        run()
