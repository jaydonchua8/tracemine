"""python demo.py

Generates 150 synthetic sessions (labelled synthetic, some written as Claude Code
transcripts and some as Codex rollouts), reads them back through formats.py, and prints
skill analytics, a few verdicts with their reasons, and the edit proposed for one skill.
Runs in a few seconds and writes nothing.
"""
from collections import Counter

import bench_skills as bs
import skilleval as se
import tracemine as tm

SKILL_MD = """---
name: db-migrate
description: Add a database migration and apply it locally.
---

1. Read db/schema.sql.
2. Create a revision with alembic.
3. Edit the migration.
4. Run `alembic upgrade head`.
5. Run the database tests.
"""


def main():
    cfg = dict(bs.TUNE, sessions=150)
    sessions, truth = bs.generate(11, cfg)
    parsed = bs.parse_all(sessions)
    agents = Counter(s["agent"] for s in parsed.values())
    calls = sum(1 for s in parsed.values() for e in s["events"] if e["kind"] == "call")
    print("synthetic input: %d sessions (%s), %d tool calls" % (
        len(parsed), ", ".join("%d %s" % (n, a) for a, n in sorted(agents.items())), calls))

    rep = se.report(parsed, {"db-migrate": SKILL_MD}, tail=2, min_coverage=0.5)
    print("\nskill analytics (success rate is over evaluated invocations)")
    print("%-15s %5s %5s %8s  %-22s %s" % ("skill", "uses", "eval", "success", "agents", "not success"))
    for a in rep["analytics"]:
        other = ", ".join("%s %d" % (v, n) for v, n in a["verdicts"].items() if n and v != "success")
        ag = ", ".join("%s %d" % (k, v) for k, v in sorted(a["agents"].items()))
        print("%-15s %5d %5d %7.0f%%  %-22s %s" % (
            a["skill"], a["invocations"], a["evaluated"], 100 * a["success_rate"], ag, other))

    print("\none example per verdict")
    seen = set()
    for r in rep["rows"]:
        if r["verdict"] != "success" and r["verdict"] not in seen:
            seen.add(r["verdict"])
            print("  %-20s %s / %s (%s)\n  %20s %s" % (
                r["verdict"], r["session"], r["skill"], r["agent"], "", r["reason"]))

    conf, missed, extra = bs.score(rep["rows"], truth)
    m = bs.metrics(conf)
    print("\nagainst the generator's labels: %d invocations, %d missed, %d extra, accuracy %.3f"
          % (m["n"], missed, extra, m["accuracy"]))

    print("\nfailure rules that belong to one skill")
    for skill, rules in sorted((k, v) for k, v in rep["rules"].items() if k):
        for x in rules:
            print("  %-12s %s | %s | %s (%d invocations, %d%% of this failure is under this skill)" % (
                skill, x["step"], x["error"][:48], x["fix"], x["sessions"], round(100 * x["share"])))
    print("failure rules seen under several skills (not attached to any): %d" % len(rep["rules"][None]))

    print("\nproposed edit, for review (nothing is written)\n")
    print(rep["diffs"].get("db-migrate", "(none)"))

    both = tm.mine_workflows(tm.formats.calls_only(parsed))
    print("\nworkflows mined across both agents: %d, top: %s" % (
        len(both), " -> ".join(both[0]["steps"]) if both else "-"))


if __name__ == "__main__":
    main()
