"""python test_skilleval.py"""
import unittest

import bench_skills as bs
import formats
import skilleval as se
import tracemine as tm


def call(tool, inp, ok=True, error="", done=True, **kw):
    return dict({"kind": "call", "tool": tool, "input": inp, "ok": ok, "error": error,
                 "done": done, "agent": "t"}, **kw)


def sh(cmd, ok=True, error="", **kw):
    return call("Bash", {"command": cmd}, ok, error, **kw)


def user(text):
    return {"kind": "user", "text": text}


def skill(name="deploy", ok=True, error=""):
    return call("Skill", {"skill": name}, ok, error)


def sess(*events, truncated=False):
    return {"agent": "t", "events": list(events), "truncated": truncated, "bad_lines": 0}


def one(*events, truncated=False, **params):
    rows = se.evaluate({"s": sess(*events, truncated=truncated)}, **params)
    assert len(rows) == 1, rows
    return rows[0]["verdict"]


class Invocations(unittest.TestCase):
    def test_signals(self):
        s = sess(user("<command-name>/deploy</command-name>"), sh("ls"),
                 user("go"), skill("review"),
                 user("go"), call("Read", {"file_path": "/r/.agents/skills/lint-fix/SKILL.md"}),
                 user("<skill>\n<name>notes</name>\n<path>x</path>\nbody</skill>"),
                 user("$review this please"))
        got = [(i["skill"], i["how"]) for i in se.invocations(s, known={"review"})]
        self.assertEqual(got, [("deploy", "slash"), ("review", "tool"), ("lint-fix", "read"),
                               ("notes", "tag"), ("review", "mention")])

    def test_things_that_are_not_invocations(self):
        s = sess(user("<command-name>/clear</command-name>"), user("check $HOME and $deploy-env"),
                 sh("ls .claude/skills"), call("Read", {"file_path": "docs/skills/README.md"}),
                 call("Read", {"file_path": ".claude/skills/deploy/scripts/run.py"}),
                 sh("cat .claude/skills/deploy/SKILL.md | head"))
        self.assertEqual(se.invocations(s, known={"deploy"}), [])

    def test_signals_in_one_turn_merge_and_keep_the_call(self):
        read = call("Read", {"file_path": ".agents/skills/deploy/SKILL.md"})
        s = sess(user("$deploy ship it"), sh("ls"), read, read, user("again"), read)
        got = se.invocations(s, known={"deploy"})
        self.assertEqual([(i["how"], i["index"]) for i in got], [("mention", 2), ("read", 5)])
        self.assertIs(got[0]["call"], read)

    def test_span_stops_at_user_next_skill_and_cap(self):
        s = sess(skill("a"), sh("ls"), skill("b"), sh("ls"), sh("ls"), sh("ls"), user("ok"), sh("pwd"))
        a, b = se.invocations(s)
        self.assertEqual([len(x) for x in se.span(s, a, b["index"])[:1]], [1])
        self.assertEqual(se.span(s, a, b["index"])[2], "skill")
        calls, nxt, ended = se.span(s, b)
        self.assertEqual((len(calls), nxt, ended), (3, "ok", "user"))
        self.assertEqual(se.span(s, b, max_calls=2)[2], "cap")


class Verdicts(unittest.TestCase):
    def test_success_with_recovered_failure(self):
        self.assertEqual(one(skill(), sh("pytest a.py", False, "ModuleNotFoundError: x"),
                             sh("python -m pytest a.py"), sh("git status"), user("thanks")), "success")

    def test_skill_error_when_load_fails(self):
        self.assertEqual(one(skill(ok=False, error="Unknown skill: deploy"), sh("ls")), "skill_error")

    def test_skill_error_when_its_script_fails(self):
        bad = sh("bash .claude/skills/deploy/scripts/check.sh", False, "check.sh: line 3: jq: command not found")
        self.assertEqual(one(skill(), bad, sh("ls"), sh("ls"), sh("ls"), sh("ls")), "skill_error")
        # the same script failing under another skill's invocation is not that skill's error
        self.assertEqual(one(skill("other"), bad, sh("ls"), sh("ls"), sh("ls"), sh("ls")), "success")
        # and a script that works on the second try is not an error
        self.assertEqual(one(skill(), bad, sh("bash .claude/skills/deploy/scripts/check.sh")), "success")

    def test_other_error_needs_the_failure_near_the_end(self):
        bad = sh("git push", False, "error: failed to push some refs")
        self.assertEqual(one(skill(), sh("ls"), bad), "other_error")
        self.assertEqual(one(skill(), sh("ls"), bad, sh("git status")), "other_error")
        self.assertEqual(one(skill(), bad, sh("ls"), sh("pwd"), sh("git status")), "success")
        self.assertEqual(one(skill(), bad, sh("ls"), sh("pwd"), sh("git status"), tail=4), "other_error")

    def test_exit_status_one_from_grep_is_not_a_failure(self):
        self.assertEqual(one(skill(), sh("grep -rn TODO src", False, "", exit=1)), "success")
        self.assertEqual(one(skill(), sh("grep -rn TODO src", False, "Exit code 1")), "success")
        self.assertEqual(one(skill(), sh("grep -rn TODO src", False, "grep: src: No such file or directory", exit=2)),
                         "other_error")
        self.assertEqual(one(skill(), sh("ls nope", False, "", exit=1)), "other_error")

    def test_unexpected_when_user_corrects(self):
        self.assertEqual(one(skill(), sh("ls"), user("no, that's not what I asked for")), "unexpected_behavior")
        self.assertEqual(one(skill(), sh("ls"), user("no rush, next rename the helper")), "success")
        self.assertEqual(one(skill(), sh("ls"), user("I didn't ask you to push")), "unexpected_behavior")
        self.assertEqual(one(skill(), sh("ls"), user("looks good. also add a test")), "success")

    def test_unexpected_when_usual_steps_are_skipped(self):
        usual = [skill(), sh("npm install"), sh("npm test"), sh("git push"), user("next")]
        events = usual * 6 + [skill(), sh("ls"), sh("pwd"), user("next")]
        rows = se.evaluate({"s": sess(*events)})
        self.assertEqual([r["verdict"] for r in rows], ["success"] * 6 + ["unexpected_behavior"])
        self.assertIn("0%", rows[-1]["reason"])
        # with too few invocations there is no profile, so nothing is flagged
        rows = se.evaluate({"s": sess(*(usual * 2 + [skill(), sh("ls"), user("next")]))})
        self.assertEqual([r["verdict"] for r in rows], ["success"] * 3)

    def test_unknown_when_cut_off(self):
        self.assertEqual(one(skill(), sh("ls"), sh("npm test", done=False)), "unknown")
        self.assertEqual(one(skill(), sh("ls"), truncated=True), "unknown")
        # a pending call in a later turn does not make an earlier invocation unknown
        self.assertEqual(one(skill(), sh("ls"), user("next"), sh("npm test", done=False)), "success")
        # a session that simply ends after finished calls is evaluated
        self.assertEqual(one(skill(), sh("ls")), "success")

    def test_empty_inputs(self):
        self.assertEqual(se.evaluate({}), [])
        self.assertEqual(se.evaluate({"s": sess()}), [])
        self.assertEqual(se.analytics([]), [])
        rep = se.report({"s": sess(user("hi"), sh("ls"))})
        self.assertEqual((rep["rows"], rep["diffs"]), ([], {}))
        self.assertEqual(one(user("<command-name>/deploy</command-name>")), "success")


class AnalyticsAndRules(unittest.TestCase):
    def test_success_rate_leaves_unknown_out(self):
        s = {"a": sess(skill(), sh("ls"), user("x"), skill(), sh("git push", False, "error: rejected")),
             "b": sess(skill(), sh("npm test", done=False))}
        (a,) = se.analytics(se.evaluate(s))
        self.assertEqual((a["invocations"], a["evaluated"], a["sessions"], a["success_rate"]), (3, 2, 2, 0.5))
        self.assertEqual(a["verdicts"]["unknown"], 1)

    def test_rule_goes_to_the_skill_it_happens_under(self):
        habit = [sh("alembic upgrade head", False, "OperationalError: connection refused"),
                 sh("docker compose up -d db"), sh("alembic upgrade head")]
        stale = [call("Edit", {"file_path": "a.py"}, False, "String to replace not found in file."),
                 call("Read", {"file_path": "a.py"}), call("Edit", {"file_path": "a.py"})]
        s = {}
        for i in range(5):
            s["m%d" % i] = sess(skill("migrate"), *(habit + stale))
            s["r%d" % i] = sess(skill("review"), sh("gh pr view 3"), *stale)
        rep = se.report(s, {"migrate": "# migrate\n"})
        self.assertEqual([(x["step"], x["fix"]) for x in rep["rules"]["migrate"]],
                         [("sh:alembic", "first run `sh:docker compose up`, then retry")])
        self.assertNotIn("review", rep["rules"])
        self.assertEqual([x["step"] for x in rep["rules"][None]], ["Edit:.py"])
        self.assertEqual(list(rep["diffs"]), ["migrate"])
        diff = rep["diffs"]["migrate"]
        self.assertIn("+++ b/skills/migrate/SKILL.md", diff)
        self.assertIn("docker compose up", diff)
        self.assertNotIn("String to replace", diff)
        self.assertFalse(any(l.startswith("-") and not l.startswith("---") for l in diff.splitlines()))


class EndToEnd(unittest.TestCase):
    def test_every_planted_invocation_is_found_in_both_formats(self):
        for cfg in (bs.TUNE, bs.HELD):
            sessions, truth = bs.generate(3, dict(cfg, sessions=80))
            rows = se.evaluate(bs.parse_all(sessions))
            conf, missed, extra = bs.score(rows, truth)
            self.assertEqual((missed, extra), (0, 0))
            self.assertEqual(sum(conf.values()), sum(len(v) for v in truth.values()))
            self.assertEqual({r["agent"] for r in rows}, {"claude_code", "codex"})

    def test_held_out_skills_and_phrases_are_disjoint_from_tune(self):
        self.assertFalse(set(bs.TUNE_SKILLS) & set(bs.HELD_SKILLS))
        self.assertFalse(set(bs.TUNE["seeds"]) & set(bs.HELD["seeds"]))
        self.assertFalse(set(bs.CORRECT) & set(bs.CORRECT_HELD))

    def test_tune_setting_floor(self):
        res = bs.run_config(dict(bs.TUNE, seeds=[21], sessions=150))
        self.assertGreater(res["metrics"]["accuracy"], 0.9)
        self.assertEqual(res["rules"]["attached_wrongly"], 0)

    def test_cli_on_fixtures(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = tm.main(["skills", "sample_codex_rollout.jsonl", "sample_session.jsonl"])
        self.assertEqual(code, 0)
        self.assertIn("db-migrate", buf.getvalue())
        with open("sample_codex_rollout.jsonl") as f:
            (row,) = se.evaluate(formats.parse(f))
        self.assertEqual((row["skill"], row["verdict"], len(row["calls"])), ("db-migrate", "success", 7))


if __name__ == "__main__":
    unittest.main()
