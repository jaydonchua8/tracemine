"""python test_tracemine.py"""
import unittest

import bench
import tracemine as tm


def bash(cmd, ok=True, error=""):
    return {"tool": "Bash", "input": {"command": cmd}, "ok": ok, "error": error}


def tool(name, path, ok=True, error=""):
    return {"tool": name, "input": {"file_path": path}, "ok": ok, "error": error}


class Load(unittest.TestCase):
    def test_transcript_shape(self):
        with open("sample_session.jsonl") as f:
            s = tm.load(f)
        calls = s["demo-1"]
        self.assertEqual([c["tool"] for c in calls],
                         ["Skill", "Bash", "Bash", "Edit", "Read", "Edit"])
        self.assertEqual([c["ok"] for c in calls], [True, False, True, False, True, True])
        self.assertIn("ModuleNotFoundError", calls[1]["error"])

    def test_flat_shape_and_bad_lines(self):
        s = tm.load(['{"session":"a","tool":"Bash","input":{"command":"ls"},"ok":false,"error":"x"}',
                     "", "{broken", "[1,2]"])
        self.assertEqual(len(s["a"]), 1)
        self.assertFalse(s["a"][0]["ok"])


class Normalise(unittest.TestCase):
    def test_step_keys(self):
        self.assertEqual(tm.step(bash("cd /repo && FOO=1 git commit -m 'x y'")), "sh:git commit")
        self.assertEqual(tm.step(bash("/usr/bin/pytest a/b.py -q")), "sh:pytest")
        self.assertEqual(tm.step(bash("python3 -m pytest a.py")), "sh:python -m pytest")
        self.assertEqual(tm.step(bash("gh pr view 12")), "sh:gh pr view")
        self.assertEqual(tm.step(bash("git -C x status")), "sh:git x")  # known wrong, see README
        self.assertEqual(tm.step(tool("Edit", "src/a.py")), "Edit:.py")
        self.assertEqual(tm.step({"tool": "Skill", "input": {"skill": "deploy"}}), "Skill:deploy")
        self.assertEqual(tm.step(bash("echo 'unterminated")), "sh:echo")

    def test_error_signature_groups_variants(self):
        a = tm.error_signature("Traceback\n  File \"a.py\", line 3\nModuleNotFoundError: No module named 'yaml'")
        b = tm.error_signature("Traceback\n  File \"zz.py\", line 77\nModuleNotFoundError: No module named 'attrs'")
        self.assertEqual(a, b)
        self.assertEqual(tm.error_signature(""), "(no message)")
        self.assertEqual(tm.error_signature("curl: (7) Failed to connect to localhost port 8080"),
                         tm.error_signature("curl: (7) Failed to connect to localhost port 3000"))

    def test_merge_signatures(self):
        m = tm.merge_signatures({"fatal: branch main has no upstream", "fatal: branch dev has no upstream",
                                 "fatal: remote origin already exists today"})
        self.assertEqual(m["fatal: branch main has no upstream"], "fatal: branch <*> has no upstream")
        self.assertEqual(m["fatal: branch dev has no upstream"], "fatal: branch <*> has no upstream")
        self.assertEqual(m["fatal: remote origin already exists today"],
                         "fatal: remote origin already exists today")


class Workflows(unittest.TestCase):
    def sessions(self):
        s = bench.generate(7, 200)
        return s

    def test_planted_found_and_noise_extension_rejected(self):
        found = {tuple(w["steps"]) for w in tm.mine_workflows(self.sessions())}
        for _, _, keys in bench.WORKFLOWS:
            self.assertIn(tuple(keys), found)
        planted = {tuple(k) for _, _, k in bench.WORKFLOWS}
        for g in found - planted:
            self.assertFalse(any(tm._contains(g, p) for p in planted), g)

    def test_failed_steps_not_in_workflows(self):
        s = {"s%d" % i: [bash("npm test", False, "Error: Cannot find module 'x'"),
                         bash("npm install"), bash("npm test")] for i in range(20)}
        self.assertEqual(tm.mine_workflows(s), [])

    def test_consecutive_repeats_collapse(self):
        self.assertEqual(tm.collapse(["a", "a", "b", "a", "a"]), ["a", "b", "a"])

    def test_poisson_tail(self):
        self.assertAlmostEqual(tm.poisson_tail(1, 1.0), 0.6321, places=3)
        self.assertLess(tm.poisson_tail(9, 3.0), 0.01)
        self.assertGreater(tm.poisson_tail(9, 3.0), 1e-4)


class Failures(unittest.TestCase):
    def test_changed_command_fix(self):
        s = {"s%d" % i: [bash("pytest t%d.py" % i, False, "ModuleNotFoundError: No module named 'm%d'" % i),
                         bash("git status"), bash("python -m pytest t%d.py" % i)] for i in range(5)}
        rules, unresolved = tm.mine_failures(s)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["fix"], "prefix the command with `python -m`")
        self.assertEqual(rules[0]["sessions"], 5)
        self.assertEqual(rules[0]["agreement"], 1.0)
        self.assertEqual(unresolved, [])

    def test_step_inserted_fix(self):
        s = {"s%d" % i: [tool("Edit", "a%d.py" % i, False, "String to replace not found in file."),
                         tool("Read", "a%d.py" % i), tool("Edit", "a%d.py" % i)] for i in range(4)}
        rules, _ = tm.mine_failures(s)
        self.assertEqual(rules[0]["fix"], "first run `Read:.py`, then retry")

    def test_value_taken_from_error(self):
        s = {}
        for i, b in enumerate(["main", "dev", "fix-a", "fix-b"]):
            s["s%d" % i] = [bash("git push", False, "fatal: The current branch %s has no upstream branch." % b),
                            bash("git push -u origin %s" % b)]
        rules, _ = tm.mine_failures(s)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["fix"], "add `-u origin <from-error>`")

    def test_inconsistent_fix_is_not_a_rule(self):
        fixes = ["mypy src --ignore-missing-imports", "mypy src --follow-imports=skip",
                 "mypy src --strict", "mypy src -v"]
        s = {"s%d" % i: [bash("mypy src", False, "error: Cannot find stub"), bash(fixes[i % 4])]
             for i in range(8)}
        rules, unresolved = tm.mine_failures(s)
        self.assertEqual(rules, [])
        self.assertEqual(unresolved[0]["recovered"], 8)

    def test_fix_outside_window_or_other_file_ignored(self):
        s = {"s%d" % i: [tool("Edit", "a.py", False, "String to replace not found in file."),
                         tool("Read", "b.py"), tool("Edit", "b.py")] for i in range(4)}
        rules, unresolved = tm.mine_failures(s)
        self.assertEqual(rules, [])
        self.assertEqual(unresolved[0]["recovered"], 0)

    def test_below_support_dropped(self):
        s = {"s0": [bash("ls", False, "boom"), bash("ls")]}
        self.assertEqual(tm.mine_failures(s), ([], []))


class SkillsAndProposals(unittest.TestCase):
    def test_skill_stats(self):
        skill = {"tool": "Skill", "input": {"skill": "deploy"}, "ok": True, "error": ""}
        s = {"a": [skill, bash("ls", False, "x"), bash("ls")], "b": [skill, bash("ls")],
             "c": [bash("ls")]}
        rows, base = tm.skill_stats(s)
        self.assertEqual(rows, [{"skill": "deploy", "invocations": 2, "sessions": 2,
                                 "error_rate_after": round(1 / 3, 3)}])
        self.assertEqual(base, round(1 / 6, 3))

    def test_propose_is_a_pure_append_diff(self):
        w = [{"steps": ["a", "b", "c"], "sessions": 9, "examples": ["s1"]}]
        r = [{"step": "sh:pytest", "error": "E", "fix": "prefix the command with `python -m`",
              "sessions": 5, "agreement": 1.0, "examples": ["s2"]}]
        d = tm.propose("# Skill\n\nBody.\n", w, r)
        lines = d.splitlines()
        self.assertTrue(lines[0].startswith("--- a/SKILL.md"))
        body = [l for l in lines[3:]]
        self.assertFalse(any(l.startswith("-") for l in body))
        self.assertTrue(any("python -m" in l and "5 sessions" in l for l in body))
        self.assertTrue(any("`a` -> `b` -> `c`" in l for l in body))
        self.assertEqual(tm.propose("# Skill\n", [], []), "")


if __name__ == "__main__":
    unittest.main()
