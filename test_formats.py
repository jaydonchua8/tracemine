"""python test_formats.py"""
import json
import unittest

import bench_skills as bs
import formats
import tracemine as tm


def cx(kind, payload):
    return json.dumps({"timestamp": "t", "type": kind, "payload": payload})


def fcall(cid, cmd, name="shell"):
    return cx("response_item", {"type": "function_call", "name": name, "call_id": cid,
                                "arguments": json.dumps({"command": cmd})})


def fout(cid, output):
    return cx("response_item", {"type": "function_call_output", "call_id": cid, "output": output})


def calls(lines):
    (s,) = formats.parse(lines).values()
    return [e for e in s["events"] if e["kind"] == "call"]


class Codex(unittest.TestCase):
    def test_fixture(self):
        with open("sample_codex_rollout.jsonl") as f:
            parsed = formats.parse(f)
        s = parsed["codex:demo-codex-1"]
        self.assertEqual(s["agent"], "codex")
        ev = s["events"]
        # environment_context is dropped, and the user message logged twice is kept once
        self.assertEqual([e["text"] for e in ev if e["kind"] == "user"],
                         ["$db-migrate add a last_login column to users",
                          "thanks, now update the README to mention it"])
        c = [e for e in ev if e["kind"] == "call"]
        self.assertEqual([tm.step(x) for x in c],
                         ["Read:.md", "sh:alembic", "Edit:.py", "sh:alembic", "sh:docker compose up",
                          "sh:alembic", "sh:grep", "sh:pytest", "Read:.md"])
        self.assertEqual([x["ok"] for x in c], [True, True, True, False, True, True, False, True, True])
        self.assertIn("OperationalError", c[3]["error"])
        self.assertEqual(c[6]["exit"], 1)
        self.assertEqual(c[6]["error"], "Exit code 1")      # headers stripped, nothing left
        self.assertFalse(c[8]["done"])                       # no output line for the last call

    def test_command_shapes(self):
        got = calls([fcall("a", ["bash", "-lc", "git status"]), fcall("b", "npm test", "shell_command"),
                     fcall("c", ["pytest", "a b.py"]),
                     cx("response_item", {"type": "function_call", "name": "exec_command", "call_id": "d",
                                          "arguments": json.dumps({"cmd": "cargo build"})})])
        self.assertEqual([c["input"]["command"] for c in got],
                         ["git status", "npm test", "pytest 'a b.py'", "cargo build"])

    def test_failure_is_read_from_the_body(self):
        got = calls([fcall("a", "make"), fout("a", json.dumps({"output": "boom", "metadata": {"exit_code": 2}})),
                     fcall("b", "make"), fout("b", "Exit code: 2\nWall time: 1 seconds\nOutput:\nboom"),
                     fcall("c", "make"), fout("c", "all good, 0 errors"),
                     fcall("d", "make"), fout("d", [{"type": "input_text", "text": "Process exited with code 3\nbad"}]),
                     fcall("e", "make"), fout("e", {"content": "nope", "success": False})])
        self.assertEqual([c["ok"] for c in got], [False, False, True, False, False])
        self.assertEqual([c.get("exit") for c in got], [2, 2, None, 3, 1])
        self.assertEqual([c["error"] for c in got], ["boom", "boom", "", "bad", "nope"])

    def test_apply_patch_and_print_commands(self):
        patch = "*** Begin Patch\n*** Add File: src/new.py\n+x\n*** End Patch"
        got = calls([cx("response_item", {"type": "custom_tool_call", "name": "apply_patch",
                                          "call_id": "a", "input": patch}),
                     fcall("b", ["bash", "-lc", "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: a.ts\n*** End Patch\nEOF"]),
                     fcall("c", "sed -n 1,80p src/app.py"), fcall("d", "cat a.py | wc -l"),
                     fcall("e", "cat a.py b.py"), fcall("f", "head -n 5 README.md")])
        self.assertEqual([(c["tool"], c["input"].get("file_path")) for c in got],
                         [("Write", "src/new.py"), ("Edit", "a.ts"), ("Read", "src/app.py"),
                          ("Bash", None), ("Bash", None), ("Read", "README.md")])

    def test_bad_input_does_not_raise(self):
        lines = [cx("response_item", {"type": "function_call", "name": "shell", "call_id": "a",
                                      "arguments": "{not json"}),
                 fout("zzz", "orphan output"), cx("response_item", {"type": "function_call_output"}),
                 cx("event_msg", {"type": "exec_command_end", "call_id": "nope", "exit_code": 1}),
                 cx("response_item", {"type": "something_new", "x": 1}),
                 json.dumps({"type": "response_item", "payload": "not a dict"}), "[]", "3", "null"]
        (s,) = formats.parse(lines).values()
        self.assertEqual(len(s["events"]), 1)
        self.assertEqual(s["events"][0]["input"], {"command": ""})
        self.assertEqual(tm.step(s["events"][0]), "sh:?")

    def test_two_rollouts_in_one_stream(self):
        lines = [cx("session_meta", {"id": "A"}), fcall("c1", "ls"), fout("c1", "x"),
                 cx("session_meta", {"id": "B"}), fcall("c1", "pwd"),
                 fout("c1", json.dumps({"output": "", "metadata": {"exit_code": 1}}))]
        p = formats.parse(lines)
        self.assertEqual(list(p), ["codex:A", "codex:B"])
        self.assertTrue(p["codex:A"]["events"][0]["ok"])
        self.assertFalse(p["codex:B"]["events"][0]["ok"])


class ClaudeCode(unittest.TestCase):
    def line(self, typ, content, **kw):
        return json.dumps(dict({"type": typ, "sessionId": "S", "message": {"content": content}}, **kw))

    def test_exit_code_split_from_message(self):
        got = calls([self.line("assistant", [{"type": "tool_use", "id": "1", "name": "Bash",
                                             "input": {"command": "npm test"}}]),
                     self.line("user", [{"type": "tool_result", "tool_use_id": "1", "is_error": True,
                                         "content": "Exit code 2\nError: Cannot find module 'x'"}])])
        self.assertEqual((got[0]["ok"], got[0]["exit"], got[0]["error"]),
                         (False, 2, "Error: Cannot find module 'x'"))

    def test_sidechain_meta_and_missing_result(self):
        use = [{"type": "tool_use", "id": "1", "name": "Read", "input": {"file_path": "a.py"}}]
        p = formats.parse([self.line("user", "do it"), self.line("user", "injected", isMeta=True),
                           self.line("user", "<system-reminder>x</system-reminder>"),
                           self.line("assistant", use, isSidechain=True),
                           self.line("assistant", use)])
        self.assertEqual(sorted(p), ["S", "S:sidechain"])
        self.assertEqual([e.get("text", e.get("tool")) for e in p["S"]["events"]], ["do it", "Read"])
        self.assertFalse(p["S"]["events"][1]["done"])
        self.assertTrue(p["S"]["events"][1]["ok"])

    def test_truncated_only_when_last_line_is_torn(self):
        ok = self.line("user", "hi")
        self.assertFalse(formats.parse([ok, "{torn", ok])["S"]["truncated"])
        p = formats.parse([ok, '{"type":"assistant","sessionId":"S","mess'])
        self.assertTrue(p["S"]["truncated"])
        self.assertEqual(p["S"]["bad_lines"], 1)
        self.assertEqual(formats.parse(["{torn"]), {})


class RoundTrip(unittest.TestCase):
    def test_same_session_gives_same_steps_from_both_agents(self):
        sessions, _ = bs.generate(5, dict(bs.TUNE, sessions=40))
        for sid, s in sessions.items():
            want = [c for k, c in s["items"] if k == "call"]
            for agent, render in (("claude_code", bs.render_claude), ("codex", bs.render_codex)):
                (parsed,) = formats.parse(render(sid, dict(s, torn=False))).values()
                got = [e for e in parsed["events"] if e["kind"] == "call"]
                self.assertEqual(len(got), len(want), (sid, agent))
                self.assertEqual([c["ok"] for c in got], [c["ok"] for c in want], (sid, agent))
                self.assertEqual([c["done"] for c in got], [c["done"] for c in want], (sid, agent))
            a = formats.parse(bs.render_claude(sid, s)).popitem()[1]
            b = formats.parse(bs.render_codex(sid, s)).popitem()[1]
            self.assertEqual([e["text"] for e in a["events"] if e["kind"] == "user"],
                             [e["text"] for e in b["events"] if e["kind"] == "user"])
            # failed calls keep the same error text, so they group together across agents
            self.assertEqual([e["error"] for e in a["events"] if e["kind"] == "call" and not e["ok"]],
                             [e["error"] for e in b["events"] if e["kind"] == "call" and not e["ok"]])


if __name__ == "__main__":
    unittest.main()
