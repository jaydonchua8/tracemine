"""formats: read agent session logs from different coding agents into one event list.

Three line shapes are accepted, and they can be mixed in one stream:

  claude_code  {"type": "user"|"assistant", "sessionId", "message": {"content": ...}}
               tool_use blocks on assistant lines, tool_result blocks on user lines.
  codex        {"timestamp", "type": "session_meta"|"response_item"|"event_msg", "payload"}
               payload variants follow ResponseItem in openai/codex
               codex-rs/protocol/src/models.rs (function_call, function_call_output,
               local_shell_call, custom_tool_call, custom_tool_call_output, message).
  flat         {"session", "tool", "input", "ok", "error"}  (used by bench.py)

parse() returns {session_id: session} where session is
  {"agent": str, "events": [event, ...], "truncated": bool, "bad_lines": int}
and an event is one of
  {"kind": "user", "text": str}
  {"kind": "call", "tool", "input", "ok", "error", "done", "agent"}
"ok" is False only when the log says the call failed. "done" is False when the log has
no result for the call (interrupted or cut off session).

Codex calls are mapped onto the Claude Code tool names so that the same work mines to
the same step keys: shell calls become Bash, apply_patch becomes Edit or Write, and a
shell command that only prints one file (cat, nl, head, tail, sed -n) becomes Read.

Standard library only.
"""
import json
import re
import shlex
from collections import OrderedDict

SHELL_TOOLS = {"shell", "shell_command", "exec_command", "local_shell", "container.exec",
               "unified_exec"}
PRINT_PROGS = {"cat", "nl", "head", "tail", "bat"}
WRAPPER_TAGS = ("<environment_context", "<user_instructions", "<turn_aborted",
                "<system-reminder", "<local-command-stdout", "<user_shell_command")
HEADER_RE = re.compile(r"^\s*(?:(?:exit code|process exited with code|wall time)[: ][^\n]*\n?"
                       r"|output:[ \t]*\n?)+", re.I)
EXIT_RE = re.compile(r"^\s*(?:exit code|process exited with code)[: ]+(-?\d+)", re.I | re.M)


def _new_session(agent):
    return {"agent": agent, "events": [], "truncated": False, "bad_lines": 0}


def _text(content):
    """Text of a message or tool result body, whether it is a string or a block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    if isinstance(content, dict):
        return str(content.get("content") or content.get("output") or content.get("text") or "")
    return ""


def _call(tool, inp, agent, ok=True, error=""):
    return {"kind": "call", "tool": tool, "input": inp if isinstance(inp, dict) else {},
            "ok": ok, "error": error, "done": False, "agent": agent}


# ---------------------------------------------------------------- codex helpers

def _command_text(args):
    """The shell line from the arguments of a Codex shell style tool call."""
    cmd = args.get("command", args.get("cmd"))
    if isinstance(cmd, list):
        cmd = [str(c) for c in cmd]
        if len(cmd) >= 3 and cmd[0].rsplit("/", 1)[-1] in ("bash", "sh", "zsh") and \
                cmd[1] in ("-lc", "-c"):
            return cmd[2]
        return " ".join(shlex.quote(c) for c in cmd)
    return str(cmd or "")


def _printed_file(command):
    """Path if the command only prints one file, else None."""
    if any(ch in command for ch in "|;&><$`"):
        return None
    try:
        toks = shlex.split(command)
    except ValueError:
        return None
    if not toks:
        return None
    prog, rest = toks[0].rsplit("/", 1)[-1], toks[1:]
    if prog == "sed" and rest[:1] == ["-n"]:
        rest = rest[2:]
    elif prog not in PRINT_PROGS:
        return None
    paths = [t for t in rest if not t.startswith("-") and not t.isdigit()]
    return paths[0] if len(paths) == 1 else None


def _patch_call(patch, agent):
    """apply_patch input -> Edit or Write on the first file it names."""
    m = re.search(r"^\*\*\* (Add|Update|Delete) File: (.+)$", patch, re.M)
    if not m:
        return _call("apply_patch", {}, agent)
    return _call("Write" if m.group(1) == "Add" else "Edit", {"file_path": m.group(2).strip()},
                 agent)


def _codex_call(name, args, agent):
    if name in SHELL_TOOLS:
        command = _command_text(args)
        if "*** Begin Patch" in command:
            return _patch_call(command, agent)
        path = _printed_file(command)
        if path is not None:
            return _call("Read", {"file_path": path}, agent)
        return _call("Bash", {"command": command}, agent)
    return _call(name, args, agent)


def _codex_result(call, output):
    """Set ok/error from a function_call_output body.

    The payload's `success` flag is not serialised by Codex, so failure has to be read
    from the body: a JSON object with metadata.exit_code, or an "Exit code: N" line.
    A body with neither is treated as a success.
    """
    body = _text(output)
    code = None
    try:
        obj = json.loads(body)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        meta = obj.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("exit_code"), int):
            code = meta["exit_code"]
        body = str(obj.get("output", body))
    if isinstance(output, dict) and output.get("success") is False and code is None:
        code = 1
    if code is None:
        m = EXIT_RE.search(body)
        if m:
            code = int(m.group(1))
    call["done"] = True
    if code:
        body = HEADER_RE.sub("", body)
        call["ok"] = False
        call["error"] = body.strip() or "Exit code %d" % code
        call["exit"] = code

# ---------------------------------------------------------------- parser

def parse(lines):
    sessions = OrderedDict()
    pending = {}
    state = {"codex": None, "last": None, "n": 0}

    def session(sid, agent):
        if sid not in sessions:
            sessions[sid] = _new_session(agent)
        state["last"] = sid
        return sessions[sid]

    def add_user(s, text):
        text = text.strip()
        if not text or text.startswith(WRAPPER_TAGS):
            return
        ev = s["events"]
        if ev and ev[-1]["kind"] == "user" and ev[-1]["text"] == text:
            return
        ev.append({"kind": "user", "text": text})

    def add_call(s, sid, call_id, call):
        s["events"].append(call)
        if call_id is not None:
            pending[(sid, call_id)] = call

    bad_at = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        state["n"] += 1
        try:
            ev = json.loads(line)
        except ValueError:
            ev = None
        if not isinstance(ev, dict):
            bad_at = (state["n"], state["last"])
            if state["last"] in sessions:
                sessions[state["last"]]["bad_lines"] += 1
            continue

        if "tool" in ev and "message" not in ev and "payload" not in ev:      # flat
            s = session(str(ev.get("session", "s0")), "flat")
            c = _call(ev["tool"], ev.get("input") or {}, ev.get("agent", "flat"),
                      bool(ev.get("ok", True)), ev.get("error") or "")
            c["done"] = True
            s["events"].append(c)
            continue

        if isinstance(ev.get("payload"), dict) and "type" in ev:               # codex
            p, kind = ev["payload"], ev["type"]
            if kind == "session_meta":
                state["codex"] = "codex:" + str(p.get("id") or p.get("session_id") or state["n"])
                session(state["codex"], "codex")
                continue
            if state["codex"] is None:
                state["codex"] = "codex:unknown"
            sid = state["codex"]
            s = session(sid, "codex")
            pt = p.get("type")
            if kind == "response_item":
                if pt == "message" and p.get("role") == "user":
                    add_user(s, _text(p.get("content")))
                elif pt == "function_call":
                    try:
                        args = json.loads(p.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    add_call(s, sid, p.get("call_id"),
                             _codex_call(str(p.get("name", "?")), args if isinstance(args, dict) else {}, "codex"))
                elif pt == "local_shell_call":
                    action = p.get("action") if isinstance(p.get("action"), dict) else {}
                    add_call(s, sid, p.get("call_id"), _codex_call("local_shell", action, "codex"))
                elif pt == "custom_tool_call":
                    if p.get("name") == "apply_patch":
                        c = _patch_call(str(p.get("input", "")), "codex")
                    else:
                        c = _call(str(p.get("name", "?")), {}, "codex")
                    add_call(s, sid, p.get("call_id"), c)
                elif pt in ("function_call_output", "custom_tool_call_output"):
                    c = pending.pop((sid, p.get("call_id")), None)
                    if c is not None:
                        _codex_result(c, p.get("output"))
            elif kind == "event_msg":
                if pt == "user_message":
                    add_user(s, str(p.get("message", "")))
                elif pt == "exec_command_end":
                    c = pending.get((sid, p.get("call_id")))
                    code = p.get("exit_code")
                    if c is not None and isinstance(code, int) and code != 0:
                        c["ok"], c["exit"] = False, code
                        c["error"] = c["error"] or str(p.get("aggregated_output") or
                                                       "Exit code %d" % code).strip()
                        c["done"] = True
            continue

        msg = ev.get("message")                                                # claude_code
        if not isinstance(msg, dict):
            continue
        sid = str(ev.get("sessionId", "s0"))
        if ev.get("isSidechain"):
            sid += ":sidechain"
        s = session(sid, "claude_code")
        content = msg.get("content")
        if isinstance(content, str):
            if ev.get("type") == "user" and not ev.get("isMeta"):
                add_user(s, content)
            continue
        if not isinstance(content, list):
            continue
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "tool_use":
                add_call(s, sid, block.get("id"),
                         _call(str(block.get("name", "?")), block.get("input") or {}, "claude_code"))
            elif bt == "tool_result":
                c = pending.pop((sid, block.get("tool_use_id")), None)
                if c is not None:
                    c["done"] = True
                    if block.get("is_error"):
                        c["ok"] = False
                        err = _text(block.get("content"))
                        m = EXIT_RE.match(err)
                        if m:               # keep the status apart so both agents agree
                            c["exit"] = int(m.group(1))
                            err = err[m.end():].strip() or "Exit code %d" % c["exit"]
                        c["error"] = err
            elif bt == "text" and ev.get("type") == "user" and not ev.get("isMeta"):
                texts.append(str(block.get("text", "")))
        if texts:
            add_user(s, "\n".join(texts))

    # A line that does not parse is a torn write only if it was the last line of the stream.
    if bad_at and bad_at[0] == state["n"] and bad_at[1] in sessions:
        sessions[bad_at[1]]["truncated"] = True
    return sessions


def calls_only(sessions):
    """{session_id: [call, ...]} for the miners in tracemine.py."""
    return {sid: [e for e in s["events"] if e["kind"] == "call"]
            for sid, s in sessions.items() if any(e["kind"] == "call" for e in s["events"])}
