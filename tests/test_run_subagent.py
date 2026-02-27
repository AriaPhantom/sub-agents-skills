#!/usr/bin/env python3
"""Tests for run_subagent.py."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "sub-agents" / "scripts"))

from run_subagent import (  # noqa: E402
    StreamProcessor,
    build_command,
    detect_caller_cli,
    execute_agent,
    extract_description,
    get_agents_dirs,
    is_path_within,
    is_small_task_prompt,
    list_agents,
    load_agent,
    normalize_approval_mode,
    normalize_gemini_config,
    parse_frontmatter,
    read_agent_text,
    resolve_cli,
    resolve_cli_command,
    resolve_include_directories,
    resolve_small_task_max_chars,
)


def _write(path: Path, content: str, encoding: str = "utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


class TestParseFrontmatter:
    def test_parses_scalars_and_lists(self):
        content = """---
name: frontend-ui
timeout_mins: 5
enabled: true
tools:
  - Read
  - Bash
---

# Agent
Body
"""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["name"] == "frontend-ui"
        assert frontmatter["timeout_mins"] == 5
        assert frontmatter["enabled"] is True
        assert frontmatter["tools"] == ["Read", "Bash"]
        assert body.startswith("# Agent")

    def test_without_frontmatter(self):
        content = "# Agent\nNo frontmatter.\n"
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content


class TestGeminiConfigNormalization:
    def test_valid_gemini_config(self):
        cfg = normalize_gemini_config(
            {
                "name": "frontend_ui",
                "description": "Frontend helper",
                "kind": "local",
                "model": "flash",
                "timeout_mins": 3,
                "approval_mode": "autoEdit",
                "include_directories": ["./ui", "../shared"],
                "tools": ["Read"],
            },
            fallback_name="fallback",
        )
        assert cfg["name"] == "frontend_ui"
        assert cfg["kind"] == "local"
        assert cfg["model"] == "flash"
        assert cfg["timeout_mins"] == 3
        assert cfg["approval_mode"] == "auto_edit"
        assert cfg["include_directories"] == ["./ui", "../shared"]
        assert cfg["ignored_fields"] == ["tools"]

    def test_invalid_gemini_name_raises(self):
        with pytest.raises(ValueError, match="Invalid Gemini subagent name"):
            normalize_gemini_config({"name": "Bad Name"}, fallback_name="fallback")

    def test_invalid_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout_mins"):
            normalize_gemini_config({"name": "valid_name", "timeout_mins": 0}, fallback_name="x")

    def test_invalid_approval_mode_raises(self):
        with pytest.raises(ValueError, match="approval_mode"):
            normalize_gemini_config({"name": "valid_name", "approval_mode": "unsafe"}, "fallback")

    def test_invalid_include_directories_raises(self):
        with pytest.raises(ValueError, match="include_directories"):
            normalize_gemini_config({"name": "valid_name", "include_directories": 123}, "fallback")


class TestApprovalModeNormalization:
    def test_normalizes_alias(self):
        assert normalize_approval_mode("autoEdit") == "auto_edit"
        assert normalize_approval_mode("auto-edit") == "auto_edit"

    def test_none_and_blank(self):
        assert normalize_approval_mode(None) is None
        assert normalize_approval_mode("   ") is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="approval_mode"):
            normalize_approval_mode(True)  # type: ignore[arg-type]


class TestSmallTaskHeuristic:
    def test_small_prompt_is_true(self):
        assert is_small_task_prompt("Fix typo in README.") is True

    def test_large_prompt_is_false(self):
        assert is_small_task_prompt("x" * 400) is False

    def test_invalid_type_is_false(self):
        assert is_small_task_prompt(None) is False

    def test_env_threshold_override(self):
        with patch.dict("os.environ", {"SUB_AGENT_SMALL_TASK_MAX_CHARS": "10"}):
            assert resolve_small_task_max_chars() == 10
            assert is_small_task_prompt("short") is True
            assert is_small_task_prompt("this is definitely longer than ten chars") is False

    def test_invalid_env_threshold_uses_default(self):
        with patch.dict("os.environ", {"SUB_AGENT_SMALL_TASK_MAX_CHARS": "bad"}):
            assert resolve_small_task_max_chars() == 220


class TestReadAgentText:
    def test_reads_utf8_sig(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.md"
            _write(path, "---\nname: test\n---\nBody\n", encoding="utf-8-sig")
            text = read_agent_text(path)
            assert not text.startswith("\ufeff")
            assert "name: test" in text


class TestGetAgentsDirs:
    def test_args_priority_and_path_list_split(self):
        joined = f"/a{os.pathsep}/b"
        assert get_agents_dirs(joined, "/cwd") == ["/a", "/b"]

    def test_env_priority(self):
        with patch.dict("os.environ", {"SUB_AGENTS_DIR": f"/env1{os.pathsep}/env2"}):
            assert get_agents_dirs(None, "/cwd") == ["/env1", "/env2"]

    def test_default_dirs(self):
        with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as home:
            with patch("run_subagent.Path.home", return_value=Path(home)):
                dirs = get_agents_dirs(None, cwd)
        assert dirs[0] == str(Path(cwd) / ".agents")
        assert dirs[1] == str(Path(cwd) / ".gemini" / "agents")
        assert dirs[2] == str(Path(home) / ".gemini" / "agents")


class TestResolveIncludeDirectories:
    def test_resolves_relative_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            (cwd / "a").mkdir()
            (cwd / "b").mkdir()

            dirs = resolve_include_directories(["a", "./a", "b"], str(cwd))
            assert dirs == [str((cwd / "a").resolve()), str((cwd / "b").resolve())]

    def test_rejects_invalid_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            with pytest.raises(ValueError, match="include_directories path is invalid"):
                resolve_include_directories(["missing-dir"], str(cwd))

    def test_rejects_more_than_five_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            entries = []
            for idx in range(6):
                path = cwd / f"d{idx}"
                path.mkdir()
                entries.append(str(path))
            with pytest.raises(ValueError, match="up to 5 directories"):
                resolve_include_directories(entries, str(cwd))


class TestResolveCliCommand:
    def test_non_windows_keeps_command(self):
        with patch("run_subagent.os.name", "posix"):
            assert resolve_cli_command("codex") == "codex"

    def test_windows_prefers_resolved_wrapper(self):
        def _which(candidate: str):
            if candidate == "codex.cmd":
                return r"C:\tools\codex.cmd"
            return None

        with patch("run_subagent.os.name", "nt"):
            with patch("run_subagent.shutil.which", side_effect=_which):
                assert resolve_cli_command("codex") == r"C:\tools\codex.cmd"

    def test_windows_fallbacks_to_plain_command(self):
        with patch("run_subagent.os.name", "nt"):
            with patch("run_subagent.shutil.which", return_value=None):
                assert resolve_cli_command("gemini") == "gemini"


class TestPathSafety:
    def test_is_path_within(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / ".agents"
            inside = base / "worker.md"
            outside = root / "outside.md"
            _write(inside, "inside")
            _write(outside, "outside")

            assert is_path_within(base, inside) is True
            assert is_path_within(base, outside) is False


class TestLoadAgent:
    def test_loads_legacy_agent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = Path(tmpdir) / ".agents"
            _write(
                legacy_dir / "backend.md",
                """---
run-agent: codex
---

# Backend Agent
Handles backend tasks.
""",
            )
            cli, context, desc, meta = load_agent([str(legacy_dir)], "backend")
            assert cli == "codex"
            assert "Backend Agent" in context
            assert desc == "Handles backend tasks."
            assert meta["schema"] == "legacy"

    def test_loads_gemini_agent_by_frontmatter_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gemini_dir = Path(tmpdir) / ".gemini" / "agents"
            (Path(tmpdir) / "src").mkdir()
            _write(
                gemini_dir / "ui-helper.md",
                """---
name: frontend-ui
description: Frontend specialist
kind: local
model: flash
timeout_mins: 2
approval_mode: autoEdit
include_directories:
  - ../src
tools:
  - Read
---

You are frontend specialist.
""",
            )
            cli, context, desc, meta = load_agent([str(gemini_dir)], "frontend-ui")
            assert cli == "gemini"
            assert "frontend specialist" in context.lower()
            assert desc == "Frontend specialist"
            assert meta["schema"] == "gemini"
            assert meta["model"] == "flash"
            assert meta["timeout_mins"] == 2
            assert meta["approval_mode"] == "auto_edit"
            assert meta["include_directories"] == ["../src"]
            assert meta["ignored_fields"] == ["tools"]

    def test_rejects_invalid_agent_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid agent name"):
                load_agent([tmpdir], "../etc/passwd")

    def test_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            load_agent([tmpdir], "missing-agent")

    def test_skips_paths_outside_agent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = Path(tmpdir) / ".agents"
            _write(
                legacy_dir / "backend.md",
                """---
run-agent: codex
---

# Backend Agent
Handles backend tasks.
""",
            )
            with patch("run_subagent.is_path_within", return_value=False):
                with pytest.raises(FileNotFoundError):
                    load_agent([str(legacy_dir)], "backend")


class TestListAgents:
    def test_lists_agents_across_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = Path(tmpdir) / ".agents"
            gemini = Path(tmpdir) / ".gemini" / "agents"
            _write(legacy / "backend.md", "---\nrun-agent: codex\n---\n\nBackend helper.\n")
            _write(
                gemini / "frontend.md",
                """---
name: frontend-ui
description: Frontend helper
kind: local
---

Instructions.
""",
            )
            agents = list_agents([str(legacy), str(gemini)])
            names = {a["name"] for a in agents}
            assert "backend" in names
            assert "frontend-ui" in names
            frontend = next(a for a in agents if a["name"] == "frontend-ui")
            assert frontend["schema"] == "gemini"
            assert frontend["cli"] == "gemini"
            assert "approval_mode" not in frontend

    def test_lists_gemini_extended_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gemini = Path(tmpdir) / ".gemini" / "agents"
            _write(
                gemini / "frontend.md",
                """---
name: frontend-ui
description: Frontend helper
kind: local
model: pro
timeout_mins: 6
approval_mode: plan
include_directories:
  - ./ui
temperature: 0.2
---

Instructions.
""",
            )
            agents = list_agents([str(gemini)])
            assert len(agents) == 1
            agent = agents[0]
            assert agent["name"] == "frontend-ui"
            assert agent["model"] == "pro"
            assert agent["timeout_mins"] == 6
            assert agent["approval_mode"] == "plan"
            assert agent["include_directories"] == ["./ui"]
            assert agent["ignored_fields"] == ["temperature"]

    def test_handles_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_dir = Path(tmpdir) / ".agents"
            _write(agent_dir / "broken.md", "---\nname: Bad Name\nkind: local\n---\nBody\n")
            agents = list_agents([str(agent_dir)])
            assert len(agents) == 1
            assert "invalid agent" in agents[0]["description"]

    def test_skips_paths_outside_agent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_dir = Path(tmpdir) / ".agents"
            _write(agent_dir / "safe.md", "---\nrun-agent: codex\n---\n\nsafe\n")
            with patch("run_subagent.is_path_within", return_value=False):
                agents = list_agents([str(agent_dir)])
            assert agents == []


class TestResolveCli:
    def test_frontmatter_priority(self):
        assert resolve_cli("claude") == "claude"
        assert resolve_cli("codex") == "codex"

    def test_invalid_frontmatter_uses_default(self):
        assert resolve_cli("not-a-cli") == "codex"

    def test_default(self):
        assert resolve_cli(None) == "codex"


class TestDetectCallerCli:
    def test_detects_from_env(self):
        with patch.dict("os.environ", {"GEMINI_CLI": "1"}, clear=False):
            assert detect_caller_cli() == "gemini"


class TestStreamProcessor:
    def test_gemini_stream(self):
        processor = StreamProcessor()
        assert not processor.process_line('{"type":"init"}')
        assert not processor.process_line(
            '{"type":"message","role":"assistant","content":"A"}'
        )
        assert not processor.process_line(
            '{"type":"message","role":"assistant","content":"B"}'
        )
        assert processor.process_line('{"type":"result","status":"success"}')
        assert processor.get_result()["result"] == "AB"

    def test_codex_stream(self):
        processor = StreamProcessor()
        assert not processor.process_line('{"type":"thread.started"}')
        assert not processor.process_line(
            '{"type":"item.completed","item":{"type":"agent_message","text":"x"}}'
        )
        assert processor.process_line('{"type":"turn.completed"}')
        assert processor.get_result()["result"] == "x"


class TestBuildCommand:
    def test_codex_command(self):
        cmd, args = build_command("codex", "prompt")
        assert Path(cmd).name.lower() in {"codex", "codex.cmd", "codex.exe", "codex.bat"}
        assert args == ["exec", "--json", "--skip-git-repo-check", "-"]

    def test_gemini_command_defaults_to_flash(self):
        cmd, args = build_command("gemini", "prompt", agent_meta={})
        assert Path(cmd).name.lower() in {"gemini", "gemini.cmd", "gemini.exe", "gemini.bat"}
        assert args[:4] == [
            "--output-format",
            "stream-json",
            "-p",
            "Use stdin as the full task context.",
        ]
        assert args[-2:] == ["-m", "flash"]

    def test_gemini_command_defaults_to_pro_for_large_prompt(self):
        _, args = build_command("gemini", "x" * 600, agent_meta={})
        assert args[-2:] == ["-m", "pro"]

    def test_gemini_command_uses_user_prompt_hint_for_auto_model(self):
        _, args = build_command(
            "gemini",
            "[System Context]\nVery long context\n\n[User Prompt]\nLong prompt body",
            agent_meta={"_user_prompt": "fix typo"},
        )
        assert args[-2:] == ["-m", "flash"]

    def test_gemini_command_prefers_agent_meta_model(self):
        cmd, args = build_command("gemini", "prompt", agent_meta={"model": "pro"})
        assert Path(cmd).name.lower() in {"gemini", "gemini.cmd", "gemini.exe", "gemini.bat"}
        assert args[-2:] == ["-m", "pro"]

    def test_gemini_command_uses_env_override(self):
        with patch.dict("os.environ", {"SUB_AGENT_GEMINI_MODEL": "pro"}):
            _, args = build_command("gemini", "prompt", agent_meta={})
        assert args[-2:] == ["-m", "pro"]

    def test_gemini_command_appends_approval_mode_and_include_directories(self):
        _, args = build_command(
            "gemini",
            "prompt",
            agent_meta={
                "model": "flash",
                "approval_mode": "plan",
                "include_directories": ["/repo/ui", "/repo/shared"],
            },
        )
        assert "--approval-mode" in args
        assert "plan" in args
        include_pairs = [
            (args[idx], args[idx + 1]) for idx in range(len(args) - 1) if args[idx] == "--include-directories"
        ]
        assert include_pairs == [
            ("--include-directories", "/repo/ui"),
            ("--include-directories", "/repo/shared"),
        ]

    def test_gemini_command_approval_mode_env_override(self):
        with patch.dict("os.environ", {"SUB_AGENT_GEMINI_APPROVAL_MODE": "auto-edit"}):
            _, args = build_command("gemini", "prompt", agent_meta={})
        assert "--approval-mode" in args
        idx = args.index("--approval-mode")
        assert args[idx + 1] == "auto_edit"

    def test_gemini_command_invalid_approval_mode_raises(self):
        with pytest.raises(ValueError, match="approval_mode"):
            build_command("gemini", "prompt", agent_meta={"approval_mode": "nope"})

    def test_cursor_includes_api_key_if_set(self):
        with patch.dict("os.environ", {"CLI_API_KEY": "abc123"}):
            _, args = build_command("cursor-agent", "prompt")
        assert args[-2:] == ["-a", "abc123"]

    def test_unknown_cli_raises(self):
        with pytest.raises(ValueError, match="Unknown CLI"):
            build_command("unknown", "prompt")


class TestExtractDescription:
    def test_extracts_first_non_heading_line(self):
        body = "# Title\n\nDescription line.\n\nMore"
        assert extract_description(body) == "Description line."

    def test_truncates_long_line(self):
        line = "a" * 150
        assert len(extract_description(f"# T\n\n{line}")) == 100


class TestExecuteAgent:
    def test_returns_cli_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "run_subagent.build_command",
                return_value=("non-existent-cli-xyz", ["arg"]),
            ):
                result = execute_agent("codex", "sys", "user", tmpdir, timeout=2000)
        assert result["status"] == "error"
        assert result["exit_code"] == 127

    def test_gemini_uses_stdin_for_prompt(self):
        process = MagicMock()
        process.communicate.return_value = (
            '\n'.join(
                [
                    '{"type":"init"}',
                    '{"type":"message","role":"assistant","content":"OK"}',
                    '{"type":"result","status":"success"}',
                ]
            ),
            "",
        )
        process.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("subprocess.Popen", return_value=process) as mock_popen:
                result = execute_agent(
                    cli="gemini",
                    system_context="System instructions",
                    prompt="User task",
                    cwd=tmpdir,
                    timeout=5000,
                    agent_meta={"model": "flash"},
                )

        assert result["status"] == "success"
        assert result["result"] == "OK"

        communicate_kwargs = process.communicate.call_args.kwargs
        assert "input" in communicate_kwargs
        assert "[System Context]" in communicate_kwargs["input"]
        assert "[User Prompt]" in communicate_kwargs["input"]
        assert communicate_kwargs["timeout"] == 5.0
        popen_kwargs = mock_popen.call_args.kwargs
        assert popen_kwargs["shell"] is False
        assert popen_kwargs["stdin"] == subprocess.PIPE

    def test_codex_uses_stdin_for_prompt(self):
        process = MagicMock()
        process.communicate.return_value = (
            '\n'.join(
                [
                    '{"type":"thread.started"}',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"DONE"}}',
                    '{"type":"turn.completed"}',
                ]
            ),
            "",
        )
        process.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("subprocess.Popen", return_value=process) as mock_popen:
                result = execute_agent(
                    cli="codex",
                    system_context="System instructions",
                    prompt="User task",
                    cwd=tmpdir,
                    timeout=5000,
                )

        assert result["status"] == "success"
        assert result["result"] == "DONE"

        communicate_kwargs = process.communicate.call_args.kwargs
        assert "[System Context]" in communicate_kwargs["input"]
        assert "[User Prompt]" in communicate_kwargs["input"]
        popen_kwargs = mock_popen.call_args.kwargs
        assert popen_kwargs["shell"] is False
        assert popen_kwargs["stdin"] == subprocess.PIPE

    def test_claude_does_not_send_stdin(self):
        process = MagicMock()
        process.communicate.return_value = ('{"type":"result","result":"DONE"}', "")
        process.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("subprocess.Popen", return_value=process) as mock_popen:
                result = execute_agent(
                    cli="claude",
                    system_context="System instructions",
                    prompt="User task",
                    cwd=tmpdir,
                    timeout=5000,
                )

        assert result["status"] == "success"
        assert result["result"] == "DONE"

        communicate_kwargs = process.communicate.call_args.kwargs
        assert communicate_kwargs.get("input") is None
        popen_kwargs = mock_popen.call_args.kwargs
        assert popen_kwargs["shell"] is False
        assert popen_kwargs["stdin"] is None

    def test_timeout_returns_partial_when_output_exists(self):
        process = MagicMock()
        timeout_exc = subprocess.TimeoutExpired(
            cmd=["fake"],
            timeout=1,
            output='{"type":"result","result":"partial-out","status":"success"}\n',
        )
        process.communicate.side_effect = [timeout_exc, ("", "")]
        process.returncode = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("subprocess.Popen", return_value=process):
                result = execute_agent(
                    cli="codex",
                    system_context="sys",
                    prompt="user",
                    cwd=tmpdir,
                    timeout=1000,
                )

        assert result["exit_code"] == 124
        assert result["status"] == "partial"
        assert result["result"] == "partial-out"

    def test_gemini_plan_mode_error_adds_guidance(self):
        process = MagicMock()
        process.communicate.return_value = (
            "",
            'Approval mode "plan" is only available when experimental.plan is enabled.',
        )
        process.returncode = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("subprocess.Popen", return_value=process):
                result = execute_agent(
                    cli="gemini",
                    system_context="System instructions",
                    prompt="User task",
                    cwd=tmpdir,
                    timeout=5000,
                    agent_meta={"approval_mode": "plan"},
                )

        assert result["status"] == "error"
        assert "experimental.plan" in result["error"]
        assert "switch approval_mode to auto_edit/default" in result["error"]
