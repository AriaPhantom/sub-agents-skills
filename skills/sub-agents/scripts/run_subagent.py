#!/usr/bin/env python3
"""
run_subagent.py - Execute external CLI AIs as sub-agents

Usage:
    scripts/run_subagent.py --agent <name> --prompt "<task>" --cwd <path>
    scripts/run_subagent.py --list

Supported CLIs: claude, cursor-agent, codex, gemini

Environment:
    SUB_AGENTS_DIR: Override default agents directory ({cwd}/.agents/)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Agent Loader - frontmatter parsing and system context extraction
# =============================================================================


def _coerce_scalar(value: str):
    """Convert simple YAML scalar strings into Python values."""
    stripped = value.strip().strip("\"'")
    lowered = stripped.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        return float(stripped)
    return stripped


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown content.
    Returns (frontmatter_dict, body_without_frontmatter)
    """
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    frontmatter_raw = match.group(1)
    body = match.group(2)

    # Lightweight YAML parser for top-level scalars and simple list values.
    frontmatter = {}
    current_list_key = None

    for line in frontmatter_raw.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- "):
            if current_list_key is not None:
                frontmatter.setdefault(current_list_key, [])
                frontmatter[current_list_key].append(_coerce_scalar(line[2:]))
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                frontmatter[key] = _coerce_scalar(value)
                current_list_key = None
            else:
                frontmatter[key] = []
                current_list_key = key
            continue

        current_list_key = None

    return frontmatter, body


def extract_description(body: str) -> str:
    """Extract description from first non-heading line of body."""
    for line in body.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:100]
    return ""


_AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_GEMINI_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def validate_agent_name(agent_name: str) -> str:
    """
    Validate agent name to prevent path traversal.
    Returns the name if valid, raises ValueError otherwise.
    """
    if not agent_name or not _AGENT_NAME_PATTERN.match(agent_name):
        raise ValueError(f"Invalid agent name: {agent_name!r}")
    return agent_name


def is_gemini_agents_path(agent_file: Path) -> bool:
    """Detect whether a file lives under a .gemini/agents directory."""
    parts = [part.lower() for part in agent_file.parts]
    for idx in range(len(parts) - 2):
        if parts[idx] == ".gemini" and parts[idx + 1] == "agents":
            return True
    return False


def normalize_gemini_config(frontmatter: dict, fallback_name: str) -> dict:
    """Normalize and lightly validate Gemini subagent frontmatter fields."""
    name = frontmatter.get("name") or fallback_name
    description = frontmatter.get("description")
    kind = frontmatter.get("kind", "local")
    model = frontmatter.get("model")
    timeout_mins = frontmatter.get("timeout_mins")

    if not isinstance(name, str) or not _GEMINI_NAME_PATTERN.match(name):
        raise ValueError(
            "Invalid Gemini subagent name. Use lowercase letters, numbers, hyphens, and underscores."
        )

    if kind not in ("local", "remote"):
        raise ValueError("Invalid Gemini subagent kind. Expected 'local' or 'remote'.")

    if timeout_mins is not None and (
        not isinstance(timeout_mins, (int, float)) or timeout_mins <= 0
    ):
        raise ValueError("Invalid Gemini timeout_mins. Expected a positive number.")

    if model is not None and not isinstance(model, str):
        raise ValueError("Invalid Gemini model. Expected a string.")

    return {
        "name": name,
        "description": description if isinstance(description, str) else "",
        "kind": kind,
        "model": model,
        "timeout_mins": timeout_mins,
    }


def read_agent_text(path: Path) -> str:
    """Read agent file content and transparently handle UTF-8 BOM files."""
    return path.read_text(encoding="utf-8-sig")


def _split_dir_list(raw_dirs: str) -> list[str]:
    """Split directory lists using the platform path separator."""
    return [item.strip() for item in raw_dirs.split(os.pathsep) if item.strip()]


def get_agents_dirs(args_agents_dir: str | None, args_cwd: str | None) -> list[str]:
    """
    Determine agent directories.
    Priority: --agents-dir > SUB_AGENTS_DIR > defaults
    Defaults: {cwd}/.agents, {cwd}/.gemini/agents, ~/.gemini/agents
    """
    if args_agents_dir:
        return _split_dir_list(args_agents_dir)

    env_dir = os.environ.get("SUB_AGENTS_DIR")
    if env_dir:
        return _split_dir_list(env_dir)

    base_dir = Path(args_cwd) if args_cwd else Path.cwd()
    candidates = [
        base_dir / ".agents",
        base_dir / ".gemini" / "agents",
        Path.home() / ".gemini" / "agents",
    ]

    unique_dirs = []
    seen = set()
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        unique_dirs.append(path_str)

    return unique_dirs


def _iter_agent_files(agents_dirs: list[str]):
    """Yield available agent files from each configured directory in priority order."""
    for agents_dir in agents_dirs:
        agents_path = Path(agents_dir)
        if not agents_path.exists():
            continue
        for ext in [".md", ".txt"]:
            for agent_file in sorted(agents_path.glob(f"*{ext}")):
                yield agents_path, agent_file


def _resolve_agent_cli(frontmatter: dict, agent_file: Path) -> str | None:
    """Resolve target CLI from frontmatter and file location."""
    run_agent = frontmatter.get("run-agent")
    if isinstance(run_agent, str) and run_agent.strip():
        return run_agent.strip()

    gemini_keys = {"kind", "tools", "model", "temperature", "max_turns", "timeout_mins"}
    if is_gemini_agents_path(agent_file) or any(key in frontmatter for key in gemini_keys):
        return "gemini"

    return None


def load_agent(agents_dirs: list[str], agent_name: str) -> tuple[str | None, str, str, dict]:
    """
    Load agent definition from configured directories.
    Returns (run_agent_cli, system_context, description, metadata)
    """
    validate_agent_name(agent_name)

    for agents_path, agent_file in _iter_agent_files(agents_dirs):
        try:
            resolved = agent_file.resolve()
            resolved.relative_to(agents_path.resolve())
        except ValueError:
            continue

        content = read_agent_text(resolved)
        frontmatter, body = parse_frontmatter(content)
        run_agent_cli = _resolve_agent_cli(frontmatter, resolved)

        metadata = {
            "schema": "legacy",
            "source": str(resolved),
            "model": None,
            "timeout_mins": None,
            "kind": "local",
        }

        effective_name = resolved.stem
        description = extract_description(body)

        if run_agent_cli == "gemini":
            gemini_cfg = normalize_gemini_config(frontmatter, resolved.stem)
            effective_name = gemini_cfg["name"]
            description = gemini_cfg["description"] or description
            metadata.update(
                {
                    "schema": "gemini",
                    "model": gemini_cfg["model"],
                    "timeout_mins": gemini_cfg["timeout_mins"],
                    "kind": gemini_cfg["kind"],
                }
            )
        elif isinstance(frontmatter.get("name"), str) and frontmatter["name"].strip():
            effective_name = frontmatter["name"].strip()

        aliases = {resolved.stem, effective_name}
        if agent_name not in aliases:
            continue

        return run_agent_cli, body.strip(), description, metadata

    raise FileNotFoundError(f"Agent definition not found: {agent_name}")


def list_agents(agents_dirs: list[str]) -> list[dict]:
    """
    List all available agents across configured directories.
    Returns list of {"name": str, "description": str, ...}
    """
    agents = []
    seen_names: set[str] = set()

    for _, agent_file in _iter_agent_files(agents_dirs):
        name = agent_file.stem
        description = ""
        cli = None
        schema = "legacy"

        try:
            content = read_agent_text(agent_file)
            frontmatter, body = parse_frontmatter(content)
            cli = _resolve_agent_cli(frontmatter, agent_file)
            description = extract_description(body)

            if cli == "gemini":
                gemini_cfg = normalize_gemini_config(frontmatter, agent_file.stem)
                name = gemini_cfg["name"]
                description = gemini_cfg["description"] or description
                schema = "gemini"
            elif isinstance(frontmatter.get("name"), str) and frontmatter["name"].strip():
                name = frontmatter["name"].strip()
        except Exception as exc:
            description = f"(invalid agent: {exc})"

        if name in seen_names:
            continue
        seen_names.add(name)

        agent_info = {
            "name": name,
            "description": description,
            "schema": schema,
            "source": str(agent_file),
        }
        if cli:
            agent_info["cli"] = cli
        agents.append(agent_info)

    return sorted(agents, key=lambda a: a["name"])


# =============================================================================
# CLI Resolver - determine which CLI to use
# =============================================================================


def detect_caller_cli() -> str | None:
    """
    Detect which CLI is calling this script based on environment.
    Returns: 'claude', 'cursor-agent', 'codex', 'gemini', or None
    """
    # Check common environment indicators
    if os.environ.get("CLAUDE_CODE"):
        return "claude"
    if os.environ.get("CURSOR_AGENT"):
        return "cursor-agent"
    if os.environ.get("CODEX_CLI"):
        return "codex"
    if os.environ.get("GEMINI_CLI"):
        return "gemini"

    # Check parent process name (best effort)
    try:
        ppid = os.getppid()
        cmdline_path = f"/proc/{ppid}/cmdline"
        if os.path.exists(cmdline_path):
            with open(cmdline_path) as f:
                cmdline = f.read().lower()
                if "claude" in cmdline:
                    return "claude"
                if "cursor" in cmdline:
                    return "cursor-agent"
                if "codex" in cmdline:
                    return "codex"
                if "gemini" in cmdline:
                    return "gemini"
    except Exception:
        pass

    return None


def resolve_cli(frontmatter_cli: str | None, default: str = "codex") -> str:
    """
    Resolve which CLI to use.
    Priority: frontmatter > caller detection > default
    """
    if frontmatter_cli:
        valid_clis = {"claude", "cursor-agent", "codex", "gemini"}
        if frontmatter_cli in valid_clis:
            return frontmatter_cli

    detected = detect_caller_cli()
    if detected:
        return detected

    return default


# =============================================================================
# Stream Processor - parse CLI output
# =============================================================================


class StreamProcessor:
    """Process streaming JSON output from various CLIs."""

    def __init__(self):
        self.result_json = None
        self.gemini_parts = []
        self.codex_messages = []
        self.is_gemini = False
        self.is_codex = False

    def process_line(self, line: str) -> bool:
        """
        Process a line from CLI output.
        Returns True when result is ready, False to continue.
        """
        line = line.strip()
        if not line or self.result_json is not None:
            return False

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return False

        # Detect Gemini format
        if data.get("type") == "init":
            self.is_gemini = True
            return False

        # Detect Codex format
        if data.get("type") == "thread.started":
            self.is_codex = True
            return False

        # Gemini: accumulate assistant messages
        if self.is_gemini and data.get("type") == "message" and data.get("role") == "assistant":
            content = data.get("content", "")
            if isinstance(content, str):
                self.gemini_parts.append(content)
            return False

        # Codex: accumulate agent_message items
        if self.is_codex and data.get("type") == "item.completed":
            item = data.get("item", {})
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                self.codex_messages.append(item["text"])
            return False

        # Codex: turn.completed signals end
        if self.is_codex and data.get("type") == "turn.completed":
            self.result_json = {
                "type": "result",
                "result": "\n".join(self.codex_messages),
                "status": "success",
            }
            return True

        # Result type signals completion
        if data.get("type") == "result":
            if self.is_gemini:
                self.result_json = {
                    "type": "result",
                    "result": "".join(self.gemini_parts),
                    "status": data.get("status", "success"),
                }
            else:
                self.result_json = data
            return True

        # Fallback: first valid JSON without type field
        if "type" not in data:
            self.result_json = data
            return True

        return False

    def get_result(self):
        return self.result_json


# =============================================================================
# Agent Executor - run CLI and capture output
# =============================================================================


def build_command(cli: str, prompt: str, agent_meta: dict | None = None) -> tuple[str, list]:
    """Build command and arguments for the specified CLI."""

    is_windows = os.name == "nt"

    if cli == "codex":
        command = "codex.cmd" if is_windows else "codex"
        # Keep codex in non-interactive mode; full multi-line prompt is sent via stdin.
        return command, ["exec", "--json", "--skip-git-repo-check", "-"]

    if cli == "claude":
        return "claude", ["--output-format", "stream-json", "--verbose", "-p", prompt]

    if cli == "gemini":
        command = "gemini.cmd" if is_windows else "gemini"
        # Keep gemini in headless mode; full multi-line prompt is sent via stdin.
        args = ["--output-format", "stream-json", "-p", "Use stdin as the full task context."]
        model = (
            (agent_meta or {}).get("model")
            or os.environ.get("SUB_AGENT_GEMINI_MODEL")
            or "flash"
        )
        if isinstance(model, str) and model.strip():
            args.extend(["-m", model.strip()])
        return command, args

    if cli == "cursor-agent":
        args = ["--output-format", "json", "-p", prompt]
        api_key = os.environ.get("CLI_API_KEY")
        if api_key:
            args.extend(["-a", api_key])
        return "cursor-agent", args

    raise ValueError(f"Unknown CLI: {cli}")


def execute_agent(
    cli: str,
    system_context: str,
    prompt: str,
    cwd: str,
    timeout: int = 600000,
    agent_meta: dict | None = None,
) -> dict:
    """
    Execute agent CLI and return result.

    Returns: {
        "result": str,
        "exit_code": int,
        "status": "success" | "partial" | "error",
        "cli": str
    }
    """
    # Format prompt with system context
    formatted_prompt = f"[System Context]\n{system_context}\n\n[User Prompt]\n{prompt}"

    command, args = build_command(cli, formatted_prompt, agent_meta=agent_meta)
    timeout_sec = timeout / 1000

    try:
        # Windows compatibility: some wrappers may need shell=True, but codex/gemini hang or mis-handle args with it.
        is_windows = os.name == "nt"
        use_shell = is_windows and cli not in {"codex", "gemini"}

        stdin_pipe = subprocess.PIPE if cli in {"codex", "gemini"} else None

        process = subprocess.Popen(
            [command] + args,
            cwd=cwd,
            stdin=stdin_pipe,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=use_shell
        )

        processor = StreamProcessor()

        try:
            # Use communicate() with timeout to avoid blocking on readline() on Windows.
            try:
                stdout, stderr = process.communicate(
                    input=formatted_prompt if cli in {"codex", "gemini"} else None,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as e:
                process.kill()
                stdout, stderr = process.communicate()

                timed_out_stdout = (e.output or "") + (stdout or "")
                for line in timed_out_stdout.splitlines():
                    processor.process_line(line)
                result = processor.get_result()

                return {
                    "result": result.get("result", "") if result else timed_out_stdout,
                    "exit_code": 124,
                    "status": "partial" if result or timed_out_stdout else "error",
                    "cli": cli,
                    "error": f"Timeout after {timeout}ms",
                }

            for line in stdout.splitlines():
                processor.process_line(line)

            result = processor.get_result()
            exit_code = process.returncode or 0

            # Determine status
            if exit_code == 0 or exit_code in (143, -15) and result:
                status = "success"
            elif result:
                status = "partial"
            else:
                status = "error"

            response = {
                "result": result.get("result", "") if result else stdout,
                "exit_code": exit_code,
                "status": status,
                "cli": cli,
            }
            if status == "error":
                error_msg = f"CLI exited with code {exit_code}"
                if stderr and stderr.strip():
                    error_msg += f": {stderr.strip()}"
                response["error"] = error_msg
            return response

        except Exception as e:
            process.kill()
            result = processor.get_result()
            return {
                "result": result.get("result", "") if result else "",
                "exit_code": 1,
                "status": "error",
                "cli": cli,
                "error": str(e),
            }

    except FileNotFoundError:
        return {
            "result": "",
            "exit_code": 127,
            "status": "error",
            "cli": cli,
            "error": f"CLI not found: {command}",
        }
    except Exception as e:
        return {"result": "", "exit_code": 1, "status": "error", "cli": cli, "error": str(e)}


# =============================================================================
# Main
# =============================================================================


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Execute external CLI AIs as sub-agents")
    parser.add_argument("--list", action="store_true", help="List available agents")
    parser.add_argument("--agent", help="Agent definition name")
    parser.add_argument("--prompt", help="Task prompt")
    parser.add_argument("--cwd", help="Working directory (absolute path)")
    parser.add_argument("--agents-dir", help="Directory containing agent definitions")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in ms")
    parser.add_argument("--cli", help="Force specific CLI (claude, cursor-agent, codex, gemini)")

    args = parser.parse_args()

    # Handle --list
    if args.list:
        agents_dirs = get_agents_dirs(args.agents_dir, args.cwd)
        agents = list_agents(agents_dirs)
        print(
            json.dumps(
                {
                    "agents": agents,
                    "agents_dir": agents_dirs[0] if agents_dirs else "",
                    "agents_dirs": agents_dirs,
                },
                ensure_ascii=False,
            )
        )
        sys.exit(0)

    # Validate required args for execution
    if not args.agent:
        print(
            json.dumps(
                {"result": "", "exit_code": 1, "status": "error", "error": "--agent is required"}
            )
        )
        sys.exit(1)

    if not args.prompt:
        print(
            json.dumps(
                {"result": "", "exit_code": 1, "status": "error", "error": "--prompt is required"}
            )
        )
        sys.exit(1)

    if not args.cwd:
        print(
            json.dumps(
                {"result": "", "exit_code": 1, "status": "error", "error": "--cwd is required"}
            )
        )
        sys.exit(1)

    # Validate cwd
    if not os.path.isabs(args.cwd):
        print(
            json.dumps(
                {
                    "result": "",
                    "exit_code": 1,
                    "status": "error",
                    "error": "cwd must be an absolute path",
                }
            )
        )
        sys.exit(1)

    if not os.path.isdir(args.cwd):
        print(
            json.dumps(
                {
                    "result": "",
                    "exit_code": 1,
                    "status": "error",
                    "error": f"cwd does not exist: {args.cwd}",
                }
            )
        )
        sys.exit(1)

    # Determine agent directories
    agents_dirs = get_agents_dirs(args.agents_dir, args.cwd)

    # Load agent definition
    try:
        run_agent_cli, system_context, _, agent_meta = load_agent(agents_dirs, args.agent)
    except FileNotFoundError as e:
        print(json.dumps({"result": "", "exit_code": 1, "status": "error", "error": str(e)}))
        sys.exit(1)
    except ValueError as e:
        print(json.dumps({"result": "", "exit_code": 1, "status": "error", "error": str(e)}))
        sys.exit(1)

    # Resolve CLI
    cli = args.cli or resolve_cli(run_agent_cli)

    if cli == "gemini" and agent_meta.get("kind") == "remote":
        print(
            json.dumps(
                {
                    "result": "",
                    "exit_code": 1,
                    "status": "error",
                    "error": (
                        "Gemini remote subagents (kind: remote) are not supported by this wrapper. "
                        "Run them from native Gemini CLI flows."
                    ),
                }
            )
        )
        sys.exit(1)

    if args.timeout is None:
        timeout = 600000
        timeout_mins = agent_meta.get("timeout_mins")
        if cli == "gemini" and isinstance(timeout_mins, (int, float)) and timeout_mins > 0:
            timeout = int(timeout_mins * 60 * 1000)
    else:
        timeout = args.timeout

    if timeout <= 0:
        print(
            json.dumps(
                {"result": "", "exit_code": 1, "status": "error", "error": "--timeout must be > 0"}
            )
        )
        sys.exit(1)

    # Execute
    result = execute_agent(
        cli=cli,
        system_context=system_context,
        prompt=args.prompt,
        cwd=args.cwd,
        timeout=timeout,
        agent_meta=agent_meta,
    )

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
