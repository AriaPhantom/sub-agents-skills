# Sub-Agents Optimization Notes (2026-02-27)

## Goal
Iteratively harden and optimize `skills/sub-agents/scripts/run_subagent.py` so it can safely orchestrate Codex/Claude/Cursor/Gemini sub-agents and better match current Gemini CLI capabilities.

## Completed Prior Rounds (already committed before this note)
1. `fa95caa` - Harden runtime + Gemini-native compatibility.
2. `1339f1e` - Default Gemini model alias in examples switched to `flash`.
3. `5930cef` - Removed shell execution path and hardened agent file boundary checks.

## Web Research Highlights (official docs)
1. Gemini subagents support frontmatter fields including `model`, `description`, `tools`, `temperature`, `max_turns`, and `kind` (`local`/`remote`).
2. Gemini CLI non-interactive mode supports `--approval-mode` and `--include-directories`.
3. Gemini CLI docs and release notes indicate Gemini 3 support is available; model routing can be controlled with `-m` aliases like `pro` and `flash`.

Sources:
- https://geminicli.com/docs/core/sub-agents/
- https://geminicli.com/docs/core/models/
- https://github.com/google-gemini/gemini-cli
- https://github.com/google-gemini/gemini-cli/releases

## Local Runtime Validation Notes
1. `gemini -m gemini-3-flash` returned `ModelNotFoundError` in this environment.
2. `gemini -m flash` worked and reported `gemini-3-flash-preview` in JSON stats.
3. `gemini -m pro` worked and reported `gemini-3.1-pro-preview` in JSON stats.
4. `approval_mode: plan` failed in this environment unless Gemini `experimental.plan` is enabled; `auto_edit` succeeded.

Conclusion: model aliases (`flash`/`pro`) are the most robust default for this environment.

## New Improvements Implemented In This Round
1. Added Gemini frontmatter support for:
   - `approval_mode` (`default`, `auto_edit`, `yolo`, `plan`; alias normalization included).
   - `include_directories` (string or list, max 5).
2. Added include directory resolution/validation:
   - Relative paths resolved against `--cwd`.
   - Invalid or non-directory paths fail fast with clear error.
   - Duplicate entries deduplicated after path resolution.
3. Extended `--list` output for Gemini agents with optional metadata:
   - `model`, `timeout_mins`, `approval_mode`, `include_directories`, `ignored_fields`.
4. Added explicit warning propagation for Gemini-native fields the wrapper cannot enforce in headless mode:
   - `tools`, `temperature`, `max_turns`.
5. Updated docs (`README.md`, `skills/sub-agents/SKILL.md`) with the above behavior and limits.
6. Added targeted Gemini plan-mode failure guidance in error output when CLI reports missing `experimental.plan`.

## Test Coverage Added/Updated
1. Approval mode normalization (`autoEdit`/`auto-edit` -> `auto_edit`, invalid mode/type errors).
2. Include directories normalization and validation (dedupe, invalid path, >5 directories).
3. Gemini config normalization for new fields and ignored native-only field detection.
4. Agent loading/listing metadata assertions for new Gemini fields.
5. Command builder assertions for `--approval-mode` and repeated `--include-directories`.

## Verification Summary For This Round
1. Targeted tests: `60 passed` (`tests/test_run_subagent.py`) after follow-up model-routing tests.
2. Lint: `ruff check` passed for changed files.

## Follow-up Policy Update (User Preference)
1. Gemini routing policy changed to: default `pro`, small tasks only `flash`.
2. Added small-task heuristic + configurability:
   - Env override: `SUB_AGENT_SMALL_TASK_MAX_CHARS` (default `220`).
   - Explicit `model` in agent frontmatter still has highest priority.
   - `SUB_AGENT_GEMINI_MODEL` still force-overrides heuristic when set.
