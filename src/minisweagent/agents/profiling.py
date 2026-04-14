"""Profiling agent for software performance optimization.

Iteratively profiles code, applies LLM-suggested optimizations, and measures speedup.
Ported from the original profiling_swefficiency agent, adapted to the v2 agent/environment contract.
"""

import re
from dataclasses import asdict, dataclass

import litellm
from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, Submitted

from minisweagent.models import get_model

TEST_CMD = "/run_tests.sh"
BUILD_CMD = "/build.sh"
PERF_SCRIPT = "/perf_script.py"
REFERENCE_PROFILING_CMD = "python /profile_prob_script.py --reference"
PROFILING_CMD = "python /profile_prob_script.py"


_HUNK_HEADER_RE = re.compile(r"^@@ .* @@ (.*)$")
# Matches `def foo` or `class Foo` (Python, Cython `cdef class`). The `\b`
# guards against partial matches inside Cython keywords like `cdef` / `cpdef`.
_DEF_CLASS_RE = re.compile(r"\b(?:def|class)\s+(\w+)")
# Fallback for C / C++ / Cython function signatures: an identifier immediately
# followed by `(`. We skip common control-flow keywords so that hunk contexts
# like `if (cond) {` don't get mis-parsed as a function name.
_C_FUNC_RE = re.compile(r"(\w+)\s*\(")
_NON_FUNC_KEYWORDS = frozenset({
    "if", "else", "elif", "while", "for", "switch", "return",
    "sizeof", "typeof", "case", "catch", "throw", "new", "delete",
})


def _extract_func_name(context: str) -> str | None:
    """Best-effort function or class name from a git hunk header context.

    Tries a Python/Cython `def|class` match first, then falls back to the
    leftmost `identifier(` pattern for C/C++/Cython function signatures.
    Returns None if nothing plausible is found.
    """
    m = _DEF_CLASS_RE.search(context)
    if m:
        return m.group(1)
    for cm in _C_FUNC_RE.finditer(context):
        name = cm.group(1)
        if name not in _NON_FUNC_KEYWORDS:
            return name
    return None


def _extract_modifications(diff: str) -> list[dict]:
    """Extract per-file modifications from a unified git diff.

    Parses `diff --git a/<path> b/<path>` lines to track the current file,
    and hunk headers `@@ -N,M +P,Q @@ <context>` to pull enclosing function
    or class names from git's "nearest enclosing function" context slot.
    Handles Python, Cython, and C/C++ hunk header conventions.

    Returns a list of {"file": str, "functions": [str]} dicts, one per
    modified file in the order they appear in the diff. Function names are
    deduped within each file while preserving order.
    """
    modifications: list[dict] = []
    current: dict | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) < 3:
                current = None
                continue
            path = parts[2]
            if path.startswith("a/"):
                path = path[2:]
            current = {"file": path, "functions": []}
            modifications.append(current)
        elif current is not None and line.startswith("@@ "):
            m = _HUNK_HEADER_RE.match(line)
            if not m:
                continue
            name = _extract_func_name(m.group(1))
            if name and name not in current["functions"]:
                current["functions"].append(name)
    return modifications


def _format_modifications(mods: list[dict]) -> str:
    """Format modification records as a short human-readable summary."""
    if not mods:
        return "(none)"
    parts: list[str] = []
    for m in mods:
        if m["functions"]:
            parts.append(f"{m['file']} ({', '.join(m['functions'])})")
        else:
            parts.append(m["file"])
    return "; ".join(parts)


@dataclass
class OptAttempt:
    runtime: float
    speedup: float
    perf_report: str
    diff: str


class ProfilingAgentConfig(AgentConfig):
    """Configuration for the profiling agent. Extends AgentConfig with profiling-specific templates."""

    timeout_template: str
    """Template for timeout error messages."""
    format_error_template: str
    """Template for format error messages when action parsing fails."""
    action_observation_template: str
    """Template for rendering action observations."""
    runtime_error_template: str
    """Template for build/test/profiler runtime error messages."""
    test_script_perf_template: str
    """Template for performance test results with speedup metrics."""
    perf_summary_template: str
    """Template for requesting profiler output summarization."""
    # summary_model_config: dict
    max_attempts: int = 5
    """Maximum number of optimization attempts before stopping."""
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    """Regex for extracting bash commands from model responses (fallback for non-tool-call models)."""


class ProfilingAgent(DefaultAgent):
    """Agent that optimizes software performance by iteratively profiling and applying LLM-suggested changes."""

    def __init__(self, model: Model, env: Environment, *, config_class: type = ProfilingAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.opt_attempts: list[OptAttempt] = []
        # self.summary_model = get_model(config=self.config.summary_model_config)
        self.summary_model = self.model

    def _render_template_with_vars(self, template: str, **extra_vars) -> str:
        """Render a Jinja2 template with standard template vars plus additional keyword arguments."""
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars(**extra_vars))

    def run(self, task: str = "", **kwargs) -> dict:
        """Run the profiling optimization loop. Returns dict with exit_status, submission, and opt_attempts."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.opt_attempts = []

        initial_perf_report = self._get_reference_profiler_report()
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(
                role="user",
                content=self._render_template_with_vars(
                    self.config.instance_template, initial_perf_report=initial_perf_report
                ),
            ),
        )

        attempt = 0
        while True:
            try:
                self.step()
            except Submitted:
                attempt += 1
                self.logger.info(f"Model reports completion (attempt {attempt}/{self.config.max_attempts})")
                # Remove the assistant message that triggered submission so it doesn't confuse the model
                if self.messages and self.messages[-1].get("role") == "assistant":
                    self.messages.pop()
                profiler_report = self._run_profiler(reference=False)
                self.logger.info("Obtained profiler report")  # set log level to debug to see
                self.logger.info(profiler_report)
                self.add_messages(self.model.format_message(role="user", content=profiler_report))
                if attempt >= self.config.max_attempts:
                    break
            except LimitsExceeded:
                self.logger.info(f"Limits exceeded. Optimization attempts: {len(self.opt_attempts) - 1}")
                break
            except litellm.exceptions.ContextWindowExceededError:
                self.logger.info(f"Context window exceeded. Optimization attempts: {len(self.opt_attempts) - 1}")
                break
            except litellm.exceptions.BadRequestError:
                self.logger.info(f"Bad request. Optimization attempts: {len(self.opt_attempts) - 1}")
                break
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)

        exit_extra = {
            "exit_status": "completed",
            "submission": "",
            "opt_attempts": self._get_opt_attempts(),
        }
        self.add_messages(self.model.format_message(role="exit", content="Profiling complete", extra=exit_extra))
        self.save(self.config.output_path)
        return exit_extra

    def step(self) -> list[dict]:
        """Query the model and execute the resulting action."""
        response = self.query()
        self.logger.info(f"LLM response: {response['content']}")
        # if response["content"] is None:
        #     self.logger.info(f"LLM response none?: {response}")

        return self._get_observation(response)

    def _get_observation(self, response: dict) -> list[dict]:
        """Execute actions from the response and return observation messages.

        Uses structured tool-call actions if available, otherwise falls back to
        regex-based action parsing from the response content.
        """
        # Try structured actions first (tool-call models)
        actions = response.get("extra", {}).get("actions", [])
        if actions:
            outputs = []
            for action in actions:
                cmd = action.get("command", "") if action.get("tool", "bash") == "bash" else ""
                if cmd.lower() == "true":
                    raise Submitted(self.model.format_message(
                        role="exit", content="",
                        extra={"exit_status": "Submitted", "submission": ""},
                    ))
                outputs.append(self._execute_action(action))
            return self.add_messages(
                *self.model.format_observation_messages(response, outputs, self.get_template_vars())
            )

        # Fallback: regex-based action parsing for text-based models
        parsed = self._parse_action(response)
        command = parsed["action"]
        if command.lower() == "true":
            raise Submitted(self.model.format_message(
                role="exit", content="",
                extra={"exit_status": "Submitted", "submission": ""},
            ))
        output = self.env.execute({"command": command})
        observation = self._render_template_with_vars(self.config.action_observation_template, output=output)
        return self.add_messages(self.model.format_message(role="user", content=observation))

    def _parse_action(self, response: dict) -> dict:
        """Parse a bash action from the response content using regex."""
        content = response.get("content", "")
        actions = re.findall(self.config.action_regex, content, re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}

        # Some models omit closing backticks
        patched = content + "\n```"
        actions = re.findall(self.config.action_regex, patched, re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}

        raise FormatError(self.model.format_message(
            role="user",
            content=self._render_template_with_vars(self.config.format_error_template, actions=actions),
        ))

    # --- Profiling methods ---

    def _get_reference_profiler_report(self) -> str:
        """Run the profiler on the reference (unmodified) code and return the report."""
        report = self._run_profiler(reference=True)
        self.logger.info("Reference profiler report obtained")
        self.logger.info(report)
        return report

    def _run_profiler(self, reference: bool = False) -> str:
        """Build, test, and profile the code. Returns a rendered report string."""
        if not reference:
            build_output = self.env.execute({"command": BUILD_CMD})
            if build_output["returncode"] != 0:
                header = (
                    "The changes you made produced the following error when building "
                    f"the repository running: `{BUILD_CMD}`."
                )
                return self._render_template_with_vars(
                    self.config.runtime_error_template, header=header, output=build_output
                )

            test_output = self.env.execute({"command": TEST_CMD})
            if test_output["returncode"] != 0:
                header = (
                    "The changes you made produced the following error when running "
                    f"the test suite using: `{TEST_CMD}`."
                )
                return self._render_template_with_vars(
                    self.config.runtime_error_template, header=header, output=test_output
                )

        profiler_cmd = REFERENCE_PROFILING_CMD if reference else PROFILING_CMD
        profiler_output = self.env.execute({"command": profiler_cmd}, cwd="/")
        if profiler_output["returncode"] != 0:
            if reference:
                raise RuntimeError(
                    f"Running profiler on reference should never error. Output: {profiler_output['output']}"
                )
            header = (
                "The changes you made produced the following error when running "
                f"the test script: `{PERF_SCRIPT}`."
            )
            return self._render_template_with_vars(
                self.config.runtime_error_template, header=header, output=profiler_output
            )

        runtime = self._get_runtime(profiler_output["output"])
        perf_report_summary = self._get_profiler_summary(profiler_output, runtime=runtime, reference=reference)

        if reference:
            self.opt_attempts.append(
                OptAttempt(runtime=runtime, speedup=1.0, perf_report=perf_report_summary, diff="")
            )
            return perf_report_summary

        speedup = self.opt_attempts[0].runtime / runtime
        self.logger.info(
            f"Speedup: {speedup:.2f}x (ref: {self.opt_attempts[0].runtime:.1f}ms, current: {runtime:.1f}ms)"
        )
        self.logger.info(
            f"Perf report summary:\n{perf_report_summary}\n"
        )
        diff = self.env.execute({"command": "git add -A && git diff --cached"}, cwd="/testbed")["output"]
        self.opt_attempts.append(
            OptAttempt(runtime=runtime, speedup=speedup, perf_report=perf_report_summary, diff=diff)
        )
        return self._render_template_with_vars(
            self.config.test_script_perf_template,
            perf_report_summary=perf_report_summary,
            speedup=speedup,
            ref_runtime=self.opt_attempts[0].runtime,
            current_runtime=runtime,
        )

    def _get_profiler_summary(self, profiler_output: dict, runtime: float, reference: bool) -> str:
        """Use the LLM to summarize profiler output.

        This is a pure text-completion call (no tool use), so we pass tools=[]
        to prevent the model from emitting tool calls and the parser from
        raising FormatError when none are found.
        """
        extra = {"profiler_output": profiler_output["output"], "current_runtime": runtime}
        if not reference and self.opt_attempts:
            extra["ref_runtime"] = self.opt_attempts[0].runtime
            extra["speedup"] = self.opt_attempts[0].runtime / runtime
            # Prior agent attempts (skip index 0, which is the reference baseline).
            prior = self.opt_attempts[1:]
            if prior:
                extra["previous_attempts"] = [
                    {
                        "speedup": att.speedup,
                        "modified": _format_modifications(_extract_modifications(att.diff)),
                    }
                    for att in prior
                ]
        msg = self._render_template_with_vars(self.config.perf_summary_template, **extra)
        messages = [
            {
                "role": "system",
                "content": "You are a performance analyst summarizing py-spy sampling profiles of Python programs.",
            },
            {"role": "user", "content": msg},
        ]
        response = self.summary_model.query(messages, tools=[]) 
        # response = self.model.query(messages, tools=[])
        # self.cost += response.get("extra", {}).get("cost", 0.0)
        return response["content"]

    def _get_runtime(self, output: str) -> float:
        """Extract runtime value from profiler output (looks for lines containing '_runtime')."""
        for line in output.splitlines():
            line = line.strip()
            if "_runtime" in line:
                try:
                    return float(line.split(":")[1])
                except Exception:
                    raise RuntimeError(f"Cannot parse runtime from line: {line}")
        raise RuntimeError(f"Cannot parse runtime from output: {output}")

    def _get_opt_attempts(self) -> list[dict]:
        """Return optimization attempts (excluding the reference baseline) as dicts."""
        return [asdict(attempt) for attempt in self.opt_attempts[1:]]

    def serialize(self, *extra_dicts) -> dict:
        """Extend serialization to include optimization attempt data."""
        return super().serialize({"opt_attempts": self._get_opt_attempts()}, *extra_dicts)
