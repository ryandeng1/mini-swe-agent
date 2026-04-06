"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

from dataclasses import dataclass, asdict
import json
import jsonlines
import os
import re
import subprocess
import time

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model

from rich.console import Console
import statistics

REFERENCE_TIMING_FILE = "/reference_timing.json"
OPTIMIZED_TIMING_FILE = "/optimized_timing.json"
console = Console(highlight=False)

class TimingAgentConfig(BaseModel):
    # Check the config files in minisweagent/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    build_error_template: str
    test_script_error_template: str
    max_attempts: int
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0

class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""

@dataclass
class OptAttempt:
    speedup: float
    perf_report: str
    diff: str

class TimingAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = TimingAgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.reference_runtime = None
        self.opt_attempts = []

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def get_timing(self, reference=False) -> str:
        BUILD_COMMAND = "/build.sh"
        build_output = self.env.execute(BUILD_COMMAND)
        if build_output["returncode"] != 0:
            if reference:
                console.print(build_output["output"])
                raise RuntimeError("failed to build reference repo")
            observation = self.render_template(self.config.build_error_template, output=build_output)
            console.print(f"build error: {build_output}", style="red")
            return observation

        if reference:
            perf_script_output = self.env.execute("python perf_script.py --reference", cwd="/")
        else:
            perf_script_output = self.env.execute("python perf_script.py", cwd="/")

        if perf_script_output["returncode"] != 0:
            if reference:
                console.print(perf_script_output["output"])
                raise RuntimeError("perf script failed on reference which should never happen")
            return self.render_template(self.config.test_script_error_template, output=perf_script_output)

        runtime = None
        for line in perf_script_output["output"].splitlines():
            if "Execution" in line or "Mean" in line:
                runtime = float(line.split(":")[1])

        if runtime is None:
            console.print(perf_script_output["output"])
            raise RuntimeError("failed to get runtime from successful run of perf script")

        if reference:
            self.reference_runtime = runtime
            self.opt_attempts.append(OptAttempt(speedup=1.0, perf_report="", diff=""))
            return f"I received the following timing output from running the perf script. Runtime: {runtime}s."
        else:
            diff = self.env.execute("git add -N . && git diff HEAD", cwd="/testbed")["output"]
            assert self.reference_runtime is not None
            speedup = self.reference_runtime / runtime 
            self.opt_attempts.append(OptAttempt(speedup=speedup, perf_report="", diff=diff))
            return f"After incorporating your changed and rebuilding the repository, I received the following timing output from running the perf script. Runtime: {runtime}s.\n Please continue to try to continue improving performance.\n"

    def get_reference_timing(self):
        initial_timing_msg = self.get_timing(reference=True)
        console.print(f"initial timing:\n{initial_timing_msg}", style="bright_cyan")
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template, initial_timing=initial_timing_msg))

    def run(self, task_type: str, **kwargs) -> list[dict]:
        """Run step() until agent is finished. Return exit status & message"""
        self.messages = []
        self.get_reference_timing()
        attempt = 0
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                attempt += 1
                console.print(f"[bold red] model thinks it is done: {self.messages[-1]} [/bold red]")
                # remove the last terminating message as to not confuse the model in its context
                self.messages.pop()
                timing_msg = self.get_timing(reference=False)
                self.add_message("user", timing_msg)
                if attempt >= self.config.max_attempts:
                    return self.get_opt_attempts()

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        console.print(f"debug: got response: {response['content']}", style="bright_yellow")
        self.add_message("assistant", **response)
        return response

    def get_opt_attempts(self) -> list[dict]:
        attempts = []
        for attempt in self.opt_attempts[1:]:
            attempts.append(asdict(attempt))
        return attempts

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            # console.print(f"[bold yellow]found action from parse action: {actions[0].strip()} [/bold yellow]")
            return {"action": actions[0].strip(), **response}
        
        # hack, GPT 5.1 likes to output text that is just missing the backticks at the end for a lot of its python commands
        add_three_backticks = response["content"] + "\n```"
        actions = re.findall(self.config.action_regex, add_three_backticks, re.DOTALL)
        if len(actions) == 1:
            response["content"] = add_three_backticks
            console.print(f"adding three backticks at the end worked")
            return {"action": actions[0].strip(), **response}
        console.print(f"did not find a single action from response.\n{response['content']}", style="bright_red")
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        if action["action"].lower() == "true":
            raise Submitted("LLM echos empty true block")
        return output | {"action": action["action"]}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
