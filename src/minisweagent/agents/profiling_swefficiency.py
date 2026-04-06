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

TEST_CMD = "/run_tests.sh"
BUILD_CMD = "/build.sh"
PERF_SCRIPT = "/perf_script.py"
REFERENCE_PROFILING_CMD = "python /profile_prob_script.py --reference"
PROFILING_CMD = "python /profile_prob_script.py"
console = Console(highlight=False)

class ProfilingAgentConfig(BaseModel):
    # Check the config files in minisweagent/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    runtime_error_template: str
    test_script_perf_template: str
    perf_summary_template: str
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
    runtime: float
    speedup: float
    perf_report: str
    diff: str

class ProfilingAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = ProfilingAgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.opt_attempts = []

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def get_reference_profiler_report(self):
        initial_perf_report = self.run_profiler(reference=True)
        console.print(f"initial perf report:\n{initial_perf_report}", style="bright_cyan")
        return initial_perf_report

    def run(self, task_type: str, **kwargs) -> list[dict]:
        """Run step() until agent is finished. Return exit status & message"""
        self.messages = []
        initial_perf_report = self.get_reference_profiler_report()
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template, initial_perf_report=initial_perf_report))
        attempt = 0
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except LimitsExceeded as e:
                console.print(f"COST EXCEEDED! Num opt attempts: {len(self.opt_attempts) - 1}", style="bright_red")
                return self.get_opt_attempts()
            except TerminatingException as e:
                attempt += 1
                console.print(f"model thinks it is done: {self.messages[-1]}", style="bright_green")
                # remove the last terminating message as to not confuse the model in its context
                self.messages.pop()
                profiler_report = self.run_profiler(reference=False)
                print(f"perf report:\n{profiler_report}")
                self.add_message("user", profiler_report)
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

    def test_profiler(self):
        raise RuntimeError("test profiler deprecated")
        profiler_cmd = REFERENCE_PROFILING_CMD

        build_output = self.env.execute(BUILD_CMD)
        if build_output["returncode"] != 0:
            console.print(build_output["output"])
            raise RuntimeError("failed to build reference repo")

        start = time.time()
        profiler_output = self.env.execute(profiler_cmd, cwd="/")
        end = time.time()
        # console.print(f"run profiler cmd: {profiler_cmd}", style="magenta")
        if profiler_output["returncode"] != 0:
            raise RuntimeError(f"running profiler on reference should never error. output: {profiler_output['output']}")
        
        timing_fname = REFERENCE_TIMING_FILE
        result = self.env.execute(f"cat {timing_fname}")
        try:
            runtime_dict = json.loads(result["output"])
        except Exception as e:
            console.print(f"output: {result['output']}")
            console.print(f"profiler output: {profiler_output['output']}")
            raise e

        console.print(f"profiler output:\n{profiler_output['output']}", style="bright_blue")

    def get_profiler_summary(self, profiler_output) -> str:
        msg = self.render_template(self.config.perf_summary_template, profiler_output=profiler_output["output"])
        messages = []
        messages.append({"role" : "system", "content" : "You are a helpful assistant that can that can analyze performance profiles for computer programs."})
        messages.append({"role" : "user", "content": msg})
        perf_report_summary = self.model.query(messages)["content"]
        return perf_report_summary

    def get_runtime(self, output: str) -> float:
        for line in output.splitlines():
            line = line.strip()
            if "_runtime" in line:
                try:
                    return float(line.split(":")[1])
                except Exception as e:
                    raise RuntimeError(f"cannot parse runtime from line: {line}")

        raise RuntimeError(f"cannot parse runtime from output: {output}")

    def run_profiler(self, reference=False) -> str:
        if not reference:
            build_output = self.env.execute(BUILD_CMD)
            if build_output["returncode"] != 0:
                console.print(f"build error: {build_output['output']}", style="red")
                build_error_header = f"The changes you made produced the following error when building the repository running: `{BUILD_CMD}`."
                return self.render_template(self.config.runtime_error_template, header=build_error_header, output=build_output)
                # return self.render_template(self.config.build_error_template, output=build_output)

            test_output = self.env.execute(TEST_CMD)
            print("--- run tests output returncode ---", test_output["returncode"])
            print(test_output["output"])
            if test_output["returncode"] != 0:
                print(f"test error: {test_output['output']}")
                test_error_header = f"The changes you made produced the following error when running the test suite using: `{TEST_CMD}`."
                return self.render_template(self.config.runtime_error_template, header=test_error_header, output=test_output)

        if reference:
            profiler_cmd = REFERENCE_PROFILING_CMD
        else:
            profiler_cmd = PROFILING_CMD

        profiler_output = self.env.execute(profiler_cmd, cwd="/")
        if profiler_output["returncode"] != 0:
            if reference:
                raise RuntimeError(f"running profiler on reference should never error. output: {profiler_output['output']}")
            else:
                test_script_error_header = f"The changes you made produced the following error when running the test script: `{PERF_SCRIPT}`."
                return self.render_template(self.config.runtime_error_template, header=test_script_error_header, output=profiler_output)
                # return self.render_template(self.config.test_script_error_template, output=profiler_output)

        runtime = self.get_runtime(profiler_output["output"])

        console.print(f"profiler output:\n{profiler_output['output']}", style="bright_blue")
        perf_report_summary = self.get_profiler_summary(profiler_output)
        if reference:
            self.opt_attempts.append(OptAttempt(runtime=runtime, speedup=1.0, perf_report=perf_report_summary, diff=""))
            return perf_report_summary
            
        speedup = self.opt_attempts[0].runtime / runtime
        console.print(f"speedup: {speedup}, ref runtime: {self.opt_attempts[0].runtime}, my runtime: {runtime}", style="bright_green")
        diff = self.env.execute("git add -A && git diff --cached", cwd="/testbed")["output"]
        self.opt_attempts.append(OptAttempt(runtime=runtime, speedup=speedup, perf_report=perf_report_summary, diff=diff))
        return self.render_template(self.config.test_script_perf_template, perf_report_summary=perf_report_summary, speedup=speedup)

    def get_opt_attempts(self) -> list[dict]:
        attempts = []
        for attempt in self.opt_attempts[1:]:
            attempts.append(asdict(attempt))
        return attempts

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        is_run_tests_cmd = TEST_CMD in self.parse_action(response)["action"]
        start = time.time()
        output = self.execute_action(self.parse_action(response))
        end = time.time()
        observation = self.render_template(self.config.action_observation_template, output=output)
        if is_run_tests_cmd:
            print(f"-- run tests observation took: {end - start}s --")
            print(output["output"])
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
