"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

from dataclasses import dataclass
import re
import subprocess
import time

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model

from rich.console import Console

console = Console(highlight=False)

class AgentConfig(BaseModel):
    # Check the config files in minisweagent/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    compiler_error_template: str
    test_script_error_template: str
    test_script_perf_template: str
    perf_diff_summary_template: str
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

class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.MAX_ATTEMPTS = 5
        self.reference_runtime = -1
        self.opt_attempts = []

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        initial_perf_report = self.run_profiler(reference=True)
        console.print(f"initial perf report:\n{initial_perf_report}", style="bright_cyan")
        self.extra_template_vars |= {"task": task, "perf_report": initial_perf_report, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        attempt = 0
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                attempt += 1
                console.print(f"[bold red] model thinks it is done: {self.messages[-1]} [/bold red]")
                # remove the last terminating message
                self.messages.pop()
                # self.add_message("user", str(e))
                profiler_msg = self.run_profiler(reference=False)
                console.print(f"[bold yellow] profiler message: {profiler_msg} [/bold yellow]")
                self.add_message("user", profiler_msg)
                if attempt >= self.MAX_ATTEMPTS:
                    console.print(f"agent is done after: {self.MAX_ATTEMPTS} attempts")
                    return self.print_best_attempt()
                    return type(e).__name__, str(e)
                # return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        console.print(f"[bold yellow]debug: did query and got response: {response['content']}[/bold yellow]")
        self.add_message("assistant", **response)
        return response

    def run_profiler(self, reference=False) -> str:
        if reference:
            PERF_SCRIPT_COMMAND = f"/profile.sh reference"
        else:
            PERF_SCRIPT_COMMAND = f"/profile.sh"
        BUILD_COMMAND = "/build.sh"
        console.print(f"run profiler build cmd: {BUILD_COMMAND}", style="magenta")
        build_output = self.env.execute(BUILD_COMMAND)
        if build_output["returncode"] != 0:
            if reference:
                print(build_output["output"])
                raise RuntimeError("failed to build reference repo")
            observation = self.render_template(self.config.compiler_error_template, output=build_output)
            console.print(f"build error: {build_output}", style="red")
            return observation
        perf_output = self.env.execute(PERF_SCRIPT_COMMAND)
        if perf_output["returncode"] != 0:
            if reference:
                print(perf_output["output"])
                raise RuntimeError("failed to run perf script as reference")
            return self.render_template(self.config.test_script_error_template, output=perf_output)
        else:
            # runtime = float(perf_output["output"].splitlines()[0])
            runtime = None
            for line in perf_output["output"].splitlines():
                if "_runtime:" in line:
                    runtime = float(line.split(":")[1])
            if runtime is None:
                raise RuntimeError("failed to parse runtime from perf script")
            perf_report = "\n".join(perf_output["output"].splitlines()[1:])
            if reference:
                self.reference_runtime = runtime
                self.opt_attempts.append(OptAttempt(speedup=1.0, perf_report=perf_report, diff=""))
                return perf_report
            else:
                speedup = self.reference_runtime / runtime
                prev_perf_report = self.opt_attempts[-1].perf_report
                prev_speedup = self.opt_attempts[-1].speedup

                perf_dict = {
                    "prev_speedup" : prev_speedup,
                    "curr_speedup" : speedup,
                    "prev_perf_report" : prev_perf_report,
                    "curr_perf_report" : perf_report
                }
                self.extra_template_vars |= perf_dict
                msg = self.render_template(self.config.perf_diff_summary_template)
                messages = []
                messages.append({"role" : "system", "content" : "You are a helpful assistant that can that can analyze performance profiles for computer programs."})
                messages.append({"role" : "user", "content": msg})
                perf_diff_summary = self.model.query(messages)["content"]

                perf_dict["perf_diff_summary"] = perf_diff_summary
                diff = self.env.execute("git diff", cwd="/workspace")["output"]
                self.opt_attempts.append(OptAttempt(speedup=speedup, perf_report=perf_report, diff=diff))
                self.extra_template_vars |= perf_dict
                return self.render_template(self.config.test_script_perf_template)

    def print_best_attempt(self):
        best_speedup = 0
        best_diff = ""
        best_idx = -1
        # do not use attempt 0 (base repo) when finding best diff
        for idx, obj in enumerate(self.opt_attempts[1:]):
            speedup = obj.speedup
            if speedup > best_speedup:
                best_speedup = speedup
                best_diff = obj.diff
                best_idx = idx

        console.print(f"[bold green] --- best attempt {best_idx}/{len(self.opt_attempts)} --- got speedup: {best_speedup} [/bold green]")
        console.print("--- diff ---")
        console.print(best_diff)

        # self.env.execute("")
        last_diff = self.env.execute("git diff", cwd="/workspace")["output"]
        return best_diff, last_diff

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        # print(f"DEBUG: got observation: {observation}")
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            # console.print(f"[bold yellow]found action from parse action: {actions[0].strip()} [/bold yellow]")
            return {"action": actions[0].strip(), **response}
        console.print(f"[bold red] no actions found from response: {response} [/bold red]")
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
