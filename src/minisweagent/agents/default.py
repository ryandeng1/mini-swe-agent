"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model

from rich.console import Console

BUILD_COMMAND = "pip install -e . --no-build-isolation"
GET_FINGERPRINT_CMD = "git diff | sha1sum"

console = Console(highlight=False)

@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    compiler_error_template: str = ""
    test_script_error_template: str = ""
    test_script_perf_template: str = ""
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
    diff: str
    speedup: float

class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.fingerprint = ""
        self.MAX_ATTEMPTS = 3
        self.reference_runtime = -1
        self.opt_attempts = []

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        # self.fingerprint = self.env.execute(GET_FINGERPRINT_CMD)["output"]
        self.run_profiler(reference=True)
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
                    self.print_best_attempt()
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

    def did_edit_files(self) -> bool:
        output = self.env.execute(GET_FINGERPRINT_CMD)["output"]
        if output != self.fingerprint:
            self.fingerprint = output
            return True
        return False

    def run_profiler(self, reference=False) -> str:
        PERF_SCRIPT_COMMAND = f"python perf_script.py --reference {reference}"
        DIFF_COMMAND = "git diff"
        console.print(f"run profiler build")
        build_output = self.env.execute(BUILD_COMMAND)
        if build_output["returncode"] != 0:
            observation = self.render_template(self.config.compiler_error_template, output=build_output)
            return observation
        console.print(f"run profiler perf script")
        perf_output = self.env.execute(PERF_SCRIPT_COMMAND)
        if perf_output["returncode"] != 0:
            observation = self.render_template(self.config.test_script_error_template, output=perf_output)
        else:
            runtime = float(perf_output["output"])
            if reference:
                self.reference_runtime = runtime
                observation = ""
            else:
                diff_output = self.env.execute(DIFF_COMMAND)
                diff = diff_output["output"]
                speedup = self.reference_runtime / runtime
                perf_dict = {"speedup" : speedup, "diff" : diff}
                self.opt_attempts.append(OptAttempt(diff=diff, speedup=speedup))
                observation = self.render_template(self.config.test_script_perf_template, output=perf_dict)
        return observation

    def print_best_attempt(self):
        best_speedup = 0
        best_diff = ""
        best_idx = -1
        for idx, obj in enumerate(self.opt_attempts):
            speedup = obj.speedup
            if speedup > best_speedup:
                best_speedup = speedup
                best_diff = obj.diff
                best_idx = idx

        console.print(f"[bold green] --- best attempt {best_idx}/{len(self.opt_attempts)} --- got speedup: {best_speedup} [/bold green]")
        console.print("--- diff ---")
        console.print(best_diff)

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        # print(f"DEBUG: got observation: {observation}")
        self.add_message("user", observation)
        return output
        action = self.parse_action(response)
        output = self.execute_action(action)
        bash_cmd = action["action"]
        # console.print(f"output from cmd: {bash_cmd} is: {output['output']}")
        if self.did_edit_files():
            build_output = self.env.execute(BUILD_COMMAND)
            console.print(f"[bold blue] len cmd: {len(bash_cmd)}, cmd: {bash_cmd} is an edit command, build return code: {build_output['returncode']}[/bold blue]")
            try:
                if build_output["returncode"] != 0:
                    build_output["edit"] = bash_cmd
                    observation = self.render_template(self.config.compiler_error_template, output=build_output)
                    self.add_message("user", observation)
                    console.print(f"[bold red] edit resulted in compiler error [/bold red]")
                    console.print(observation)
                    console.print(f"[bold red] end observation [/bold red]")
                    return build_output
                else:
                    observation = self.render_template(self.config.action_observation_template, output=output)
                    self.add_message("user", observation)
                    return output
            except Exception as e:
                console.print(f"len: {len(build_output)}, build output: {build_output}")
                raise e
        else:
            observation = self.render_template(self.config.action_observation_template, output=output)
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
            # print(f"debug: got output from action: {output}")
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
