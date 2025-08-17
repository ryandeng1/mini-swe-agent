from collections.abc import Callable
from dataclasses import dataclass

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent, Submitted


@dataclass
class DoubleCheckingAgentConfig(AgentConfig):
    submit_unlocked_template: str = "Please rerun the command and submit the final output."


class DoubleCheckingAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = DoubleCheckingAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.submit_unlocked = False

    def has_finished(self, output: dict[str, str]):
        try:
            super().has_finished(output)
        except Submitted:
            if self.submit_unlocked:
                raise
            else:
                self.submit_unlocked = True
                self.add_message(role="user", content=self.config.submit_unlocked_template)
