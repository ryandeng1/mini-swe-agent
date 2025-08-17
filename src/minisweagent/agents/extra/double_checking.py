from collections.abc import Callable
from dataclasses import dataclass

from minisweagent.agents.default import AgentConfig, DefaultAgent


@dataclass
class DoubleCheckingAgentConfig(AgentConfig):
    submit_unlocked_template: str = "Please rerun the command and submit the final output."


class DoubleCheckingAgent(DefaultAgent):
    def __init__(self, *, config_class: Callable = DoubleCheckingAgentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self.submit_unlocked = False

    def has_finished(self, output: dict[str, str]):
        if self.submit_unlocked:
            super().has_finished(output)
        else:
            self.submit_unlocked = True
            self.add_message(role="user", content=self.config.submit_unlocked_template)
