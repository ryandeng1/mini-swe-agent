import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from jinja2 import Template

from minisweagent import Model
from minisweagent.models import get_model
from minisweagent.utils.log import logger


@dataclass
class SampleModelConfig:
    decider_template: str
    decider_model_kwargs: dict
    sample_model_kwargs: list[dict]
    model_name: str = "sample"
    n_samples: int = 10
    model_kwargs: Any = None  # ignored
    n_threads: int = 1


class SampleModel(Model):
    def __init__(self, *, config_class: Callable = SampleModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.decider_model = get_model(config=self.config.decider_model_kwargs)
        self.sample_models = [get_model(config=config) for config in self.config.sample_model_kwargs]

    @property
    def n_calls(self) -> int:
        # Don't include the sample model calls in the total number of calls
        return self.decider_model.n_calls

    @property
    def cost(self) -> float:
        return self.decider_model.cost + sum(model.cost for model in self.sample_models)

    def get_template_vars(self) -> dict:
        return {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "n_samples": self.config.n_samples,
        }

    def _process_sample(self, i_sample: int, messages: list[dict]) -> str | None:
        """Process a single sample and return the extracted action or None if invalid."""
        model = self.sample_models[i_sample % len(self.sample_models)]
        response = model.query(messages)
        _actions = re.findall(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
        if len(_actions) != 1:
            logger.warning(f"Sample {i_sample} returned {len(_actions)} actions, expected 1")
            return None
        return _actions[0].strip()

    def _get_samples(self, messages: list[dict]) -> list[dict]:
        actions = []
        with ThreadPoolExecutor(max_workers=self.config.n_threads) as executor:
            futures = [
                executor.submit(self._process_sample, i_sample, messages) for i_sample in range(self.config.n_samples)
            ]
            for future in futures:
                action = future.result()
                if action is not None:
                    actions.append(action)
        actions = list(set(actions))
        logger.debug(f"Got {len(actions)} unique actions from {self.config.n_samples} samples")
        return actions

    def query(self, messages: list[dict]) -> dict:
        actions = self._get_samples(messages)
        extra_prompt = Template(self.config.decider_template).render(actions=actions)
        messages = [*messages, {"role": "user", "content": extra_prompt}]
        result = self.decider_model.query(messages)
        result["extra_prompt"] = extra_prompt
        return result
