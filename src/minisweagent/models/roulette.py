import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

from minisweagent import Model
from minisweagent.models import get_model


@dataclass
class RouletteModelConfig:
    model_kwargs: list[dict]
    model_name: str = "roulette"


class RouletteModel:
    def __init__(self, *, config_class: Callable = RouletteModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.models = [get_model(config=config) for config in self.config.model_kwargs]

    @property
    def cost(self) -> float:
        return sum(model.cost for model in self.models)

    @property
    def n_calls(self) -> int:
        return sum(model.n_calls for model in self.models)

    def get_template_vars(self) -> dict:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def select_model(self) -> Model:
        return random.choice(self.models)

    def query(self, *args, **kwargs) -> dict:
        model = self.select_model()
        response = model.query(*args, **kwargs)
        response["model_name"] = model.config.model_name
        return response


@dataclass
class InterleavingModelConfig:
    model_kwargs: list[dict]
    model_name: str = "interleaving"
    sequence: list[int] | None = None
    """If set to 0, 0, 1, we will return the first model 2 times, then the second model 1 time,
    then the first model again, etc."""


class InterleavingModel(RouletteModel):
    def __init__(self, *, config_class: Callable = InterleavingModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def select_model(self) -> Model:
        if self.config.sequence is None:
            i_model = self.n_calls % len(self.models)
        else:
            i_model = self.config.sequence[self.n_calls % len(self.config.sequence)]
        return self.models[i_model]


@dataclass
class FirstThenModelConfig:
    model_kwargs: list[dict]
    calls_per_model: list[int]
    """When we have 3 models, and calls_per_model is [2, 1], we will return the first model 2 times,
    then the second model 1 time, and finally the 3rd model all the remaining times."""
    model_name: str = "first_then"

    def __post_init__(self):
        if len(self.model_kwargs) != len(self.calls_per_model) - 1:
            raise ValueError("calls_per_model must have one less element than the number of models")


class FirstThenModel(InterleavingModel):
    def select_model(self) -> Model:
        calls_so_far = 0
        for i, calls in enumerate(self.config.calls_per_model):
            calls_so_far += calls
            if self.n_calls < calls_so_far:
                return self.models[i]
        return self.models[-1]
