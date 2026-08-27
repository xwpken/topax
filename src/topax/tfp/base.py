from abc import ABC, abstractmethod


class Transform(ABC):

    @abstractmethod
    def __call__(self, x):
        ...

    def __add__(self, other):
        return Pipeline([self]) + other


class Pipeline(Transform):

    def __init__(self, steps=None):
        self.steps = steps or []

    def __call__(self, x):
        for step in self.steps:
            x = step(x)
        return x

    def __add__(self, other):
        if isinstance(other, Pipeline):
            return Pipeline(self.steps + other.steps)
        return Pipeline(self.steps + [other])

    def __getitem__(self, idx):
        return self.steps[idx]

    def __len__(self):
        return len(self.steps)
