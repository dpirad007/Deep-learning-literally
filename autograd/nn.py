from ion import Atom
import random


class Neuron:

    def __init__(self, nin):
        self.w = [Atom(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Atom(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum([wi*xi for wi, xi in zip(self.w, x)]) + self.b
        out = act.tanh()
        return out
