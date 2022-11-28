from ion import Atom
import random


class Neuron:

    def __init__(self, nin):
        self.w = [Atom(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Atom(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum([wi*xi for wi, xi in zip(self.w, x)]) + self.b
        o = act.tanh()
        return o

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        o = [n(x) for n in self.neurons]
        return o[0] if len(o) == 1 else o

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:

    def __init__(self, nin, nouts):
        nlist = [nin]+nouts
        self.layers = [Layer(nlist[i], nlist[i+1])
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            o = layer(x)
        return o

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
