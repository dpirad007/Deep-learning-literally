import math
import graphviz


class Atom:

    def __init__(self, data, children=(), _op="", label=""):
        self.data = data
        self._prev = set(children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Atom(data={self.data})"

    def __add__(self, other):

        other = other if isinstance(other, Atom) else Atom(other)
        o = Atom(self.data + other.data, (self, other), "+")

        def backward():
            print("__add__ backward invoked")
            print(f"Self: {self}, other: {other}, o: {o}")
            self.grad += 1.0 * o.grad
            other.grad += 1.0 * o.grad
            print(
                f"self.grad: {self.grad}, other.grad: {other.grad}, o.grad: {o.grad}")

        o._backward = backward

        return o

    def __radd__(self, other):  # other + self
        return self + other

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        o = Atom(self.data ** other, (self,), f"**{other}")

        def backward():
            self.grad += (other*(self.data**(other-1))) * o.grad
        o._backward = backward

        return o

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):

        other = other if isinstance(other, Atom) else Atom(other)
        o = Atom(self.data * other.data, (self, other), "*")

        def backward():
            print("__mul__ backward invoked")
            print(f"Self: {self}, other: {other}, o: {o}")

            self.grad += other.data * o.grad
            other.grad += self.data * o.grad
            print(
                f"self.grad: {self.grad}, other.grad: {other.grad}, o.grad: {o.grad}")

        o._backward = backward

        return o

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):
        return self*(other**-1)

    def __rtruediv__(self, other):
        return other*(self**-1)

    def exp(self):
        t = math.exp(self.data)
        o = Atom(t, (self,), "exp")

        def backward():
            print("exp backward invoked")
            print(f"Self: {self}, o: {o}")
            self.grad += t * o.grad
            print(
                f"self.grad: {self.grad}, o.grad: {o.grad}")

        o._backward = backward
        return o

    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        o = Atom(t, (self,), "tanh")

        def backward():
            print("tanh backward invoked")
            print(f"Self: {self}, o: {o}")
            self.grad += (1 - t**2) * o.grad
            print(
                f"self.grad: {self.grad}, o.grad: {o.grad}")

        o._backward = backward
        return o

    def backward(self):

        self.grad = 1.0

        topo = []
        vis = set()

        def sorttopo(node):
            if node not in vis:
                vis.add(node)
                for child in node._prev:
                    sorttopo(child)
                topo.append(node)
        sorttopo(self)
        for node in reversed(topo):
            node._backward()


def traverse(root):
    nodes, edges = set(), set()

    def build(node):
        if node not in nodes:
            nodes.add(node)
        for child in node._prev:
            edges.add((child, node))
            build(child)
    build(root)
    return nodes, edges


def graph(root):

    dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})

    nodes, edges = traverse(root)

    for node in nodes:
        uid = str(id(node))
        dot.node(name=uid, label="%s | data %.4f | grad %.4f" %
                 (node.label, node.data, node.grad), shape="record")

    for e1, e2 in edges:
        dot.edge(str(id(e1)), str(id(e2)))
    return dot
