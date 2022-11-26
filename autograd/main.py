from ion import *

# simple example

# a = Atom(1.0, label="a")
# b = Atom(7.0, label="b")
# f = Atom(4.0, label="f")
# e = Atom(3.0, label="e")


# c = a+b
# c.label = "c"

# d = c*e
# d.label = "d"

# L = d+f
# L.label = "L"
# graph(L).view()

# neuron example
# x1 = Atom(2.0, label="x1")
# w1 = Atom(-3.0, label="w1")
# x2 = Atom(0.0, label="x2")
# w2 = Atom(1.0, label="w2")
# b = Atom(6.8813735870195432, label='b')

# x1w1 = x1*w1
# x1w1.label = "x1w1"
# x2w2 = x2*w2
# x2w2.label = "x2w2"

# x1w1x2w2 = x1w1+x2w2
# x1w1x2w2.label = "x1w1x2w2"

# n = x1w1x2w2 + b
# n.label = "n"

# o = n.tanh()
# o.label = "tanh"

# o.backward()

# graph(o).view()

a = Atom(10.0)
b = Atom(20.0)

print(a**1)
