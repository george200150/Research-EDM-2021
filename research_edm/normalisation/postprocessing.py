import math


def log(x):
    x = float(x)
    return math.log10(x + 1)


def asinh(x):
    x = float(x)
    return math.log(x + math.sqrt(x * x + 1))


def identic(x):
    return x


class Wrap:
    def __init__(self, f):
        self.f = f
        self.name = f.__name__

    def __call__(self, x):
        if x == "":
            return 0.0
        else:
            return self.f(x)

# transform = Wrap(log)
# transform = Wrap(asinh)
# transform = Wrap(identic)
