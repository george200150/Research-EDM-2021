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


class Log(Wrap):
    def __init__(self):
        super(Log, self).__init__(log)


class Asinh(Wrap):
    def __init__(self):
        super(Asinh, self).__init__(asinh)


class Identic(Wrap):
    def __init__(self):
        super(Identic, self).__init__(identic)


preprocessings_listing = [Log(), Asinh(), Identic()]  # could be singleton
underscore_preprocs_names = ["_" + x.name for x in preprocessings_listing]
default_t = Identic()
