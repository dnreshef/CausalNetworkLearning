import numpy as np

"""
Must use the first k dimensions or the last k dimensions.
For line and plane, must pass through 0.
"""

def make_func(a):
    def func(sample_size, D, noise):
        r, fx = a()
        x = (r[1] - r[0]) * np.random.rand(sample_size, D) + r[0]
        y = np.apply_along_axis(lambda x: [fx(x) + noise * np.random.standard_normal()], 1, x)
        return x, y
    return func

@make_func
def _parabola():
    return (-1, 1), lambda x: 1 - (x[-1] ** 2)
parabola = (_parabola, lambda x: 1 - x[-1] ** 2, lambda x: 1, np.array([-1]))

@make_func
def _paraboloid():
    return (-1, 1), lambda x: 1 - (x[-1] ** 2) - (x[-2] ** 2)
paraboloid = (_paraboloid, lambda x: 1 - x[-2] ** 2 - x[-1] ** 2, lambda x: 1, np.array([-2, -1]))

@make_func
def _sine():
    return (-0.5, 3), lambda x: np.sin(x[-1] * np.pi) - 0.5 * x[-1]
sine = (_sine, lambda x: (np.sin(x[-1] * np.pi) - 0.5 * x[-1]), lambda x: 0.7627, np.array([-1]))

@make_func
def _line():
    return (-2, 2), lambda x: x[0]
line = (_line, lambda x: x[0], lambda x: x[0] + 2, np.array([0])) 

@make_func
def _plane():
    return (-2, 2), lambda x: x[0] + 2 * x[1]
plane = (_plane, lambda x: x[0] + 2 * x[1], lambda x: x[0] + 2 * x[1] + 6, np.array([0, 1])) 

@make_func
def _hyperbolic():
    return (-1, 1), lambda x: x[0] * x[1]
#hyperbola = (_hyperbolic, lambda x: x[0] * x[1],

# Deprecated

#@make_func
#def _corrugated_curve():
#    return (-7, 7), lambda x: 8 * np.sin(np.pi * x[-2]/2) + 4 * x[-2] - (x[-1] ** 2)
#corrugated_curve = (_corrugated_curve,
#                    lambda x: np.abs(28.41 - (8 * np.sin(np.pi * x[-2]/2) + 4 * x[-2] - (x[-1] ** 2))),
#                    np.array([-2, -1]))

