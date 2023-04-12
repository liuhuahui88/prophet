import numpy as np
import pandas as pd

from prophet.utils.graph import Graph


class Log(Graph.Function):

    def compute(self, inputs):
        return np.log(inputs[0])


class Sign(Graph.Function):

    def compute(self, inputs):
        return np.sign(inputs[0])


class Positive(Graph.Function):

    def compute(self, inputs):
        return (np.sign(inputs[0]) + 1) / 2


class Negative(Graph.Function):

    def compute(self, inputs):
        return (1 - np.sign(inputs[0])) / 2


class Add(Graph.Function):

    def compute(self, inputs):
        return pd.DataFrame({'Add': inputs[0].iloc[:, 0] + inputs[1].iloc[:, 0]})


class Sub(Graph.Function):

    def compute(self, inputs):
        return pd.DataFrame({'Sub': inputs[0].iloc[:, 0] - inputs[1].iloc[:, 0]})


class Mul(Graph.Function):

    def compute(self, inputs):
        return pd.DataFrame({'Mul': inputs[0].iloc[:, 0] * inputs[1].iloc[:, 0]})


class Div(Graph.Function):

    def compute(self, inputs):
        return pd.DataFrame({'Div': inputs[0].iloc[:, 0] / inputs[1].iloc[:, 0]})
