import numpy as np
import pandas as pd

from prophet.utils.graph import Graph


class Clip(Graph.Function):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def compute(self, inputs):
        return np.clip(inputs[0], self.min_value, self.max_value)


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


class Indicator(Graph.Function):
    def __init__(self, cond):
        self.cond = cond

    def compute(self, inputs):
        line = inputs[0].iloc[:, 0].tolist()

        result = []
        for i in range(len(line)):
            if self.cond(line[i]):
                result.append(1)
            else:
                result.append(0)

        df = pd.DataFrame({'Indic': result})
        return df


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
