import pandas as pd

from prophet.utils.graph import Graph


class Get(Graph.Function):

    def __init__(self, input_name, output_name=None):
        self.input_name = input_name
        self.output_name = output_name if output_name is not None else input_name

    def compute(self, inputs):
        output_df = pd.DataFrame()
        output_df[self.output_name] = inputs[0][self.input_name]
        return output_df


class Fill(Graph.Function):

    def __init__(self, default_value):
        self.default_value = default_value

    def compute(self, inputs):
        return inputs[0].fillna(value=self.default_value)


class Merge(Graph.Function):

    def __init__(self, functions):
        self.functions = functions

    def compute(self, inputs):
        return pd.concat([function.compute(inputs) for function in self.functions], axis=1)
