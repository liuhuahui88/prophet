from abc import abstractmethod

import scipy as sp
import pandas as pd

from prophet.utils.graph import Graph


class FeatureManager:

    def __init__(self, names, graph=None):
        self.graph = graph if graph is not None else self.create_default_graph()
        self.names = names

    def get(self, features: pd.DataFrame):
        ctx = {'price': features}
        return self.graph.compute(self.names, ctx)

    @staticmethod
    def create_default_graph():
        graph = Graph()
        graph.register('price')
        graph.register('gain', FeatureManager.Gain(), ['price'])
        graph.register('mean', FeatureManager.Mean('Close', 'Mean', [5, 10, 20, 30]), ['price'])
        graph.register('std', FeatureManager.Std('Close', 'Std', [5, 10, 20, 30]), ['price'])
        graph.register('skew', FeatureManager.Skew('Close', 'Skew', [5, 10, 20, 30]), ['price'])
        return graph

    class Gain(Graph.Function):

        def compute(self, inputs):
            price = inputs[0]
            gain = pd.DataFrame()
            for i in range(1, 30):
                gain['gain{}'.format(i)] = price['Close-0'] / price['Close-{}'.format(i)]
            return gain

    class Agg(Graph.Function):

        def __init__(self, input_prefix, output_prefix, window_sizes):
            self.window_sizes = window_sizes
            self.input_prefix = input_prefix
            self.output_prefix = output_prefix

        def compute(self, inputs):
            input_df = inputs[0]
            output_df = pd.DataFrame()
            for window_size in self.window_sizes:
                data = input_df[['{}-{}'.format(self.input_prefix, i) for i in range(window_size)]]
                output_df['{}-{}'.format(self.output_prefix, window_size)] = self.aggregate(data)
            return output_df

        @abstractmethod
        def aggregate(self, data):
            pass

    class Mean(Agg):

        def aggregate(self, data):
            return data.mean(axis=1)

    class Std(Agg):

        def aggregate(self, data):
            return data.std(axis=1)

    class Skew(Agg):

        def aggregate(self, data):
            return sp.stats.skew(data, axis=1)
