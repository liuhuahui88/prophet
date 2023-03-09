import pandas as pd
import numpy as np

from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class LabelManager:

    def __init__(self, names, graph=None):
        self.names = names
        self.graph = graph if graph is not None else self.create_default_graph()

    def extract(self, history: pd.DataFrame, window_size):
        action = pd.DataFrame()
        action['Action'] = history['Action']
        action = action[window_size - 1:]
        action = action.reset_index(drop=True)

        price = pd.DataFrame()
        for i in range(window_size):
            price['Close-{}'.format(i)] = history.shift(-i)['Close']
        price = price[window_size - 1:]
        price = price.reset_index(drop=True)

        ctx = {'action': action, 'price': price}
        return self.graph.compute(self.names, ctx)

    @staticmethod
    def create_default_graph():
        graph = Graph()
        graph.register('action')
        graph.register('price')
        graph.register('action_when_empty', LabelManager.FillNaN(Const.ASK), ['action'])
        graph.register('action_when_full', LabelManager.FillNaN(Const.BID), ['action'])
        graph.register('next_direction', LabelManager.NextDirection(), ['price'])
        graph.register('next_log_gain', LabelManager.NextLogGain(), ['price'])
        return graph

    class FillNaN(Graph.Function):

        def __init__(self, default_value):
            self.default_value = default_value

        def compute(self, inputs):
            return inputs[0].copy().fillna(value=self.default_value)

    class NextDirection(Graph.Function):

        def compute(self, inputs):
            price = inputs[0]
            next_direction = pd.DataFrame()
            next_direction['NextDirection'] = price['Close-1'] > price['Close-0']
            next_direction['NextDirection'].apply(lambda x: 1 if x else 0)
            return next_direction

    class NextLogGain(Graph.Function):

        def compute(self, inputs):
            price = inputs[0]
            next_log_gain = pd.DataFrame()
            next_log_gain['NextLogGain'] = np.log((price['Close-1'] / price['Close-0']).fillna(value=1))
            return next_log_gain
