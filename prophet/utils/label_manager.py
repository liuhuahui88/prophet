import pandas as pd

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

        ctx = {'action': action}
        return self.graph.compute(self.names, ctx)

    @staticmethod
    def create_default_graph():
        graph = Graph()
        graph.register('action')
        graph.register('action_when_empty', LabelManager.FillNaN(Const.ASK), ['action'])
        graph.register('action_when_full', LabelManager.FillNaN(Const.BID), ['action'])
        return graph

    class FillNaN(Graph.Function):

        def __init__(self, default_value):
            self.default_value = default_value

        def compute(self, inputs):
            return inputs[0].copy().fillna(value=self.default_value)
