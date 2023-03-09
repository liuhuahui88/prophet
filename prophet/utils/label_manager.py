import pandas as pd

from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class LabelManager:

    def __init__(self, names, graph=None):
        self.graph = graph if graph is not None else self.create_default_graph()
        self.names = names

    def get(self, labels: pd.DataFrame):
        ctx = {'action': labels}
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
