import pandas as pd

from prophet.fg.utils.action_generator import ActionGenerator
from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class Oracle(Graph.Function):

    def __init__(self, commission_rate):
        self.action_generator = ActionGenerator(commission_rate)

    def compute(self, inputs):
        actions, advantages, cum_gains = self.action_generator.generate(inputs[0].iloc[:, 0])
        df = pd.DataFrame({
            'EmptyAction': actions[Const.EMPTY], 'FullAction': actions[Const.FULL],
            'EmptyAdvantage': advantages[Const.EMPTY], 'FullAdvantage': advantages[Const.FULL],
            'EmptyCumGain': cum_gains[Const.EMPTY], 'FullCumGain': cum_gains[Const.FULL],
        })
        return df
