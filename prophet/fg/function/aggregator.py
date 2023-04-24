from abc import abstractmethod

from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class Aggregator(Graph.Function):

    def __init__(self, window_size=None, alpha=None, mode=Const.PAST):
        self.window_size = window_size
        self.alpha = alpha
        self.mode = mode

    def compute(self, inputs):
        input_df = inputs[0]
        if self.mode == Const.FUTURE:
            input_df = input_df.iloc[::-1]
        if self.alpha is not None:
            window = input_df.ewm(alpha=self.alpha, min_periods=1)
        else:
            window = input_df.rolling(self.window_size, min_periods=1, center=(self.mode == Const.CENTER))
        output_df = self.aggregate(window)
        if self.mode == Const.FUTURE:
            output_df = output_df.iloc[::-1]
        return output_df

    @abstractmethod
    def aggregate(self, data):
        pass


class Rank(Aggregator):

    def aggregate(self, data):
        return data.rank(pct=True)


class Spread(Aggregator):

    def aggregate(self, data):
        return data.max() - data.min()


class Max(Aggregator):

    def aggregate(self, data):
        return data.max()


class Min(Aggregator):

    def aggregate(self, data):
        return data.min()


class Mean(Aggregator):

    def aggregate(self, data):
        return data.mean()


class Std(Aggregator):

    def aggregate(self, data):
        return data.std().fillna(0)


class Skew(Aggregator):

    def aggregate(self, data):
        return data.skew().fillna(0)


class Kurt(Aggregator):

    def aggregate(self, data):
        return data.kurt().fillna(0)
