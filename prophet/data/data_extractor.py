from abc import abstractmethod

import pandas as pd
import numpy as np

from prophet.utils.action_generator import ActionGenerator
from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class DataExtractor:

    def __init__(self, names, graph=None):
        self.names = names
        self.graph = graph if graph is not None else self.create_default_graph()

    def extract(self, history: pd.DataFrame):
        return self.graph.compute(self.names, {'history': history})

    @staticmethod
    def create_default_graph():
        commission_rate = 0.01
        discount = (1 - commission_rate) / (1 + commission_rate)

        graph = Graph()

        graph.register('history')

        graph.register('price', DataExtractor.Get('Close', 'Price'), ['history'])

        graph.register('past_price', DataExtractor.Merge([DataExtractor.Shift(i) for i in range(1, 30)]), ['price'])

        graph.register('log_price', DataExtractor.Log(), ['price'])
        graph.register('past_log_gain', DataExtractor.Merge([DataExtractor.Diff(i) for i in range(1, 30)]), ['log_price'])

        graph.register('mean_price', DataExtractor.Merge([DataExtractor.Mean(i) for i in [5, 10, 20, 30]]), ['price'])
        graph.register('std_price', DataExtractor.Merge([DataExtractor.Std(i) for i in [5, 10, 20, 30]]), ['price'])
        graph.register('skew_price', DataExtractor.Merge([DataExtractor.Skew(i) for i in [5, 10, 20, 30]]), ['price'])

        graph.register('next_log_gain', DataExtractor.Diff(1, future=True), ['log_price'])
        graph.register('next_direction', DataExtractor.Sign(), ['next_log_gain'])

        graph.register('expert_action', DataExtractor.Get('ExpertAction', 'Action'), ['history'])
        graph.register('expert_action_when_empty', DataExtractor.Fill(Const.ASK), ['expert_action'])
        graph.register('expert_action_when_full', DataExtractor.Fill(Const.BID), ['expert_action'])

        graph.register('days_to_cross_ub_of_bid', DataExtractor.DaysToCross(1, 1 / discount), ['price'])
        graph.register('days_to_cross_lb_of_bid', DataExtractor.DaysToCross(-1, 1), ['price'])

        graph.register('days_to_cross_ub_of_ask', DataExtractor.DaysToCross(1, 1), ['price'])
        graph.register('days_to_cross_lb_of_ask', DataExtractor.DaysToCross(-1, discount), ['price'])

        graph.register('indicator_of_bid', DataExtractor.Indicator(1, 1 / discount), ['price'])
        graph.register('indicator_of_ask', DataExtractor.Indicator(discount, 1), ['price'])

        graph.register('perfect_action', DataExtractor.PerfectAction(commission_rate), ['price'])
        graph.register('perfect_action_when_empty', DataExtractor.Get('EmptyAction', 'Action'), ['perfect_action'])
        graph.register('perfect_action_when_full', DataExtractor.Get('FullAction', 'Action'), ['perfect_action'])

        return graph

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

    class Log(Graph.Function):

        def compute(self, inputs):
            return np.log(inputs[0])

    class Sign(Graph.Function):

        def compute(self, inputs):
            return np.sign(inputs[0])

    class Shift(Graph.Function):

        def __init__(self, offset):
            self.offset = offset

        def compute(self, inputs):
            return inputs[0].shift(self.offset).fillna(0)

    class Diff(Graph.Function):

        def __init__(self, distance, future=False):
            self.distance = distance
            self.future = future

        def compute(self, inputs):
            input_df = inputs[0]
            if self.future:
                return (-(input_df - input_df.shift(-self.distance))).fillna(0)
            else:
                return (input_df - input_df.shift(self.distance)).fillna(0)

    class Agg(Graph.Function):

        def __init__(self, window_size, future=False):
            self.window_size = window_size
            self.future = future

        def compute(self, inputs):
            input_df = inputs[0]
            if self.future:
                input_df = input_df.loc[::-1]
            output_df = self.aggregate(input_df.rolling(self.window_size, min_periods=1))
            if self.future:
                output_df = output_df.loc[::-1]
            return output_df

        @abstractmethod
        def aggregate(self, data):
            pass

    class Mean(Agg):

        def aggregate(self, data):
            return data.mean()

    class Std(Agg):

        def aggregate(self, data):
            return data.std().fillna(0)

    class Skew(Agg):

        def aggregate(self, data):
            return data.skew().fillna(0)

    class Merge(Graph.Function):

        def __init__(self, functions):
            self.functions = functions

        def compute(self, inputs):
            return pd.concat([function.compute(inputs) for function in self.functions], axis=1)

    class DaysToCross(Graph.Function):

        def __init__(self, direction, multiplier):
            self.direction = direction
            self.multiplier = multiplier

        def compute(self, inputs):
            prices = inputs[0].iloc[:, 0]
            result = []
            for i in range(len(prices)):
                days_to_cross = float('inf')
                for j in range(i + 1, len(prices)):
                    if self.direction * (prices[j] - prices[i] * self.multiplier) > 0:
                        days_to_cross = j - i
                        break
                result.append(days_to_cross)
            df = pd.DataFrame({'DaysToCross': result})
            return df

    class Indicator(Graph.Function):

        def __init__(self, lb, ub):
            self.lb = lb
            self.ub = ub

        def compute(self, inputs):
            prices = inputs[0].iloc[:, 0]
            result = []
            for i in range(len(prices)):
                f = Const.DOWN
                for j in range(i + 1, len(prices)):
                    if prices[j] > prices[i] * self.ub:
                        f = Const.UP
                        break
                    if prices[j] < prices[i] * self.lb:
                        f = f = Const.DOWN
                        break
                result.append(f)
            df = pd.DataFrame({'Indicator': result})
            return df

    class PerfectAction(Graph.Function):

        def __init__(self, commission_rate):
            self.action_generator = ActionGenerator(commission_rate)

        def compute(self, inputs):
            cum_gains, actions, advantages = self.action_generator.generate(inputs[0].iloc[:, 0])
            df = pd.DataFrame({'EmptyCumGain': cum_gains[Const.EMPTY], 'FullCumGain': cum_gains[Const.FULL],
                               'EmptyAction': actions[Const.EMPTY], 'FullAction': actions[Const.FULL],
                               'EmptyAdvantage': advantages[Const.EMPTY], 'FullAdvantage': advantages[Const.FULL],
            })
            return df
