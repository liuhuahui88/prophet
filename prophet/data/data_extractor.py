from abc import abstractmethod

import pandas as pd
import numpy as np

from prophet.utils.action_generator import ActionGenerator
from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class DataExtractor:

    def __init__(self, commission_rate):
        self.graph = self.__create_graph(commission_rate)

    def extract(self, history: pd.DataFrame, names):
        return self.graph.compute(names, {'history': history})

    @staticmethod
    def __create_graph(commission_rate):
        discount = (1 - commission_rate) / (1 + commission_rate)

        graph = Graph()

        graph.register('history')

        graph.register('price', DataExtractor.Get('Close', 'Price'), ['history'])

        graph.register('prices', DataExtractor.Merge([DataExtractor.Shift(i) for i in range(0, 30)]), ['price'])

        graph.register('log_price', DataExtractor.Log(), ['price'])
        graph.register('log_gain', DataExtractor.Diff(1), ['log_price'])

        graph.register('short_term_stat', DataExtractor.Mean(10), ['log_price'])
        graph.register('long_term_stat', DataExtractor.Mean(25), ['log_price'])
        graph.register('flip', DataExtractor.Flip(), ['short_term_stat', 'long_term_stat'])

        graph.register('next_log_gain', DataExtractor.Shift(-1), ['log_gain'])
        graph.register('next_direction', DataExtractor.Sign(), ['next_log_gain'])

        graph.register('expert_action', DataExtractor.Get('ExpertAction', 'Action'), ['history'])
        graph.register('expert_action_when_empty', DataExtractor.Fill(Const.ASK), ['expert_action'])
        graph.register('expert_action_when_full', DataExtractor.Fill(Const.BID), ['expert_action'])

        graph.register('days_to_cross_ub_of_bid', DataExtractor.DaysToCross(Const.UP, 1 / discount), ['price'])
        graph.register('days_to_cross_lb_of_bid', DataExtractor.DaysToCross(Const.DOWN, 1), ['price'])

        graph.register('days_to_cross_ub_of_ask', DataExtractor.DaysToCross(Const.UP, 1), ['price'])
        graph.register('days_to_cross_lb_of_ask', DataExtractor.DaysToCross(Const.DOWN, discount), ['price'])

        graph.register('perfect_indicator_of_bid', DataExtractor.PerfectIndicator(1, 1 / discount), ['price'])
        graph.register('perfect_indicator_of_ask', DataExtractor.PerfectIndicator(discount, 1), ['price'])

        graph.register('perfect_action', DataExtractor.PerfectAction(commission_rate), ['price'])
        graph.register('perfect_action_when_empty', DataExtractor.Get('EmptyAction', 'Action'), ['perfect_action'])
        graph.register('perfect_action_when_full', DataExtractor.Get('FullAction', 'Action'), ['perfect_action'])

        graph.register('perfect_advantage_when_empty', DataExtractor.PerfectAdvantage('Empty'), ['perfect_action'])
        graph.register('perfect_advantage_when_full', DataExtractor.PerfectAdvantage('Full'), ['perfect_action'])

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

        def __init__(self, window_size, mode=Const.PAST):
            self.window_size = window_size
            self.mode = mode

        def compute(self, inputs):
            input_df = inputs[0]
            if self.mode == Const.FUTURE:
                input_df = input_df.iloc[::-1]
            rolling = input_df.rolling(self.window_size, min_periods=1, center=(self.mode == Const.CENTER))
            output_df = self.aggregate(rolling)
            if self.mode == Const.FUTURE:
                output_df = output_df.iloc[::-1]
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

    class Kurt(Agg):

        def aggregate(self, data):
            return data.kurt().fillna(0)

    class Merge(Graph.Function):

        def __init__(self, functions):
            self.functions = functions

        def compute(self, inputs):
            return pd.concat([function.compute(inputs) for function in self.functions], axis=1)

    class Flip(Graph.Function):

        def compute(self, inputs):
            line1 = inputs[0].iloc[:, 0].tolist()
            line2 = inputs[1].iloc[:, 0].tolist()

            previous_state = 0
            result = []
            for i in range(len(line1)):
                current_state = np.sign(line1[i] - line2[i])
                if current_state == 0:
                    result.append(0)
                elif previous_state == 0:
                    result.append(current_state)
                elif current_state * previous_state == 1:
                    result.append(0)
                else:
                    result.append(current_state)
                previous_state = current_state

            df = pd.DataFrame({'Flip': result})
            return df

    class DaysToCross(Graph.Function):

        def __init__(self, direction, multiplier):
            self.direction = direction
            self.multiplier = multiplier

        def compute(self, inputs):
            prices = inputs[0].iloc[:, 0].tolist()
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

    class PerfectIndicator(Graph.Function):

        def __init__(self, lb, ub):
            self.lb = lb
            self.ub = ub

        def compute(self, inputs):
            prices = inputs[0].iloc[:, 0].tolist()
            result = []
            for i in range(len(prices)):
                f = Const.DOWN
                for j in range(i + 1, len(prices)):
                    if prices[j] > prices[i] * self.ub:
                        f = Const.UP
                        break
                    if prices[j] < prices[i] * self.lb:
                        f = Const.DOWN
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
                               'EmptyAdvantage': advantages[Const.EMPTY], 'FullAdvantage': advantages[Const.FULL]
                               })
            return df

    class PerfectAdvantage(Graph.Function):

        def __init__(self, position_name):
            self.cum_gain_name = '{}CumGain'.format(position_name)
            self.action_name = '{}Action'.format(position_name)
            self.advantage_name = '{}Advantage'.format(position_name)

        def compute(self, inputs):
            output_df = pd.DataFrame()
            output_df['Advantage'] = inputs[0][self.action_name]
            output_df['Advantage'] = output_df['Advantage'].apply(lambda x: -1 if x == Const.ASK else 1)
            output_df['Advantage'] = output_df['Advantage'] * inputs[0][self.advantage_name]
            return output_df
