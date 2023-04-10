from abc import abstractmethod

import numpy as np
import pandas as pd

from prophet.utils.action_generator import ActionGenerator
from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class DataExtractor:

    def __init__(self, commission_rate):
        self.graph = self.__create_graph(commission_rate)

    def extract(self, history: pd.DataFrame, names):
        return self.graph.compute(names, {'history': history})

    def extract_and_concat(self, histories, names):
        histories = [history for history in histories if len(history) != 0]
        datas = [self.extract(history, names) for history in histories]
        return {name: pd.concat([data[name] for data in datas]) for name in names}

    @staticmethod
    def __create_graph(commission_rate):
        graph = Graph()

        graph.register('history')

        graph.register('price', DataExtractor.Get('Close', 'Price'), ['history'])
        graph.register('volume', DataExtractor.Get('Volume'), ['history'])

        graph.register('prices', DataExtractor.Merge([DataExtractor.Shift(i) for i in range(0, 30)]), ['price'])

        graph.register('log_price', DataExtractor.Log(), ['price'])

        graph.register('log_gain', DataExtractor.Diff(1), ['log_price'])
        graph.register('log_gain_mean', DataExtractor.Mean(20), ['log_gain'])
        graph.register('log_gain_std', DataExtractor.Std(20), ['log_gain'])
        graph.register('log_gain_skew', DataExtractor.Skew(20), ['log_gain'])
        graph.register('log_gain_kurt', DataExtractor.Kurt(20), ['log_gain'])

        graph.register('log_price_rank', DataExtractor.Rank(20), ['log_price'])
        graph.register('log_price_rank_mean', DataExtractor.Mean(5), ['log_price_rank'])
        graph.register('log_price_rank_std', DataExtractor.Std(5), ['log_price_rank'])
        graph.register('log_price_rank_skew', DataExtractor.Skew(5), ['log_price_rank'])
        graph.register('log_price_rank_kurt', DataExtractor.Kurt(5), ['log_price_rank'])

        graph.register('log_price_rank_diff', DataExtractor.Diff(1), ['log_price_rank'])
        graph.register('log_price_rank_diff_mean', DataExtractor.Mean(5), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_std', DataExtractor.Std(5), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_skew', DataExtractor.Skew(5), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_kurt', DataExtractor.Kurt(5), ['log_price_rank_diff'])

        graph.register('log_price_rank_diff_diff', DataExtractor.Diff(1), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_diff_mean', DataExtractor.Mean(5), ['log_price_rank_diff_diff'])
        graph.register('log_price_rank_diff_diff_std', DataExtractor.Std(5), ['log_price_rank_diff_diff'])
        graph.register('log_price_rank_diff_diff_skew', DataExtractor.Skew(5), ['log_price_rank_diff_diff'])
        graph.register('log_price_rank_diff_diff_kurt', DataExtractor.Kurt(5), ['log_price_rank_diff_diff'])

        graph.register('keep_lose', DataExtractor.Keep(lambda n: n < 0), ['log_gain'])
        graph.register('keep_gain', DataExtractor.Keep(lambda n: n > 0), ['log_gain'])

        graph.register('short_term_stat', DataExtractor.Mean(10), ['log_price'])
        graph.register('long_term_stat', DataExtractor.Mean(25), ['log_price'])
        graph.register('flip', DataExtractor.Flip(), ['short_term_stat', 'long_term_stat'])

        graph.register('next_log_gain', DataExtractor.Shift(-1), ['log_gain'])
        graph.register('next_direction', DataExtractor.Sign(), ['next_log_gain'])

        graph.register('oracle', DataExtractor.Oracle(commission_rate), ['price'])
        graph.register('oracle_empty_advantage', DataExtractor.Get('EmptyAdvantage', 'Advantage'), ['oracle'])
        graph.register('oracle_full_advantage', DataExtractor.Get('FullAdvantage', 'Advantage'), ['oracle'])

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

    class Rank(Agg):

        def aggregate(self, data):
            return data.rank(pct=True)

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

    class Keep(Graph.Function):

        def __init__(self, cond):
            self.cond = cond

        def compute(self, inputs):
            line = inputs[0].iloc[:, 0].tolist()

            cnt = 0
            result = []
            for i in range(len(line)):
                if self.cond(line[i]):
                    cnt += 1
                else:
                    cnt = 0
                result.append(cnt)

            df = pd.DataFrame({'Keep': result})
            return df

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
