import pandas as pd

from prophet.fg.function.aggregator import Mean, Skew, Std, Kurt, Rank
from prophet.fg.function.commons import Get, Merge
from prophet.fg.function.math import Clip, Positive, Log, Indicator
from prophet.fg.function.oracle import Oracle
from prophet.fg.function.series import Shift, Diff, Flip, Keep
from prophet.utils.graph import Graph


class DataExtractor:

    def __init__(self, commission_rate):
        self.graph = self.create_graph(commission_rate)

    def extract(self, history: pd.DataFrame, names):
        return self.graph.compute(names, {'history': history})

    def extract_and_concat(self, histories, names):
        histories = [history for history in histories if len(history) != 0]
        datas = [self.extract(history, names) for history in histories]
        return {name: pd.concat([data[name] for data in datas]) for name in names}

    @staticmethod
    def create_graph(commission_rate):
        graph = Graph()

        graph.register('history')

        graph.register('price', Get('Close', 'Price'), ['history'])
        graph.register('volume', Get('Volume'), ['history'])

        graph.register('prices', Merge([Shift(i) for i in range(0, 30)]), ['price'])

        graph.register('log_price', Log(), ['price'])
        graph.register('log_volume', Log(), ['volume'])

        graph.register('log_volume_rank', Rank(20), ['log_volume'])
        graph.register('log_volume_rank_mean', Mean(5), ['log_volume_rank'])
        graph.register('log_volume_rank_std', Std(5), ['log_volume_rank'])
        graph.register('log_volume_rank_skew', Skew(5), ['log_volume_rank'])
        graph.register('log_volume_rank_kurt', Kurt(5), ['log_volume_rank'])

        graph.register('log_gain', Diff(1), ['log_price'])
        graph.register('log_gain_mean', Mean(20), ['log_gain'])
        graph.register('log_gain_std', Std(20), ['log_gain'])
        graph.register('log_gain_skew', Skew(20), ['log_gain'])
        graph.register('log_gain_kurt', Kurt(20), ['log_gain'])

        graph.register('log_price_rank', Rank(20), ['log_price'])
        graph.register('log_price_rank_mean', Mean(5), ['log_price_rank'])
        graph.register('log_price_rank_std', Std(5), ['log_price_rank'])
        graph.register('log_price_rank_skew', Skew(5), ['log_price_rank'])
        graph.register('log_price_rank_kurt', Kurt(5), ['log_price_rank'])

        graph.register('log_price_rank_diff', Diff(1), ['log_price_rank'])
        graph.register('log_price_rank_diff_mean', Mean(5), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_std', Std(5), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_skew', Skew(5), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_kurt', Kurt(5), ['log_price_rank_diff'])

        graph.register('log_price_rank_diff_diff', Diff(1), ['log_price_rank_diff'])
        graph.register('log_price_rank_diff_diff_mean', Mean(5), ['log_price_rank_diff_diff'])
        graph.register('log_price_rank_diff_diff_std', Std(5), ['log_price_rank_diff_diff'])
        graph.register('log_price_rank_diff_diff_skew', Skew(5), ['log_price_rank_diff_diff'])
        graph.register('log_price_rank_diff_diff_kurt', Kurt(5), ['log_price_rank_diff_diff'])

        graph.register('keep_lose', Keep(lambda n: n < 0), ['log_gain'])
        graph.register('keep_gain', Keep(lambda n: n > 0), ['log_gain'])

        graph.register('short_term_stat', Mean(10), ['log_price'])
        graph.register('long_term_stat', Mean(25), ['log_price'])
        graph.register('flip', Flip(), ['short_term_stat', 'long_term_stat'])

        graph.register('next_log_gain', Shift(-1), ['log_gain'])
        graph.register('next_clipped_log_gain', Clip(-0.08, 0.08), ['next_log_gain'])

        graph.register('next_inc', Positive(), ['next_log_gain'])
        graph.register('next_significant_inc', Indicator(lambda n: n > 0.01), ['next_log_gain'])

        graph.register('oracle', Oracle(commission_rate), ['price'])
        graph.register('oracle_empty_advantage', Get('EmptyAdvantage', 'Advantage'), ['oracle'])
        graph.register('oracle_full_advantage', Get('FullAdvantage', 'Advantage'), ['oracle'])

        return graph
