import pandas as pd

from prophet.fg.function.aggregator import Mean, Skew, Std, Kurt, Rank, Min, Max, Spread
from prophet.fg.function.commons import Get, Merge
from prophet.fg.function.math import Clip, Log, Indicator
from prophet.fg.function.oracle import Oracle
from prophet.fg.function.series import Shift, Diff, Ordered, RRank, Pearson
from prophet.utils.graph import Graph


class DataExtractor:

    def __init__(self, commission_rate):
        self.graph = self.create_graph(commission_rate)

    def extract(self, history: pd.DataFrame, names):
        return self.graph.compute(names, {'history': history})

    def extract_and_concat(self, histories, names):
        histories = [history for history in histories if len(history) != 0]
        datas = [self.extract(history, names) for history in histories]
        return {name: pd.concat([data[name] for data in datas]).reset_index(drop=True) for name in names}

    @staticmethod
    def create_graph(commission_rate):
        graph = Graph()

        DataExtractor.register_raw_features(graph)

        window = 10

        graph.register('log_price', Log(), ['price'])
        DataExtractor.register_statistics(graph, 'log_price', window)

        graph.register('log_price_diff', Diff(1), ['log_price'])
        DataExtractor.register_statistics(graph, 'log_price_diff', window)

        graph.register('log_price_diff_diff', Diff(1), ['log_price_diff'])
        DataExtractor.register_statistics(graph, 'log_price_diff_diff', window)

        graph.register('log_price_diff_diff_diff', Diff(1), ['log_price_diff_diff'])
        DataExtractor.register_statistics(graph, 'log_price_diff_diff_diff', window)

        DataExtractor.register_example_features(graph)

        DataExtractor.register_labels(graph, commission_rate)

        return graph

    @staticmethod
    def register_raw_features(graph):
        graph.register('history')

        graph.register('date', Get('Date'), ['history'])

        graph.register('open', Get('Open'), ['history'])
        graph.register('high', Get('High'), ['history'])
        graph.register('low', Get('Low'), ['history'])
        graph.register('close', Get('Close'), ['history'])

        graph.register('price', Get('Close', 'Price'), ['history'])
        graph.register('volume', Get('Volume'), ['history'])

    @staticmethod
    def register_example_features(graph):
        graph.register('prices', Merge([Shift(i) for i in range(0, 30)]), ['price'])

    @staticmethod
    def register_labels(graph, commission_rate):
        graph.register('next_log_price_diff', Diff(1, future=True), ['log_price'])
        graph.register('next_clipped_log_price_diff', Clip(-0.08, 0.08), ['next_log_price_diff'])

        graph.register('next_inc', Indicator(lambda n: n > 0), ['next_log_price_diff'])
        graph.register('next_significant_inc', Indicator(lambda n: n > 0.01), ['next_log_price_diff'])

        graph.register('oracle', Oracle(commission_rate), ['price'])
        graph.register('oracle_empty_advantage', Get('EmptyAdvantage', 'Advantage'), ['oracle'])
        graph.register('oracle_full_advantage', Get('FullAdvantage', 'Advantage'), ['oracle'])

    @staticmethod
    def register_statistics(graph, dependent_name, window):
        graph.register(dependent_name + '_mean', Mean(window), [dependent_name])
        graph.register(dependent_name + '_std', Std(window), [dependent_name])
        graph.register(dependent_name + '_skew', Skew(window), [dependent_name])
        graph.register(dependent_name + '_kurt', Kurt(window), [dependent_name])
        graph.register(dependent_name + '_min', Min(window), [dependent_name])
        graph.register(dependent_name + '_max', Max(window), [dependent_name])
        graph.register(dependent_name + '_spread', Spread(window), [dependent_name])
        graph.register(dependent_name + '_rank', Rank(window), [dependent_name])
        graph.register(dependent_name + '_rrank', RRank(window), [dependent_name])
        graph.register(dependent_name + '_ordered', Ordered(window), [dependent_name])

    @staticmethod
    def register_target_correlation(graph, feature, target, window):
        correlation = '{}_{}_corr'.format(feature, target)
        graph.register(correlation, Pearson(window), [feature, target])
        graph.register(correlation + '_shift', Shift(1), [correlation])
