from unittest import TestCase

import numpy as np
import pandas as pd

from prophet.fg.function.aggregator import Aggregator
from prophet.fg.function.commons import Get, Fill, Merge
from prophet.fg.function.math import Log, Sign
from prophet.fg.function.series import Shift, Diff, Keep, Flip, Ordered
from prophet.utils.constant import Const
from prophet.utils.graph import Graph


class TestFeatureGeneration(TestCase):

    def test_get(self):
        x = pd.DataFrame({'n': [1, 2, 3], 'm': [4, 5, 6]})

        get = Get('n')
        y1 = pd.DataFrame({'n': [1, 2, 3]})
        self.assert_df_equal(get.compute([x]), y1)

        get_as = Get('m', 'x')
        y2 = pd.DataFrame({'x': [4, 5, 6]})
        self.assert_df_equal(get_as.compute([x]), y2)

    def test_fill(self):
        self.check(Fill(2), [1, np.nan, 3], [1, 2, 3])

    def test_log(self):
        self.check(Log(), [1, 1, 1], [0, 0, 0])

    def test_sign(self):
        self.check(Sign(), [10, 0, -10], [1, 0, -1])

    def test_shift(self):
        self.check(Shift(0), [1, 2, 3], [1, 2, 3])
        self.check(Shift(2), [1, 2, 3], [0, 0, 1])
        self.check(Shift(-2), [1, 2, 3], [3, 0, 0])

    def test_diff(self):
        self.check(Diff(0), [1, 2, 3], [0, 0, 0])
        self.check(Diff(0, future=True), [1, 2, 3], [0, 0, 0])
        self.check(Diff(1), [1, 2, 3], [0, 1, 1])
        self.check(Diff(1, future=True), [1, 2, 3], [1, 1, 0])

    def test_agg(self):

        class Sum(Aggregator):

            def aggregate(self, data):
                return data.sum()

        x = [1, 2, 3, 4, 5]

        self.check(Sum(1), x, x)
        self.check(Sum(1, mode=Const.FUTURE), x, x)

        self.check(Sum(2), x, [1, 3, 5, 7, 9])
        self.check(Sum(2, mode=Const.FUTURE), x, [3, 5, 7, 9, 5])

        self.check(Sum(3, mode=Const.CENTER), x, [3, 6, 9, 12, 9])

        self.check(Sum(alpha=0.5), x, [1, 2.5, 4.25, 6.125, 8.0625])

    def test_merge(self):
        x = pd.DataFrame({'n': [1, 2, 3], 'm': [4, 5, 6]})

        get_n = Get('n')
        get_m = Get('m')
        merge = Merge([get_n, get_m])

        self.assert_df_equal(merge.compute([x]), x)

    def test_keep(self):
        x = pd.DataFrame({'n': [-1, 1, 1, 1, -1, -1, 1]})

        actual = Keep(lambda z: z < 0).compute([x])
        expected = pd.DataFrame({'Keep': [1, 0, 0, 0, 1, 2, 0]})
        self.assert_df_equal(actual, expected)

    def test_ordered(self):
        x = pd.DataFrame({'n': [-1, 1, 1, 1, -1, -1, 1]})

        actual = Ordered(2).compute([x])
        expected = pd.DataFrame({'Ordered': [0.5, 1, 0.5, 0.5, 0, 0.5, 1]})
        self.assert_df_equal(actual, expected)

    def test_flip(self):
        x1 = pd.DataFrame({'n': [0, 0, 1, 0, 0, 1, 1, 0]})
        x2 = pd.DataFrame({'n': [0, 1, 0, 1, 1, 0, 0, 0]})

        actual = Flip().compute([x1, x2])
        expected = pd.DataFrame({'Flip': [0, -1, 1, -1, 0, 1, 0, 0]})
        self.assert_df_equal(actual, expected)

    def check(self, f: Graph.Function, x, y):
        dx = pd.DataFrame({'n': x})
        dy = pd.DataFrame({'n': y})
        self.assert_df_equal(f.compute([dx]), dy)

    def assert_df_equal(self, dx, dy):
        self.assertTrue((dx == dy).all().all())
