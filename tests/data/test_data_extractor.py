from unittest import TestCase

import numpy as np
import pandas as pd

from prophet.data.data_extractor import DataExtractor
from prophet.utils.graph import Graph


class TestDataExtractor(TestCase):

    def test_get(self):
        x = pd.DataFrame({'n': [1, 2, 3], 'm': [4, 5, 6]})

        get = DataExtractor.Get('n')
        y1 = pd.DataFrame({'n': [1, 2, 3]})
        self.assert_df_equal(get.compute([x]), y1)

        get_as = DataExtractor.Get('m', 'x')
        y2 = pd.DataFrame({'x': [4, 5, 6]})
        self.assert_df_equal(get_as.compute([x]), y2)

    def test_fill(self):
        self.check(DataExtractor.Fill(2), [1, np.nan, 3], [1, 2, 3])

    def test_log(self):
        self.check(DataExtractor.Log(), [1, 1, 1], [0, 0, 0])

    def test_sign(self):
        self.check(DataExtractor.Sign(), [10, 0, -10], [1, 0, -1])

    def test_shift(self):
        self.check(DataExtractor.Shift(0), [1, 2, 3], [1, 2, 3])
        self.check(DataExtractor.Shift(2), [1, 2, 3], [0, 0, 1])
        self.check(DataExtractor.Shift(-2), [1, 2, 3], [3, 0, 0])

    def test_diff(self):
        self.check(DataExtractor.Diff(0), [1, 2, 3], [0, 0, 0])
        self.check(DataExtractor.Diff(0, future=True), [1, 2, 3], [0, 0, 0])
        self.check(DataExtractor.Diff(1), [1, 2, 3], [0, 1, 1])
        self.check(DataExtractor.Diff(1, future=True), [1, 2, 3], [1, 1, 0])

    def test_agg(self):

        class Sum(DataExtractor.Agg):

            def aggregate(self, data):
                return data.sum()

        x = [1, 2, 3, 4, 5]

        self.check(Sum(1), x, x)
        self.check(Sum(1, future=True), x, x)

        self.check(Sum(2), x, [1, 3, 5, 7, 9])
        self.check(Sum(2, future=True), x, [3, 5, 7, 9, 5])

    def test_merge(self):
        x = pd.DataFrame({'n': [1, 2, 3], 'm': [4, 5, 6]})

        get_n = DataExtractor.Get('n')
        get_m = DataExtractor.Get('m')
        merge = DataExtractor.Merge([get_n, get_m])

        self.assert_df_equal(merge.compute([x]),x)

    def check(self, f: Graph.Function, x, y):
        dx = pd.DataFrame()
        dx['n'] = x

        dy = pd.DataFrame()
        dy['n'] = y

        self.assert_df_equal(f.compute([dx]), dy)

    def assert_df_equal(self, dx, dy):
        self.assertTrue((dx == dy).all().all())
