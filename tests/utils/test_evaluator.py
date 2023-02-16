from unittest import TestCase

from prophet.utils.evaluator import Evaluator


class TestEvaluator(TestCase):

    def test_gain(self):
        evaluator = Evaluator()

        evaluator.feed(100)
        evaluator.feed(100)
        evaluator.feed(400)

        self.assertAlmostEqual(evaluator.get_gain_cum(), 4.0, 2)
        self.assertAlmostEqual(evaluator.get_gain_avg(), 2.0, 2)
        self.assertAlmostEqual(evaluator.get_gain_std(), 2.0, 2)

    def test_sharp_ratio(self):
        self.__feed_and_check_sharp_ratio([100, 100, 100], 1.000)

        self.__feed_and_check_sharp_ratio([100, 200, 400], 2.000)

        self.__feed_and_check_sharp_ratio([100, 100, 400], 1.500)
        self.__feed_and_check_sharp_ratio([100, 400, 400], 1.500)

        self.__feed_and_check_sharp_ratio([100, 50, 400], 1.250)
        self.__feed_and_check_sharp_ratio([100, 800, 400], 1.250)

    def __feed_and_check_sharp_ratio(self, values, expected_sharp_ratio):
        evaluator = Evaluator()
        for value in values:
            evaluator.feed(value)
        self.assertAlmostEqual(evaluator.get_sharp_ratio(), expected_sharp_ratio, 2)

    def test_worst_drawdown(self):
        evaluator = Evaluator()

        self.__feed_and_check_worst_drawdown(evaluator, 200, 1.000)
        self.__feed_and_check_worst_drawdown(evaluator, 300, 1.000)
        self.__feed_and_check_worst_drawdown(evaluator, 200, 0.666)
        self.__feed_and_check_worst_drawdown(evaluator, 250, 0.666)
        self.__feed_and_check_worst_drawdown(evaluator, 200, 0.666)
        self.__feed_and_check_worst_drawdown(evaluator, 150, 0.500)
        self.__feed_and_check_worst_drawdown(evaluator, 400, 0.500)
        self.__feed_and_check_worst_drawdown(evaluator, 100, 0.250)

    def __feed_and_check_worst_drawdown(self, evaluator, value, expected_worst_drawdown):
        evaluator.feed(value)
        self.assertAlmostEqual(evaluator.get_worst_drawdown(), expected_worst_drawdown, 2)
