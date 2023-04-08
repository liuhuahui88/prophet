from unittest import TestCase

import numpy as np

from prophet.utils.action_generator import ActionGenerator


class TestActionGenerator(TestCase):

    def test_generate_without_commission(self):
        generator = ActionGenerator(0)

        prices = [1, 2, 4, 4.2, 4, 8, 16]
        actions, advantages, cum_gains = generator.generate(prices)

        self.check_matrix_equal([[1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 0, 1, 1, 1]], actions, 0)
        self.check_matrix_equal([[2, 2, 1.05, 0.95, 2, 2, 1], [2, 2, 1.05, 0.95, 2, 2, 1]], np.exp(advantages), 2)
        self.check_matrix_equal([[16.8, 8.4, 4.2, 4, 4, 2, 1], [16.8, 8.4, 4.2, 4, 4, 2, 1]], np.exp(cum_gains), 2)

    def test_generate_with_commission(self):
        generator = ActionGenerator(0.1)

        prices = [1, 2, 4, 4.2, 4, 8, 16]
        actions, advantages, cum_gains = generator.generate(prices)

        self.check_matrix_equal([[1, 1, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]], actions, 0)
        self.check_matrix_equal([[2, 2, 1, 0.95, 2, 1.82, 1], [2.44, 2.44, 1.22, 1.16, 2.44, 2.22, 1]], np.exp(advantages), 2)
        self.check_matrix_equal([[14.55, 7.27, 3.64, 3.64, 3.64, 1.82, 1], [16, 8, 4, 3.81, 4, 2, 1]], np.exp(cum_gains), 2)

    def check_matrix_equal(self, matrix1, matrix2, places):
        self.assertEqual(len(matrix1), len(matrix2))
        for i in range(len(matrix1)):
            self.check_list_equal(matrix1[i], matrix2[i], places)

    def check_list_equal(self, list1, list2, places):
        self.assertEqual(len(list1), len(list2))
        for i in range(len(list1)):
            self.assertAlmostEqual(list1[i], list2[i], places)
