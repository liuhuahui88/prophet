from unittest import TestCase

from prophet.utils.gain_distribution import GainDistribution


class TestGainDistribution(TestCase):

    def test_cdf(self):
        dist = GainDistribution(3, 0.5, 0)

        dist.update(1, 10)
        self.assertAlmostEqual(dist.cdf(), 1.0, 4)

        dist.update(1, 10)
        self.assertAlmostEqual(dist.cdf(), 1.0, 4)

        dist.update(-1, 5)
        self.assertAlmostEqual(dist.cdf(), 0.6, 4)

        dist.update(-4, 15)
        self.assertAlmostEqual(dist.cdf(), 0.75, 4)
