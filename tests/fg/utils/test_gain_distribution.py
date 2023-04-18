from unittest import TestCase

from prophet.fg.utils.gain_distribution import GainDistribution


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

    def test_rv(self):
        dist = GainDistribution(3, 0.5, 0)

        dist.update(1, 10)
        self.assertAlmostEqual(dist.rv(), 0.5, 4)

        dist.update(1, 10)
        self.assertAlmostEqual(dist.rv(), 1.0, 4)

        dist.update(1, 5)
        self.assertAlmostEqual(dist.rv(), 1.0, 4)

        dist.update(-4, 15)
        self.assertAlmostEqual(dist.rv(), 0.3333, 4)