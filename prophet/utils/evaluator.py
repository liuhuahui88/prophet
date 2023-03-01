import numpy as np
import scipy as sp


class Evaluator:

    def __init__(self, init_value=None):
        self.values = []
        if init_value is not None:
            self.values = [init_value]

    def feed(self, value):
        self.values.append(value)

    def get_gain_cum(self):
        return self.__calculate_gain(self.values[0], self.values[-1])

    def get_gain_avg(self):
        return np.exp(np.log(self.get_gain_cum()) / (len(self.values) - 1))

    def get_gain_std(self):
        log_gains = self.__get_log_gains()
        return np.exp(np.std(log_gains))

    def get_gain_skew(self):
        log_gains = self.__get_log_gains()
        return np.exp(sp.stats.skew(log_gains))

    def get_quantile(self, qs):
        log_gains = self.__get_log_gains()
        return np.exp([np.quantile(log_gains, q) for q in qs])

    def get_wtl(self):
        log_gains = self.__get_log_gains()

        num_win = sum(x > 0 for x in log_gains)
        num_tie = sum(x == 0 for x in log_gains)
        num_lose = sum(x < 0 for x in log_gains)

        num = num_win + num_tie + num_lose
        return [num_win / num, num_tie / num, num_lose / num]

    def get_sharp_ratio(self):
        return np.exp(np.log(self.get_gain_avg()) / np.log(self.get_gain_std()))

    def get_worst_drawdown(self):
        worst_drawdown = 1
        max_value = min_value = self.values[0]
        for value in self.values:
            if value > max_value:
                max_value = min_value = value
            elif value < min_value:
                min_value = value
                drawdown = self.__calculate_gain(max_value, min_value)
                worst_drawdown = min(drawdown, worst_drawdown)
        return worst_drawdown

    def __str__(self):
        return 'cum={:.4f}, avg={:.4f}, std={:.4f}, skew={:.4f},' \
               ' quantile={}, wtl={}, sharp={:.4f}, drawdown={:.4f}'.format(
            self.get_gain_cum(),
            self.get_gain_avg(),
            self.get_gain_std(),
            self.get_gain_skew(),
            str(['{:.4f}'.format(n) for n in self.get_quantile([0, 0.1, 0.5, 0.9, 1])]).replace("'", ""),
            str(['{:.4f}'.format(n) for n in self.get_wtl()]).replace("'", ""),
            self.get_sharp_ratio(),
            self.get_worst_drawdown())

    def __get_gains(self):
        gains = []
        for i in range(len(self.values) - 1):
            gain = self.__calculate_gain(self.values[i], self.values[i + 1])
            gains.append(gain)
        return gains

    def __get_log_gains(self):
        return np.log(self.__get_gains())

    @staticmethod
    def __calculate_gain(value1, value2):
        return value2 / value1
