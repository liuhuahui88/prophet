import numpy as np


class Evaluator:

    def __init__(self):
        self.values = []

    def feed(self, value):
        self.values.append(value)

    def get_gain_cum(self):
        return self.__calculate_gain(self.values[0], self.values[-1])

    def get_gain_avg(self):
        return np.exp(np.log(self.get_gain_cum()) / (len(self.values) - 1))

    def get_gain_std(self):
        log_gains = []
        for i in range(len(self.values) - 1):
            gain = self.__calculate_gain(self.values[i], self.values[i + 1])
            log_gains.append(np.log(gain))
        return np.exp(np.std(log_gains))

    def get_sharp_ratio(self):
        return 1 + (self.get_gain_avg() - 1) / self.get_gain_std()

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
        return 'gain_cum={:.4f}, gain_avg={:.4f}, gain_std={:.4f}, sharp_ratio={:.4f}, worst_drawdown={:.4f}'.format(
            self.get_gain_cum(),
            self.get_gain_avg(),
            self.get_gain_std(),
            self.get_sharp_ratio(),
            self.get_worst_drawdown())

    @staticmethod
    def __calculate_gain(value1, value2):
        return value2 / value1