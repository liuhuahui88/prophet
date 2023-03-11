import numpy as np

from prophet.utils.constant import Const


class ActionGenerator:

    def __init__(self, commission_rate):
        self.commission_rate = commission_rate
        self.bid_friction = np.log(1 / (1 + commission_rate))
        self.ask_friction = np.log(1 - commission_rate)

    def generate(self, prices):
        gains = [np.log(prices[i]) - np.log(prices[i - 1]) for i in range(1, len(prices))]

        cum_gains = [[0], [0]]
        actions = [[Const.ASK], [Const.BID]]
        advantages = [[0], [0]]

        for gain in reversed(gains):
            empty_cum_gains = cum_gains[Const.EMPTY][-1]
            full_cum_gains = cum_gains[Const.FULL][-1]

            empty_ask_empty = empty_cum_gains
            empty_bid_full = full_cum_gains + self.bid_friction + gain
            cum_gains[Const.EMPTY].append(max(empty_ask_empty, empty_bid_full))
            actions[Const.EMPTY].append(Const.ASK if empty_ask_empty > empty_bid_full else Const.BID)
            advantages[Const.EMPTY].append(abs(empty_ask_empty - empty_bid_full))

            full_ask_empty = empty_cum_gains + self.ask_friction
            full_bid_full = full_cum_gains + gain
            cum_gains[Const.FULL].append(max(full_ask_empty, full_bid_full))
            actions[Const.FULL].append(Const.ASK if full_ask_empty > full_bid_full else Const.BID)
            advantages[Const.FULL].append(abs(full_ask_empty - full_bid_full))

        for matrix in [cum_gains, actions, advantages]:
            matrix[Const.EMPTY].reverse()
            matrix[Const.FULL].reverse()

        return cum_gains, actions, advantages
