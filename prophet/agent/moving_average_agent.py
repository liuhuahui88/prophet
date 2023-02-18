import collections
import numpy as np

from prophet.agent.abstract_agent import *


class MovingAverageAgent(Agent):

    def __init__(self, capital_id, fast_window_size, slow_window_size):
        self.capital_id = capital_id

        self.fast_window_size = fast_window_size
        self.slow_window_size = slow_window_size

        self.fast_queue = collections.deque([], maxlen=fast_window_size)
        self.slow_queue = collections.deque([], maxlen=slow_window_size)

    def handle(self, ctx: Agent.Context):
        close = ctx.get_prices()[self.capital_id]
        self.fast_queue.append(close)
        self.slow_queue.append(close)
        if len(self.fast_queue) != self.fast_window_size:
            return
        if len(self.slow_queue) != self.slow_window_size:
            return

        if np.mean(self.slow_queue) > np.mean(self.fast_queue):
            volume = ctx.get_account().get_capital(self.capital_id)
            ctx.trade(self.capital_id, -volume)
        else:
            cash = ctx.get_account().get_cash()
            price = ctx.get_prices()[self.capital_id]
            volume = int(cash / price)
            ctx.trade(self.capital_id, volume)
