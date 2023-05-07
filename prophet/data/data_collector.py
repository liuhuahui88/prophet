import pandas as pd

from prophet.agent.abstract_agent import Agent


class DataCollector:

    def __init__(self):
        self.prices = {}

    def feed(self, ctx: Agent.Context):
        for s in ctx.get_prices():
            if s not in self.prices:
                self.prices[s] = []
            self.prices[s].append(ctx.get_prices()[s])

    def get(self, n=0):
        if n == 0:
            return {s: pd.DataFrame(dict(Close=p)) for s, p in self.prices.items()}
        else:
            return {s: pd.DataFrame(dict(Close=p[-n:])) for s, p in self.prices.items()}
