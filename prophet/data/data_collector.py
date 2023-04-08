import pandas as pd

from prophet.agent.abstract_agent import Agent


class DataCollector:

    def __init__(self, symbol):
        self.symbol = symbol
        self.history = pd.DataFrame(columns=['Close', 'Volume'])

    def feed(self, ctx: Agent.Context):
        data = dict()
        data['Close'] = ctx.get_prices()[self.symbol]
        data['Volume'] = ctx.get_volumes()[self.symbol]
        record = pd.DataFrame(data, index=[len(self.history)])
        self.history = pd.concat([self.history, record])

    def get(self):
        return self.history
