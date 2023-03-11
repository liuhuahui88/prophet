from prophet.agent.abstract_agent import Agent
from prophet.utils.action_generator import ActionGenerator
from prophet.utils.constant import Const


class PerfectAgent(Agent):

    def __init__(self, symbol, storage, commission_rate):
        self.symbol = symbol
        self.history = storage.load_history(symbol)

        self.indexes = {self.history.iloc[i].Date: i for i in range(len(self.history))}

        action_generator = ActionGenerator(commission_rate)
        prices = [self.history['Close'].iloc[i] for i in range(len(self.history))]
        cum_gains, actions, advantages = action_generator.generate(prices)

        self.history['EmptyAction'] = actions[Const.EMPTY]
        self.history['FullAction'] = actions[Const.FULL]

    def handle(self, ctx: Agent.Context):
        index = self.indexes[ctx.get_date()]

        volume = ctx.get_account().get_volume(self.symbol)
        action_column_name = 'EmptyAction' if volume == 0 else 'FullAction'

        action = self.history[action_column_name].iloc[index]

        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)
