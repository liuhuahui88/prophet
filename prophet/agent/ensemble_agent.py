from prophet.agent.abstract_agent import Agent


class EnsembleAgent(Agent):

    def __init__(self, agents):
        self.agents = agents

    def handle(self, ctx: Agent.Context):
        best_symbol = None
        best_score = -float('inf')

        for agent in self.agents:
            score = agent.predict(ctx)
            if score is None:
                continue

            if score > best_score:
                best_symbol = agent.symbol
                best_score = score

        if best_symbol is None:
            return

        volumes = ctx.get_volumes()
        for s in volumes:
            if s != best_symbol:
                ctx.ask(s)

        if best_score > 0:
            ctx.bid(best_symbol)
        else:
            ctx.ask(best_symbol)
