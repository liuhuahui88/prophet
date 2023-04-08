from prophet.agent.abstract_agent import Agent


class EnsembleAgent(Agent):

    def __init__(self, agents):
        self.agents = agents

    def handle(self, ctx: Agent.Context):
        best_score = -float('inf')
        best_symbol = None
        for agent in self.agents:
            agent.update(ctx)
            score = agent.predict(ctx)
            if score > best_score:
                best_score = score
                best_symbol = agent.symbol

        volumes = ctx.get_volumes()
        for s in volumes:
            if s != best_symbol:
                ctx.ask(s)

        if best_score > 0:
            ctx.bid(best_symbol)
        else:
            ctx.ask(best_symbol)
