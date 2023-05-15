import scipy as sp
import numpy as np

from prophet.agent.abstract_agent import Agent


class SmartAgent(Agent):

    def __init__(self, caches, delta, global_threshold, local_threshold, top_k, weighted):
        self.caches = caches
        self.delta = delta
        self.global_threshold = global_threshold
        self.local_threshold = local_threshold
        self.top_k = top_k
        self.weighted = weighted

    def handle(self, ctx: Agent.Context):
        scores = self.collect_score(ctx)

        portfolio = self.construct_portfolio(scores)

        for symbol in ctx.get_prices().keys():
            ctx.ask(symbol)

        cash = ctx.get_account().get_cash()
        for symbol, weight in portfolio.items():
            ctx.bid(symbol, int(cash * weight))

    def collect_score(self, ctx: Agent.Context):
        date = ctx.get_date()
        account = ctx.get_account()

        scores = {}
        for symbol, cache in self.caches.items():
            if date not in cache:
                continue

            score = cache[date]
            if account.get_volume(symbol) > 0:
                score += self.delta

            scores[symbol] = score

        return scores

    def construct_portfolio(self, score_dict):
        if np.mean(list(score_dict.values())) < self.global_threshold:
            return dict()

        score_dict = {k: v for k, v in score_dict.items() if v >= self.local_threshold}
        if len(score_dict) == 0:
            return dict()

        symbols = list(score_dict.keys())
        scores = list(score_dict.values())

        indexes = sorted(range(len(symbols)), key=lambda i: scores[i], reverse=True)

        n = min(self.top_k, len(symbols))
        best_symbols = [symbols[i] for i in indexes[:n]]
        best_scores = [scores[i] for i in indexes[:n]]
        best_weights = [1 / n] * n if not self.weighted else sp.special.softmax(best_scores)

        return dict(zip(best_symbols, best_weights))
