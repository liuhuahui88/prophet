import pandas as pd
import numpy as np
import tensorflow as tf

from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor
from prophet.utils.constant import Const
from prophet.utils.metric import Metric


class SmartAgent(Agent):

    def __init__(self, symbol, commission_rate):
        self.symbol = symbol
        self.history = pd.DataFrame(columns=['Close', 'Volume'])

        ask_friction = np.log(1 - commission_rate)
        bid_friction = np.log(1 / (1 + commission_rate))
        self.delta = - (ask_friction + bid_friction)

        self.data_predictor = DataPredictor(self.create_model(self.delta), DataExtractor(commission_rate))

    def handle(self, ctx: Agent.Context):
        self.update(ctx)
        score = self.predict(ctx)
        action = Const.BID if score > 0 else Const.ASK
        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

    def update(self, ctx: Agent.Context):
        data = {'Close': ctx.get_prices()[self.symbol], 'Volume': ctx.get_volumes()[self.symbol]}
        record = pd.DataFrame(data, index=[len(self.history)])
        self.history = pd.concat([self.history, record])

    def predict(self, ctx: Agent.Context):
        position = Const.EMPTY if ctx.get_account().get_volume(self.symbol) == 0 else Const.FULL

        # accelerate the prediction by processing the latest history only
        history = self.history.tail(Const.WINDOW_SIZE)

        result = self.data_predictor.predict(history)

        # select the score for the last sample in prediction result
        score = result.ravel()[-1]

        if position == Const.FULL:
            score += self.delta

        return score

    def observe(self, history: pd.DataFrame):
        self.data_predictor.train([history], 0.9, 1, 100, 100)

    @staticmethod
    def create_model(delta):
        prices = tf.keras.layers.Input(name='prices', shape=(30,))
        inputs = [prices]

        x = tf.keras.layers.Concatenate()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(1, activation='linear', name='oracle_empty_advantage')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=x)

        model.compile(optimizer='adam',
                      loss={'oracle_empty_advantage': 'mse'},
                      metrics=[Metric.create_hard_advt(delta),
                               Metric.create_hinge_advt(delta),
                               Metric.create_soft_advt(delta),
                               Metric.me, Metric.r2])

        return model
