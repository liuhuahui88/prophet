import pandas as pd
import numpy as np
import tensorflow as tf

from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor
from prophet.utils.constant import Const


class SmartAgent(Agent):

    def __init__(self, symbol, commission_rate):
        self.symbol = symbol
        self.history = pd.DataFrame(columns=['Close'])

        ask_friction = np.log(1 - commission_rate)
        bid_friction = np.log(1 / (1 + commission_rate))
        self.delta = - (ask_friction + bid_friction)

        self.data_predictor = DataPredictor(self.create_model(self.delta), DataExtractor(commission_rate))

    def handle(self, ctx: Agent.Context):
        self.update(ctx)

        action = self.predict(ctx)
        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

    def update(self, ctx: Agent.Context):
        record = pd.DataFrame({'Close': ctx.get_prices()[self.symbol]}, index=[len(self.history)])
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

        return Const.BID if score > 0 else Const.ASK

    def observe(self, history: pd.DataFrame):
        self.data_predictor.train(history, 0.9, 1, 100, 100)

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
        x = tf.keras.layers.Dense(1, activation='linear', name='perfect_advantage_when_empty')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=x)

        def hard(tensor):
            return (tf.sign(tensor) + 1) / 2

        def hinge(tensor):
            width = delta / 2
            margin_pct = 0.1
            margin = width * margin_pct
            return tf.keras.activations.relu(tensor + margin)

        def soft(tensor):
            return tf.keras.activations.sigmoid(tensor)

        def advt_single(y_true, y_pred, indicator_fn):
            return tf.abs(y_true) * indicator_fn(-y_pred * tf.sign(y_true))

        def advt_pair(y_true, y_pred, indicator_fn):
            empty_advt = advt_single(y_true, y_pred, indicator_fn)
            full_advt = advt_single(y_true + delta, y_pred + delta, indicator_fn)
            return tf.reduce_mean(tf.maximum(empty_advt, full_advt))

        def hard_advt(y_true, y_pred):
            return advt_pair(y_true, y_pred, hard)

        def hinge_advt(y_true, y_pred):
            return advt_pair(y_true, y_pred, hinge)

        def soft_advt(y_true, y_pred):
            return advt_pair(y_true, y_pred, soft)

        model.compile(optimizer='adam',
                      loss={'perfect_advantage_when_empty': 'mse'},
                      metrics=[hard_advt, hinge_advt, soft_advt])

        return model
