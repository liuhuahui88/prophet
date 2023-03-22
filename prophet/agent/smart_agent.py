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
        self.data_predictor.train(history, 0.9, 100, 100)

    @staticmethod
    def create_model(delta):
        input0 = tf.keras.layers.Input(name='price', shape=(1,))
        input1 = tf.keras.layers.Input(name='past_price', shape=(29,))
        input2 = tf.keras.layers.Input(name='past_log_gain', shape=(29,))
        input3 = tf.keras.layers.Input(name='mean_price', shape=(4,))
        input4 = tf.keras.layers.Input(name='std_price', shape=(4,))
        input5 = tf.keras.layers.Input(name='skew_price', shape=(4,))
        input6 = tf.keras.layers.Input(name='kurt_price', shape=(4,))
        input7 = tf.keras.layers.Input(name='flip', shape=(1,))
        inputs = [input0, input1, input2, input3, input4, input5, input6, input7]

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

        def advt_single(y_true, y_pred, indicator_fn):
            return tf.reduce_mean(tf.abs(y_true) * indicator_fn(-y_pred * tf.sign(y_true)))

        def advt_pair(y_true, y_pred, indicator_fn):
            empty_advt = advt_single(y_true, y_pred, indicator_fn)
            full_advt = advt_single(y_true + delta, y_pred + delta, indicator_fn)
            return tf.maximum(empty_advt, full_advt)

        def hard_advt(y_true, y_pred):
            return advt_pair(y_true, y_pred, lambda tensor: (tf.sign(tensor) + 1) / 2)

        def hinge_advt(y_true, y_pred):
            return advt_pair(y_true, y_pred, lambda tensor: tf.keras.activations.relu(tensor + 1))

        def soft_advt(y_true, y_pred):
            return advt_pair(y_true, y_pred, lambda tensor: tf.keras.activations.sigmoid(tensor))

        model.compile(optimizer='adam',
                      loss={'perfect_advantage_when_empty': 'mse'},
                      metrics=[hard_advt, hinge_advt, soft_advt])

        return model
