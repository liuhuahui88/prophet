import pandas as pd
import tensorflow as tf

from prophet.agent.abstract_agent import Agent
from prophet.data.data_predictor import DataPredictor
from prophet.utils.constant import Const


class SmartAgent(Agent):

    WINDOW_SIZE = 30

    def __init__(self, symbol):
        self.symbol = symbol
        self.history = pd.DataFrame(columns=['Close'])
        self.data_predictor = DataPredictor(self.create_model())

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
        history = self.history.tail(self.WINDOW_SIZE)

        result = self.data_predictor.predict(history)

        # select the score for the last sample in prediction result
        score = result[position].ravel()[-1]

        return Const.BID if score > 0 else Const.ASK

    def observe(self, history: pd.DataFrame):
        self.data_predictor.train(history, 0.9, 100, 100)

    @staticmethod
    def create_model():
        input0 = tf.keras.layers.Input(name='price', shape=(1,))
        input1 = tf.keras.layers.Input(name='past_price', shape=(29,))
        input2 = tf.keras.layers.Input(name='past_log_gain', shape=(29,))
        input3 = tf.keras.layers.Input(name='mean_price', shape=(4,))
        input4 = tf.keras.layers.Input(name='std_price', shape=(4,))
        input5 = tf.keras.layers.Input(name='skew_price', shape=(4,))
        inputs = [input0, input1, input2, input3, input4, input5]

        x = tf.keras.layers.Concatenate()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)

        y = tf.keras.layers.Dense(128, activation='relu')(x)
        y = tf.keras.layers.BatchNormalization(momentum=0)(y)
        y = tf.keras.layers.Dense(128, activation='relu')(y)
        y = tf.keras.layers.BatchNormalization(momentum=0)(y)
        y = tf.keras.layers.Dense(1, activation='linear', name='perfect_advantage_when_empty')(y)

        z = tf.keras.layers.Dense(128, activation='relu')(x)
        z = tf.keras.layers.BatchNormalization(momentum=0)(z)
        z = tf.keras.layers.Dense(128, activation='relu')(z)
        z = tf.keras.layers.BatchNormalization(momentum=0)(z)
        z = tf.keras.layers.Dense(1, activation='linear', name='perfect_advantage_when_full')(z)

        model = tf.keras.models.Model(inputs=inputs, outputs=[y, z])

        def bce(y_true, y_pred):
            action_true = (tf.sign(y_true) + 1) / 2
            return tf.keras.losses.binary_crossentropy(action_true, y_pred, from_logits=True)

        def soft_advt(y_true, y_pred):
            return tf.reduce_mean(tf.abs(y_true) * tf.sigmoid(-tf.sign(y_true) * y_pred))

        def hinge_advt(y_true, y_pred):
            return tf.reduce_mean(tf.abs(y_true) * tf.keras.activations.relu(-tf.sign(y_true) * y_pred + 1))

        def hard_advt(y_true, y_pred):
            return tf.reduce_mean(tf.abs(y_true) * (tf.sign(-y_true * y_pred) + 1) / 2)

        def avg_true(y_true, y_pred):
            return tf.reduce_mean((tf.sign(y_true) + 1) / 2, axis=-1)

        def avg_pred(y_true, y_pred):
            return tf.reduce_mean(y_pred, axis=-1)

        model.compile(optimizer='adam',
                      loss={
                          'perfect_advantage_when_empty': bce,
                          'perfect_advantage_when_full': bce},
                      metrics=[hard_advt, hinge_advt, soft_advt, avg_true, avg_pred])

        return model
