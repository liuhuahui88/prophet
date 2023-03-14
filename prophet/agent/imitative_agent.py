import datetime

import pandas as pd
import tensorflow as tf

from prophet.agent.abstract_agent import Agent
from prophet.utils.constant import Const
from prophet.data.data_extractor import DataExtractor


class ImitativeAgent(Agent):

    WINDOW_SIZE = 30

    def __init__(self, symbol):
        self.symbol = symbol
        self.history = pd.DataFrame(columns=['Close'])

        feature_names = ['price', 'past_price', 'past_log_gain', 'mean_price', 'std_price', 'skew_price']
        self.feature_extractor = DataExtractor(feature_names)

        label_names = ['expert_action_when_empty', 'expert_action_when_full']
        self.label_extractor = DataExtractor(label_names)

        self.model_for_train = None
        self.model_when_full = None
        self.model_when_empty = None

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
        # accelerate the prediction by generating the latest rather than the whole samples
        history = self.history.tail(self.WINDOW_SIZE)

        features = self.feature_extractor.extract(history)
        dataset = tf.data.Dataset.from_tensor_slices(features).batch(len(history))

        # select the right model based on current position
        position = Const.EMPTY if ctx.get_account().get_volume(self.symbol) == 0 else Const.FULL
        model = self.model_when_empty if position == Const.EMPTY else self.model_when_full

        # select the score for the last sample in prediction result
        score = model.predict(dataset, verbose=False).ravel()[-1]

        return Const.BID if score > 0.5 else Const.ASK

    def observe(self, history: pd.DataFrame, expert_actions):
        history = history.copy()
        history['ExpertAction'] = expert_actions

        features = self.feature_extractor.extract(history)
        labels = self.label_extractor.extract(history)
        train_dataset, test_dataset = self.create_datasets(features, labels, len(history), 0.9)

        self.model_for_train, self.model_when_empty, self.model_when_full = self.create_model()

        self.train_model(self.model_for_train, train_dataset, test_dataset, 100)

    @staticmethod
    def create_datasets(features, labels, num_samples, train_pct):
        num_train_samples = int(train_pct * num_samples)
        num_test_samples = num_samples - num_train_samples

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        train_dataset = dataset.take(num_train_samples).batch(num_train_samples)
        test_dataset = dataset.skip(num_train_samples).batch(num_test_samples)

        return train_dataset, test_dataset

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
        y = tf.keras.layers.Dense(1, activation='sigmoid', name='expert_action_when_empty')(y)

        z = tf.keras.layers.Dense(128, activation='relu')(x)
        z = tf.keras.layers.BatchNormalization(momentum=0)(z)
        z = tf.keras.layers.Dense(128, activation='relu')(z)
        z = tf.keras.layers.BatchNormalization(momentum=0)(z)
        z = tf.keras.layers.Dense(1, activation='sigmoid', name='expert_action_when_full')(z)

        model_for_train = tf.keras.models.Model(inputs=inputs, outputs=[y, z])
        model_when_empty = tf.keras.models.Model(inputs=inputs, outputs=[y])
        model_when_full = tf.keras.models.Model(inputs=inputs, outputs=[z])

        return model_for_train, model_when_empty, model_when_full

    @staticmethod
    def train_model(model: tf.keras.models.Model, train_dataset, test_dataset, epochs):
        model.compile(optimizer='adam',
                      loss={'expert_action_when_empty': 'bce', 'expert_action_when_full': 'bce'},
                      metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=False,
                  callbacks=[tensor_board_callback, early_stopping_callback])

        model.evaluate(train_dataset)
        model.evaluate(test_dataset)

        return model
