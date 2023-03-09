import collections
import datetime

import pandas as pd
import tensorflow as tf

from prophet.agent.abstract_agent import Agent
from prophet.utils.constant import Const
from prophet.utils.feature_manager import FeatureManager
from prophet.utils.label_manager import LabelManager


class ImitativeAgent(Agent):

    def __init__(self, symbol, window_size):
        self.symbol = symbol
        self.window_size = window_size
        self.price_queue = collections.deque([], maxlen=window_size)

        self.feature_manager = FeatureManager(['price', 'gain', 'mean', 'std', 'skew'])
        self.label_manager = LabelManager(['action_when_empty', 'action_when_full'])

        self.model_for_train = None
        self.model_when_full = None
        self.model_when_empty = None

    def handle(self, ctx: Agent.Context):
        self.price_queue.append(ctx.get_prices()[self.symbol])

        if len(self.price_queue) != self.window_size:
            return

        action = self.predict(ctx)
        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

    def predict(self, ctx: Agent.Context):
        raw_features = self.extract_raw_features(self.price_queue, self.window_size)
        features = self.feature_manager.get(raw_features)
        dataset = tf.data.Dataset.from_tensor_slices(features).batch(1)

        position = Const.EMPTY if ctx.get_account().get_volume(self.symbol) == 0 else Const.FULL
        model = self.model_when_empty if position == Const.EMPTY else self.model_when_full

        score = model.predict(dataset, verbose=False).ravel()
        return Const.BID if score > 0.5 else Const.ASK

    def observe(self, history: pd.DataFrame, actions):
        raw_features, raw_labels = self.extract_raw_samples(history, actions, self.window_size)
        features = self.feature_manager.get(raw_features)
        labels = self.label_manager.get(raw_labels)
        weights = self.balance_samples(raw_labels)
        train_dataset, test_dataset = self.create_datasets(features, labels, weights, 0.9)

        self.model_for_train, self.model_when_empty, self.model_when_full = self.create_model()

        self.train_model(self.model_for_train, train_dataset, test_dataset, 100)

    @staticmethod
    def extract_raw_features(price_queue, window_size):
        return pd.DataFrame({'Close-{}'.format(i): price_queue[-(i + 1)] for i in range(window_size)}, index=[0])

    @staticmethod
    def extract_raw_samples(history: pd.DataFrame, actions, window_size):
        features = pd.DataFrame()
        for i in range(window_size):
            features['Close-{}'.format(i)] = history.shift(i)['Close']
        features = features[window_size - 1:]
        features = features.reset_index(drop=True)

        labels = pd.DataFrame()
        labels['Action'] = actions[window_size - 1:]
        labels = labels.reset_index(drop=True)

        return features, labels

    @staticmethod
    def balance_samples(labels):
        labels = labels.reset_index()
        labels.columns = ['Index', 'Action']

        statistics = labels.groupby(by=['Action'], dropna=False).size().reset_index()
        statistics.columns = ['Action', 'Weight']
        statistics['Weight'] = statistics['Weight'].sum() / statistics['Weight']

        merge = pd.merge(labels, statistics, on=['Action']).sort_values('Index').reset_index()

        return merge['Weight']

    @staticmethod
    def create_datasets(features, labels, weights, train_pct):
        num_samples = len(weights)
        num_train_samples = int(train_pct * num_samples)
        num_test_samples = num_samples - num_train_samples

        dataset = tf.data.Dataset.from_tensor_slices((features, labels, weights))

        train_dataset = dataset.take(num_train_samples).batch(num_train_samples)
        test_dataset = dataset.skip(num_train_samples).batch(num_test_samples)

        return train_dataset, test_dataset

    @staticmethod
    def create_model():
        input1 = tf.keras.layers.Input(name='price', shape=(30,))
        input2 = tf.keras.layers.Input(name='gain', shape=(29,))
        input3 = tf.keras.layers.Input(name='mean', shape=(4,))
        input4 = tf.keras.layers.Input(name='std', shape=(4,))
        input5 = tf.keras.layers.Input(name='skew', shape=(4,))
        inputs = [input1, input2, input3, input4, input5]

        x = tf.keras.layers.Concatenate()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)

        y = tf.keras.layers.Dense(128, activation='relu')(x)
        y = tf.keras.layers.BatchNormalization(momentum=0)(y)
        y = tf.keras.layers.Dense(128, activation='relu')(y)
        y = tf.keras.layers.BatchNormalization(momentum=0)(y)
        y = tf.keras.layers.Dense(1, activation='sigmoid', name='action_when_empty')(y)

        z = tf.keras.layers.Dense(128, activation='relu')(x)
        z = tf.keras.layers.BatchNormalization(momentum=0)(z)
        z = tf.keras.layers.Dense(128, activation='relu')(z)
        z = tf.keras.layers.BatchNormalization(momentum=0)(z)
        z = tf.keras.layers.Dense(1, activation='sigmoid', name='action_when_full')(z)

        model_for_train = tf.keras.models.Model(inputs=inputs, outputs=[y, z])
        model_when_empty = tf.keras.models.Model(inputs=inputs, outputs=[y])
        model_when_full = tf.keras.models.Model(inputs=inputs, outputs=[z])

        return model_for_train, model_when_empty, model_when_full

    @staticmethod
    def train_model(model: tf.keras.models.Model, train_dataset, test_dataset, epochs):
        model.compile(optimizer='adam',
                      loss={'action_when_empty': 'bce', 'action_when_full': 'bce'},
                      weighted_metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=False,
                  callbacks=[tensor_board_callback, early_stopping_callback])

        model.evaluate(train_dataset)
        model.evaluate(test_dataset)

        return model
