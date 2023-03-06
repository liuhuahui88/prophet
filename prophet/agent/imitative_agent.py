import collections

import pandas as pd
import tensorflow as tf

from prophet.agent.abstract_agent import Agent
from prophet.utils.constant import Const


class ImitativeAgent(Agent):

    def __init__(self, symbol, window_size):
        self.symbol = symbol
        self.window_size = window_size
        self.price_queue = collections.deque([], maxlen=window_size)
        self.model = None

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
        features = self.extract_features(ctx, self.symbol, self.price_queue, self.window_size)
        features = self.transform_features(features)

        dataset = tf.data.Dataset.from_tensor_slices(features).batch(1)

        score = self.model.predict(dataset, verbose=False).ravel()
        return Const.BID if score > 0.5 else Const.ASK

    def observe(self, history: pd.DataFrame, actions):
        features, labels = self.extract_samples(history, actions, self.window_size)
        features, labels = self.augment_samples(features, labels)
        features = self.transform_features(features)

        train_dataset, test_dataset = self.create_datasets(features, labels, 0.9)

        self.model = self.create_model()
        self.train_model(self.model, train_dataset, test_dataset, 100)

    @staticmethod
    def extract_features(ctx: Agent.Context, symbol, price_queue, window_size):
        feature_dict = {'Close-{}'.format(i): price_queue[-(i + 1)] for i in range(window_size)}

        volume = ctx.get_account().get_volume(symbol)
        feature_dict['Position'] = Const.FULL if volume != 0 else Const.EMPTY

        return pd.DataFrame(feature_dict, index=[0])

    @staticmethod
    def transform_features(features: pd.DataFrame):
        position = features.pop('Position')
        return {'position': position, 'price': features}

    @staticmethod
    def extract_samples(history: pd.DataFrame, actions, window_size):
        features = pd.DataFrame()
        for i in range(window_size):
            features['Close-{}'.format(i)] = history.shift(i)['Close']
        features = features[window_size - 1:]

        labels = pd.DataFrame()
        labels['Action'] = actions[window_size - 1:]

        return features, labels

    @staticmethod
    def augment_samples(features: pd.DataFrame, labels: pd.DataFrame):
        full_pos_features, full_pos_labels = ImitativeAgent.augment_samples_inner(features, labels, Const.FULL, Const.BID)
        empty_pos_features, empty_pos_labels = ImitativeAgent.augment_samples_inner(features, labels, Const.EMPTY, Const.ASK)

        features = pd.concat([full_pos_features, empty_pos_features], ignore_index=True)
        labels = pd.concat([full_pos_labels, empty_pos_labels], ignore_index=True)

        return features, labels

    @staticmethod
    def augment_samples_inner(features: pd.DataFrame, labels: pd.DataFrame, position, default_action):
        features = features.copy()
        features['Position'] = position

        labels = labels.copy()
        labels['Action'].fillna(value=default_action, inplace=True)

        return features, labels

    @staticmethod
    def create_datasets(features, labels, train_pct):
        num_samples = len(labels)
        num_train_samples = int(train_pct * num_samples)
        num_test_samples = num_samples - num_train_samples

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)

        train_dataset = dataset.take(num_train_samples).batch(num_train_samples)
        test_dataset = dataset.skip(num_train_samples).batch(num_test_samples)

        return train_dataset, test_dataset

    @staticmethod
    def create_model():
        input1 = tf.keras.layers.Input(name='position', shape=(1,))
        input2 = tf.keras.layers.Input(name='price', shape=(30,))
        inputs = [input1, input2]

        x = tf.keras.layers.Concatenate()(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=x)

    @staticmethod
    def train_model(model, train_dataset, test_dataset, epochs):
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=False)
        model.evaluate(train_dataset)
        model.evaluate(test_dataset)
        return model
