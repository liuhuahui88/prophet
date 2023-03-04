import collections

import pandas as pd
import tensorflow as tf

from prophet.agent.abstract_agent import Agent


class ImitativeAgent(Agent):

    BID = 1
    ASK = 0

    FULL = 1
    EMPTY = 0

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
        if action == self.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

    def observe(self, history: pd.DataFrame, actions):
        raw_features, raw_labels = self.extract_samples(history, actions, self.window_size)

        full_pos_features, full_pos_labels = self.augment_samples(raw_features, raw_labels, self.FULL, self.BID)
        empty_pos_features, empty_pos_labels = self.augment_samples(raw_features, raw_labels, self.EMPTY, self.ASK)

        features = pd.concat([full_pos_features, empty_pos_features], ignore_index=True)
        labels = pd.concat([full_pos_labels, empty_pos_labels], ignore_index=True)

        transformed_features = self.transform_features(features)

        self.model = self.train_model(transformed_features, labels)

    def predict(self, ctx: Agent.Context):
        feature_dict = {'Close-{}'.format(i): self.price_queue[-(i + 1)] for i in range(self.window_size)}

        volume = ctx.get_account().get_volume(self.symbol)
        feature_dict['Position'] = self.FULL if volume != 0 else self.EMPTY

        feature_df = pd.DataFrame(feature_dict, index=[0])
        transformed_feature_df = self.transform_features(feature_df)
        dataset = tf.data.Dataset.from_tensor_slices(transformed_feature_df).batch(1)
        score = self.model.predict(dataset, verbose=False).ravel()

        return self.BID if score > 0.5 else self.ASK

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
    def augment_samples(features: pd.DataFrame, labels: pd.DataFrame, position, default_action):
        features = features.copy()
        features['Position'] = position

        labels = labels.copy()
        labels['Action'].fillna(value=default_action, inplace=True)

        return features, labels

    @staticmethod
    def transform_features(features: pd.DataFrame):
        return features

    @staticmethod
    def train_model(features: pd.DataFrame, labels: pd.DataFrame):
        train_pct = 0.9

        num_samples = len(features)
        num_train_samples = int(train_pct * num_samples)
        num_test_samples = num_samples - num_train_samples

        dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
        dataset.shuffle(num_samples)

        train_dataset = dataset.take(num_train_samples).batch(num_train_samples)
        test_dataset = dataset.skip(num_train_samples).batch(num_test_samples)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(momentum=0),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
        model.fit(train_dataset, epochs=100, validation_data=test_dataset, verbose=False)
        model.evaluate(train_dataset)
        model.evaluate(test_dataset)
        return model
