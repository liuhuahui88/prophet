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
        raw_samples = self.generate_samples(history, actions, self.window_size)

        full_pos_samples = self.augment_samples(raw_samples, self.FULL, self.BID)
        empty_pos_samples = self.augment_samples(raw_samples, self.EMPTY, self.ASK)

        samples = pd.concat([full_pos_samples, empty_pos_samples], ignore_index=True)

        self.model = self.train_model(samples)

    def predict(self, ctx: Agent.Context):
        feature_dict = {'Close-{}'.format(i): self.price_queue[-(i + 1)] for i in range(self.window_size)}

        volume = ctx.get_account().get_volume(self.symbol)
        feature_dict['Position'] = self.FULL if volume != 0 else self.EMPTY

        feature_df = pd.DataFrame(feature_dict, index=[0])
        dataset = tf.data.Dataset.from_tensor_slices(feature_df).batch(1)
        score = self.model.predict(dataset, verbose=False).ravel()

        return self.BID if score > 0.5 else self.ASK

    @staticmethod
    def generate_samples(history: pd.DataFrame, actions, window_size):
        raw_samples = pd.DataFrame()
        for i in range(window_size):
            raw_samples['Close-{}'.format(i)] = history.shift(i)['Close']
        raw_samples['Action'] = actions
        raw_samples = raw_samples[window_size - 1:]
        return raw_samples

    @staticmethod
    def augment_samples(raw_samples, position, default_action):
        samples = raw_samples.copy()
        samples['Position'] = position
        samples['Action'].fillna(value=default_action, inplace=True)
        return samples

    @staticmethod
    def train_model(samples: pd.DataFrame):
        target = samples.pop('Action')
        dataset = tf.data.Dataset.from_tensor_slices((samples.values, target.values))
        train_dataset = dataset.batch(len(samples))
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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_dataset, epochs=100, verbose=False)
        model.evaluate(train_dataset)
        return model
