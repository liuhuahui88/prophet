import datetime

import tensorflow as tf
import pandas as pd

from prophet.data.data_extractor import DataExtractor


class DataPredictor:

    def __init__(self, model: tf.keras.models.Model, data_extractor: DataExtractor):
        self.model = model
        self.data_extractor = data_extractor

    def predict(self, history: pd.DataFrame):
        features = self.data_extractor.extract(history, self.model.input_names)
        dataset = tf.data.Dataset.from_tensor_slices(features).batch(len(history))
        return self.model.predict(dataset, verbose=False)

    def train(self, histories, train_pct, batch_pct, epochs, patience, verbose=False):
        dataset, size = self.create_dataset(histories)
        self.fit_model(dataset, size, train_pct, batch_pct, epochs, patience, verbose)
        self.eval_model(dataset, size, verbose)

    def create_dataset(self, histories):
        size = 0
        for history in histories:
            size += len(history)

        features = self.data_extractor.extract_and_concat(histories, self.model.input_names)
        labels = self.data_extractor.extract_and_concat(histories, self.model.output_names)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        samples = pd.concat([v for v in features.values()] + [v for v in labels.values()], axis=1)
        samples.to_csv('csvs/samples.csv', index=False)

        return dataset, size

    def fit_model(self, dataset, size, train_pct, batch_pct, epochs, patience, verbose):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_dir = "logs/fit/" + timestamp
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        dataset = dataset.shuffle(size, reshuffle_each_iteration=False)

        train_size = int(size * train_pct)
        batch_size = int(size * batch_pct)

        train_dataset = dataset.take(train_size).batch(batch_size)
        test_dataset = dataset.skip(train_size).batch(batch_size)

        self.model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=verbose,
                       callbacks=[tensor_board_callback, early_stopping_callback])

    def eval_model(self, dataset, size, verbose):
        eval_dataset = dataset.batch(size)

        self.model.evaluate(eval_dataset, verbose=verbose)

        predictions = self.model.predict(eval_dataset, verbose=False)
        if len(self.model.output_names) == 1:
            predictions = [predictions]

        df = pd.DataFrame()
        for i in range(len(self.model.output_names)):
            df[self.model.output_names[i]] = predictions[i].ravel()
        df.to_csv('csvs/prediction.csv', index=False)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
