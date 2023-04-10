import datetime

import tensorflow as tf
import pandas as pd

from prophet.data.data_extractor import DataExtractor


class DataPredictor:

    def __init__(self, model: tf.keras.models.Model, data_extractor: DataExtractor):
        self.model = model
        self.data_extractor = data_extractor

    def predict(self, history: pd.DataFrame):
        size = len(history)

        features = self.data_extractor.extract(history, self.model.input_names)

        dataset = tf.data.Dataset.from_tensor_slices(features)

        results = self.invoke_model(dataset, size)

        return results

    def train(self, histories, train_pct, batch_pct, epochs, patience, verbose=False):
        size = sum([len(history) for history in histories])

        features = self.data_extractor.extract_and_concat(histories, self.model.input_names)
        labels = self.data_extractor.extract_and_concat(histories, self.model.output_names)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        self.fit_model(dataset, size, train_pct, batch_pct, epochs, patience, verbose)

        results = self.invoke_model(dataset, size)

        values = list(features.values()) + list(labels.values()) + list(results.values())
        samples = pd.concat([value.reset_index(drop=True) for value in values], axis=1)
        samples.to_csv('csvs/feature_label_result.csv', index=False)

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

    def invoke_model(self, dataset, size):
        predictions = self.model.predict(dataset.batch(size), verbose=False)

        if len(self.model.output_names) == 1:
            predictions = [predictions]

        results = {}
        for i in range(len(self.model.output_names)):
            df = pd.DataFrame({'Prediction': predictions[i].ravel()})
            results[self.model.output_names[i]] = df

        return results

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
