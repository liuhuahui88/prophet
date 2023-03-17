import datetime

import tensorflow as tf
import pandas as pd

from prophet.data.data_extractor import DataExtractor


class DataPredictor:

    def __init__(self, model: tf.keras.models.Model):
        self.model = model
        self.feature_extractor = DataExtractor(model.input_names)
        self.label_extractor = DataExtractor(model.output_names)

    def predict(self, history: pd.DataFrame):
        features = self.feature_extractor.extract(history)
        dataset = tf.data.Dataset.from_tensor_slices(features).batch(len(history))
        return self.model.predict(dataset, verbose=False)

    def train(self, history: pd.DataFrame, train_pct, epochs, patience):
        features = self.feature_extractor.extract(history)
        labels = self.label_extractor.extract(history)
        train_dataset, test_dataset = self.create_dataset(features, labels, len(history), train_pct)
        self.fit_model(self.model, train_dataset, test_dataset, epochs, patience)
        self.eval_model(self.model, train_dataset, 'train')
        self.eval_model(self.model, test_dataset, 'test')

    @staticmethod
    def create_dataset(features, labels, num_samples, train_pct):
        num_train_samples = int(train_pct * num_samples)
        num_test_samples = num_samples - num_train_samples

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        train_dataset = dataset.take(num_train_samples).batch(num_train_samples)
        test_dataset = dataset.skip(num_train_samples).batch(num_test_samples)

        return train_dataset, test_dataset

    @staticmethod
    def fit_model(model: tf.keras.models.Model, train_dataset, test_dataset, epochs, patience):
        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=False,
                  callbacks=[tensor_board_callback, early_stopping_callback])

    @staticmethod
    def eval_model(model: tf.keras.models.Model, dataset, name):
        model.evaluate(dataset)

        predictions = model.predict(dataset, verbose=False)
        df = pd.DataFrame()
        df['Prediction'] = predictions.ravel()
        df.to_csv('csvs/prediction_{}.csv'.format(name), index=False)
