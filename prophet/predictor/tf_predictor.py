import datetime

import tensorflow as tf
import pandas as pd

from prophet.predictor.abstract_predictor import Predictor


class TfPredictor(Predictor):

    def __init__(self, model, batch_size=None, epochs=None, monitor=None, patience=None, verbose=None):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose

    def get_feature_names(self):
        return self.model.input_names

    def get_label_names(self):
        return self.model.output_names

    def fit(self, train_sample_set: Predictor.SampleSet, test_sample_set: Predictor.SampleSet):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_dir = "logs/fit/" + timestamp
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor, patience=self.patience, restore_best_weights=True)

        train_dataset = self.__create_dataset(train_sample_set, self.batch_size)
        test_dataset = self.__create_dataset(test_sample_set, self.batch_size)

        self.model.fit(train_dataset, epochs=self.epochs, validation_data=test_dataset, verbose=self.verbose,
                       callbacks=[tensor_board_callback, early_stopping_callback])

    def predict(self, sample_set: Predictor.SampleSet):
        dataset = tf.data.Dataset.from_tensor_slices(sample_set.features).batch(sample_set.size)
        predictions = self.model.predict(dataset, verbose=False)

        if len(self.model.output_names) == 1:
            predictions = [predictions]

        results = {}
        for i, name in enumerate(self.model.output_names):
            results[name] = pd.DataFrame({'Prediction': predictions[i].ravel()})
        return results

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        return TfPredictor(tf.keras.models.load_model(path, compile=False))

    @staticmethod
    def __create_dataset(sample_set, batch_size):
        return tf.data.Dataset.from_tensor_slices((sample_set.features, sample_set.labels))\
            .cache().shuffle(sample_set.size, reshuffle_each_iteration=True).batch(batch_size)
