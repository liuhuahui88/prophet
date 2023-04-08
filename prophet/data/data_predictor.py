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

    def train(self, histories, sample_pct, batch_pct, epochs, patience):
        num_samples = 0
        for history in histories:
            num_samples += len(history)

        features = self.extract_and_concat(histories, self.data_extractor, self.model.input_names)
        labels = self.extract_and_concat(histories, self.data_extractor, self.model.output_names)
        train_dataset, test_dataset = self.create_dataset(features, labels, num_samples, sample_pct, batch_pct)
        self.fit_model(self.model, train_dataset, test_dataset, epochs, patience)
        self.eval_model(self.model, train_dataset, 'train')
        self.eval_model(self.model, test_dataset, 'test')

    @staticmethod
    def extract_and_concat(histories, data_extractor, names):
        datas = [data_extractor.extract(history, names) for history in histories]
        return {name: pd.concat([data[name] for data in datas]) for name in names}

    @staticmethod
    def create_dataset(features, labels, num_samples, train_pct, batch_pct):
        num_train_samples = int(num_samples * train_pct)
        num_test_samples = num_samples - num_train_samples

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(num_samples, reshuffle_each_iteration=False)

        train_dataset = dataset.take(num_train_samples).batch(int(num_train_samples * batch_pct))
        test_dataset = dataset.skip(num_train_samples).batch(num_test_samples)

        samples = pd.concat([v for v in features.values()] + [v for v in labels.values()], axis=1)
        samples.to_csv('csvs/samples.csv', index=False)

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
        if len(model.output_names) == 1:
            predictions = [predictions]

        df = pd.DataFrame()
        for i in range(len(model.output_names)):
            df[model.output_names[i]] = predictions[i].ravel()
        df.to_csv('csvs/prediction_{}.csv'.format(name), index=False)
