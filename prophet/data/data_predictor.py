import datetime

import tensorflow as tf
import pandas as pd

from prophet.data.data_extractor import DataExtractor


class DataPredictor:

    def __init__(self, model):
        self.model = model

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        return DataPredictor(tf.keras.models.load_model(path, compile=False))

    def predict(self, history, data_extractor: DataExtractor):
        size = len(history)

        features = data_extractor.extract(history, self.model.input_names)

        dataset = tf.data.Dataset.from_tensor_slices(features)

        results = self.__invoke_model(dataset, size)

        return results

    def learn(self, histories, train_end_date, data_extractor,
              batch_size, epochs, monitor, patience, verbose, debug):
        features = data_extractor.extract_and_concat(histories, self.model.input_names)
        labels = data_extractor.extract_and_concat(histories, self.model.output_names)
        dates = data_extractor.extract_and_concat(histories, ['date'])['date']['Date']

        train_features, train_labels, train_dataset, train_size =\
            self.__create_dataset(features, labels, dates, lambda d: d <= train_end_date)
        test_features, test_labels, test_dataset, test_size =\
            self.__create_dataset(features, labels, dates, lambda d: d > train_end_date)

        self.__fit_model(train_dataset, train_size, test_dataset, test_size,
                         batch_size, epochs, monitor, patience, verbose)

        if debug:
            train_results = self.__invoke_model(train_dataset, train_size)
            test_results = self.__invoke_model(test_dataset, test_size)

            self.__save_sample(train_features, train_labels, train_results, 'train')
            self.__save_sample(test_features, test_labels, test_results, 'test')

    def __create_dataset(self, features, labels, dates, date_cond):
        features = {k: v[dates.apply(date_cond)] for k, v in features.items()}
        labels = {k: v[dates.apply(date_cond)] for k, v in labels.items()}
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        size = len(dates)
        return features, labels, dataset, size

    def __fit_model(self, train_dataset, train_size, test_dataset, test_size,
                    batch_size, epochs, monitor, patience, verbose):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_dir = "logs/fit/" + timestamp
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)

        train_dataset = train_dataset.shuffle(train_size, reshuffle_each_iteration=False).batch(batch_size).cache()
        test_dataset = test_dataset.shuffle(test_size, reshuffle_each_iteration=False).batch(batch_size).cache()

        self.model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=verbose,
                       callbacks=[tensor_board_callback, early_stopping_callback])

    def __invoke_model(self, dataset, size):
        predictions = self.model.predict(dataset.batch(size), verbose=False)

        if len(self.model.output_names) == 1:
            predictions = [predictions]

        results = {}
        for i in range(len(self.model.output_names)):
            df = pd.DataFrame({'Prediction': predictions[i].ravel()})
            results[self.model.output_names[i]] = df

        return results

    def __save_sample(self, features, labels, results, prefix):
        values = list(features.values()) + list(labels.values()) + list(results.values())
        samples = pd.concat([value.reset_index(drop=True) for value in values], axis=1)
        samples.to_csv('csvs/{}_samples.csv'.format(prefix), index=False)
