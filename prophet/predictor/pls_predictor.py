import os
import shutil
import pickle

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from prophet.predictor.abstract_predictor import Predictor


class PlsPredictor(Predictor):

    def __init__(self, feature_names, label_names, components):
        self.feature_names = feature_names
        self.label_names = label_names
        self.model = PLSRegression(components)

    def get_feature_names(self):
        return self.feature_names

    def get_label_names(self):
        return self.label_names

    def fit(self, train_sample_set, test_sample_set):
        features = self.__concat(train_sample_set.features)
        labels = self.__concat(train_sample_set.labels)
        self.model.fit(features, labels)

    def predict(self, sample_set):
        features = self.__concat(sample_set.features)
        predictions = self.model.predict(features)

        predictions = np.reshape(predictions, [len(self.label_names), -1])

        results = {}
        for i, name in enumerate(self.label_names):
            results[name] = pd.DataFrame({'Prediction': predictions[i]})
        return results

    def save(self, path):
        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)
        pickle.dump(self, open(path + '/model.pk', 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path + '/model.pk', 'rb'))

    @staticmethod
    def __concat(datas):
        return pd.concat(datas.values(), axis=1)
