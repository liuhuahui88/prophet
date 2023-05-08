from abc import ABC, abstractmethod


class Predictor(ABC):

    class SampleSet:

        def __init__(self, ids, features, labels, size):
            self.ids = ids
            self.features = features
            self.labels = labels
            self.size = size

    @abstractmethod
    def get_feature_names(self):
        pass

    @abstractmethod
    def get_label_names(self):
        pass

    @abstractmethod
    def fit(self, train_sample_set: SampleSet, test_sample_set: SampleSet):
        pass

    @abstractmethod
    def predict(self, sample_set: SampleSet):
        pass
