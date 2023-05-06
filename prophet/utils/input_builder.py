import tensorflow as tf


class InputBuilder:

    def __init__(self, default_shape=(1,)):
        self.default_shape = default_shape
        self.input_name_and_shape = {}

    def append(self, name, shape=None):
        assert name not in self.input_name_and_shape
        self.input_name_and_shape[name] = shape if shape is not None else self.default_shape
        return self

    def append_all(self, prefix, suffixes, shape=None):
        for suffix in suffixes:
            self.append(prefix + '_' + suffix, shape)
        return self

    def build(self):
        return [tf.keras.layers.Input(name=name, shape=shape) for name, shape in self.input_name_and_shape.items()]
