import tensorflow as tf
import tensorflow_probability as tfp


class Operator:

    @staticmethod
    def hard_indicator(tensor):
        return (tf.sign(tensor) + 1) / 2

    @staticmethod
    def soft_indicator(tensor):
        return tf.keras.activations.sigmoid(tensor)

    @staticmethod
    def rank(tensor):
        tensor = tf.expand_dims(tensor, axis=-1)
        dims = len(tensor.shape)
        permutation = list(range(dims - 2)) + [dims - 1, dims - 2]
        rank = tf.sign(tensor - tf.transpose(tensor, permutation))
        return tf.reduce_mean(rank, axis=-1)

    @staticmethod
    def risk(distribution, covariance=0, axis=-1, keepdims=False):
        r = tf.math.reduce_sum(tf.square(distribution), axis=axis, keepdims=keepdims)
        return tf.sqrt(r * (1 - covariance) + covariance)

    @staticmethod
    def entropy(distribution, axis=-1, keepdims=False):
        return tf.reduce_sum(-distribution * tf.math.log(distribution), axis=axis, keepdims=keepdims)

    @staticmethod
    def mean_invariant_clip(tensor, pct, side, axis=-1):
        threshold = tfp.stats.percentile(tensor, pct, axis=axis, keepdims=True)
        mask = Operator.hard_indicator(side * (tensor - threshold))

        value = tf.reduce_sum(mask * tensor, axis=axis, keepdims=True)
        cnt = tf.reduce_sum(mask, axis=axis, keepdims=True)
        avg = value / cnt

        return mask * avg + (1 - mask) * tensor

