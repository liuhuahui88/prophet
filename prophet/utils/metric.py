import tensorflow as tf
import tensorflow_probability as tfp

from prophet.utils.operator import Operator


class Metric:

    @staticmethod
    def dummy(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    @staticmethod
    def strategy_soft(y_true, y_pred):
        distribution = tf.nn.softmax(y_pred)
        reward = tf.math.log(tf.reduce_sum(tf.exp(y_true) * distribution, axis=-1))
        base = tf.math.log(tf.reduce_mean(tf.exp(y_true), axis=-1))
        return -(reward - base)

    @staticmethod
    def strategy_top(y_true, y_pred):
        pct = tfp.stats.percentile(y_pred, 90, axis=-1, keepdims=True)
        distribution = tf.nn.softmax(tf.minimum(y_pred, pct) * 100000000)
        reward = tf.math.log(tf.reduce_sum(tf.exp(y_true) * distribution, axis=-1))
        base = tf.math.log(tf.reduce_mean(tf.exp(y_true), axis=-1))
        return -(reward - base)

    @staticmethod
    def strategy_hard(y_true, y_pred):
        distribution = tf.nn.softmax(y_pred * 100000000)
        reward = tf.math.log(tf.reduce_sum(tf.exp(y_true) * distribution, axis=-1))
        base = tf.math.log(tf.reduce_mean(tf.exp(y_true), axis=-1))
        return -(reward - base)

    @staticmethod
    def entropy(y_true, y_pred):
        distribution = tf.nn.softmax(y_pred)
        entropy = Operator.entropy(distribution) / tf.math.log(1.0 * y_pred.shape[-1])
        return entropy

    @staticmethod
    def soft_rank(y_true, y_pred):
        y_true_res_mat = tf.reshape(y_true, [-1, 1]) - tf.reshape(y_true, [1, -1])
        y_pred_res_mat = tf.reshape(y_pred, [-1, 1]) - tf.reshape(y_pred, [1, -1])
        return tf.reduce_mean(Operator.soft_indicator(-y_pred_res_mat * tf.sign(y_true_res_mat)))

    @staticmethod
    def hard_rank(y_true, y_pred):
        y_true_res_mat = tf.reshape(y_true, [-1, 1]) - tf.reshape(y_true, [1, -1])
        y_pred_res_mat = tf.reshape(y_pred, [-1, 1]) - tf.reshape(y_pred, [1, -1])
        return tf.reduce_mean(Operator.hard_indicator(-y_pred_res_mat * tf.sign(y_true_res_mat)))

    @staticmethod
    def me(y_true, y_pred):
        return tf.reduce_mean(y_pred - y_true)

    @staticmethod
    def r2(y_true, y_pred):
        target_var = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))

        residual = y_pred - y_true
        residual_var = tf.reduce_mean(tf.square(residual - tf.reduce_mean(residual)))

        return residual_var / target_var

    @staticmethod
    def point_wise_advt(y_true, y_pred, indicator_fn):
        return tf.abs(y_true) * indicator_fn(-y_pred * tf.sign(y_true))

    @staticmethod
    def pair_wise_advt(y_true, y_pred, delta, indicator_fn):
        empty_advt = Metric.point_wise_advt(y_true, y_pred, indicator_fn)
        full_advt = Metric.point_wise_advt(y_true + delta, y_pred + delta, indicator_fn)
        return tf.reduce_mean(tf.maximum(empty_advt, full_advt))

    @staticmethod
    def create_hard_advt(delta):
        def hard_advt(y_true, y_pred):
            return Metric.pair_wise_advt(y_true, y_pred, delta, Operator.hard_indicator)

        return hard_advt

    @staticmethod
    def create_soft_advt(delta):
        def soft_advt(y_true, y_pred):
            return Metric.pair_wise_advt(y_true, y_pred, delta, Operator.soft_indicator)

        return soft_advt
