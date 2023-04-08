import tensorflow as tf


class Metric:

    @staticmethod
    def me(y_true, y_pred):
        return tf.reduce_mean(y_pred - y_true)

    @staticmethod
    def r2(y_true, y_pred):
        target_var = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))

        residual = y_pred - y_true
        residual_var = tf.reduce_mean(tf.square(residual - tf.reduce_mean(residual)))

        return residual_var / target_var
