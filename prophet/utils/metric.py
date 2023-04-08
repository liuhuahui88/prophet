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
        def hard(tensor):
            return (tf.sign(tensor) + 1) / 2

        def hard_advt(y_true, y_pred):
            return Metric.pair_wise_advt(y_true, y_pred, delta, hard)

        return hard_advt

    @staticmethod
    def create_hinge_advt(delta):
        def hinge(tensor):
            width = delta / 2
            margin_pct = 0.1
            margin = width * margin_pct
            return tf.keras.activations.relu(tensor + margin)

        def hinge_advt(y_true, y_pred):
            return Metric.pair_wise_advt(y_true, y_pred, delta, hinge)

        return hinge_advt

    @staticmethod
    def create_soft_advt(delta):
        def soft(tensor):
            return tf.keras.activations.sigmoid(tensor)

        def soft_advt(y_true, y_pred):
            return Metric.pair_wise_advt(y_true, y_pred, delta, soft)

        return soft_advt
