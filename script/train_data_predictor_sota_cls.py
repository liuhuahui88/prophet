import numpy as np
import tensorflow as tf

from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.metric import Metric

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.0
    log_friction = -(np.log(1 - commission_rate) + np.log(1 / (1 + commission_rate)))

    inputs = [
        # tf.keras.layers.Input(name='log_gain', shape=(1,)),
        tf.keras.layers.Input(name='log_gain_mean', shape=(1,)),
        tf.keras.layers.Input(name='log_gain_std', shape=(1,)),
        tf.keras.layers.Input(name='log_gain_skew', shape=(1,)),
        tf.keras.layers.Input(name='log_gain_kurt', shape=(1,)),

        tf.keras.layers.Input(name='log_price_rank', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_mean', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_std', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_skew', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_kurt', shape=(1,)),

        tf.keras.layers.Input(name='log_price_rank_diff', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_mean', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_std', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_skew', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_kurt', shape=(1,)),

        tf.keras.layers.Input(name='log_price_rank_diff_diff', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_diff_mean', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_diff_std', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_diff_skew', shape=(1,)),
        tf.keras.layers.Input(name='log_price_rank_diff_diff_kurt', shape=(1,)),

        tf.keras.layers.Input(name='keep_lose', shape=(1,)),
        tf.keras.layers.Input(name='keep_gain', shape=(1,)),
    ]

    t_inputs = []
    for inpt in inputs:
        inpt = tf.keras.layers.Dense(16, activation='relu')(inpt)
        inpt = tf.keras.layers.Dense(16, activation='relu')(inpt)
        t_inputs.append(inpt)

    x = tf.keras.layers.Concatenate()(inputs + t_inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='linear', name='next_direction')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['AUC', Metric.create_hard_advt(log_friction)])

    data_predictor = DataPredictor()
    data_predictor.set_data_extractor(DataExtractor(commission_rate))
    data_predictor.set_model(model)

    symbols = [s for s in storage.get_symbols() if s[0] == '3' and s <= '300800']

    histories = [storage.load_history(s, '2010-01-01', '2022-01-01') for s in symbols]

    data_predictor.train(histories, 0.9, 0.001, 1000, 3, True)

    data_predictor.save_model('models/experimental')
