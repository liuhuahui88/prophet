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
    extractor = DataExtractor(commission_rate)

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
    x = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='zeros', name='next_log_gain')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam', loss='mse',
                  metrics=[Metric.r2, Metric.create_hard_advt(log_friction)])

    predictor = DataPredictor(model)

    symbols = storage.get_symbols(lambda s: s[0] == '3' and s <= '300800')

    train_histories = storage.load_histories(symbols, '2010-01-01', '2021-01-01')
    test_histories = storage.load_histories(symbols, '2021-01-01', '2022-01-01')

    predictor.learn(train_histories, test_histories, extractor, 1000, 1000, 3, True)

    predictor.save('models/temp_reg')
