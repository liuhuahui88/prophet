import numpy as np
import tensorflow as tf

from prophet.agent.naive_agent import NaiveAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.agent.smart_agent import SmartAgent
from prophet.bt.back_tester import BackTester
from prophet.bt.broker import Broker
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.metric import Metric

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.01
    log_friction = -(np.log(1 - commission_rate) + np.log(1 / (1 + commission_rate)))

    prices = tf.keras.layers.Input(name='prices', shape=(30,))
    inputs = [prices]

    x = tf.keras.layers.Concatenate()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0)(x)
    x = tf.keras.layers.Dense(1, activation='linear', name='oracle_empty_advantage')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam',
                  loss={'oracle_empty_advantage': 'mse'},
                  metrics=[Metric.create_hard_advt(log_friction),
                           Metric.create_hinge_advt(log_friction),
                           Metric.create_soft_advt(log_friction),
                           Metric.me, Metric.r2])

    data_predictor = DataPredictor()
    data_predictor.set_data_extractor(DataExtractor(commission_rate))
    data_predictor.set_model(model)

    symbol = '600000'
    symbols = ['600000']

    histories = [storage.load_history(s, '2010-01-01', '2011-01-01') for s in symbols]

    data_predictor.learn(histories, histories, 10000, 200, 200, False)

    data_predictor.save_model('models/experimental')

    bt = BackTester(storage, Broker(commission_rate))

    bt.register('SMT', SmartAgent(symbol, data_predictor, 'oracle_empty_advantage', 0, log_friction))
    bt.register('B&H', NaiveAgent(symbol))
    bt.register('ORA', OracleAgent(symbol, storage, commission_rate))

    result = bt.back_test(symbols, '2010-01-01', '2011-01-01')
    result.print()
    result.plot('SMT')
