import tensorflow as tf

from prophet.utils.metric import Metric
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':
    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.01)

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
                  metrics=[Metric.create_hard_advt(play_ground.log_friction),
                           Metric.create_hinge_advt(play_ground.log_friction),
                           Metric.create_soft_advt(play_ground.log_friction),
                           Metric.me, Metric.r2])
    symbol = '600000'
    symbols = [symbol]

    predictor = play_ground.train_predictor(
        symbols, '2010-01-01', '2010-12-01', '2011-01-01',
        model, 10000, 200, 200)

    predictor.save('models/example')

    result = play_ground.test_smart_agent(
        symbol, '2010-07-01', '2011-01-01',
        {'EXAMPLE': predictor}, with_baseline=True, with_oracle=True)

    result.print()
    result.plot()
