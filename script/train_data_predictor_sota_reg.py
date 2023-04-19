import tensorflow as tf

from prophet.utils.metric import Metric
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':

    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    inputs = [
        # tf.keras.layers.Input(name='log_price_diff', shape=(1,)),
        tf.keras.layers.Input(name='log_price_diff_mean', shape=(1,)),
        tf.keras.layers.Input(name='log_price_diff_std', shape=(1,)),
        tf.keras.layers.Input(name='log_price_diff_skew', shape=(1,)),
        tf.keras.layers.Input(name='log_price_diff_kurt', shape=(1,)),

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
    x = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='zeros', name='next_log_price_diff')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam', loss='mse', metrics=[Metric.me, Metric.r2])

    symbols = play_ground.storage.get_symbols(lambda s: s[0] == '3' and s <= '300800')

    predictor = play_ground.train_predictor(
        symbols, '2010-01-01', '2021-01-01', '2022-01-01',
        model, 1000, 1000, 'val_loss', 3, verbose=True)

    predictor.save('models/temp_reg')
