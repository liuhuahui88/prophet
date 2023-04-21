import tensorflow as tf

from prophet.utils.metric import Metric
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':
    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    inputs = [
        tf.keras.layers.Input(name='log_price_diff_rank', shape=(1,)),
        tf.keras.layers.Input(name='log_price_diff_ord1', shape=(1,)),
        tf.keras.layers.Input(name='log_price_diff_ord2', shape=(1,)),
        tf.keras.layers.Input(name='log_price_diff_ord3', shape=(1,)),

        tf.keras.layers.Input(name='log_price_rank', shape=(1,)),
        tf.keras.layers.Input(name='log_price_ord1', shape=(1,)),
        tf.keras.layers.Input(name='log_price_ord2', shape=(1,)),
        tf.keras.layers.Input(name='log_price_ord3', shape=(1,)),
    ]

    dropout_inputs = [tf.keras.layers.Dropout(0.2)(node) for node in inputs]

    embedding_inputs = []
    for node in dropout_inputs:
        node = tf.keras.layers.Dense(16, activation='relu')(node)
        node = tf.keras.layers.Dense(16, activation='relu')(node)
        embedding_inputs.append(node)

    x = tf.keras.layers.Concatenate()(dropout_inputs + embedding_inputs)
    x = tf.keras.layers.Dense(1, activity_regularizer='l2', name='next_log_price_diff')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam',
                  loss=Metric.soft_rank,
                  metrics=[Metric.me, Metric.hard_rank])

    symbols = play_ground.storage.get_symbols(lambda s: s[0] == '3' and s <= '300800')

    predictor = play_ground.train_predictor(
        symbols, '2010-01-01', '2021-01-01', '2022-01-01',
        model, 1000, 1000, 'val_loss', 3, verbose=True)

    predictor.save('models/temp_rank')
