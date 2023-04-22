import tensorflow as tf

from prophet.utils.input_builder import InputBuilder
from prophet.utils.metric import Metric
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':

    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    suffixes = ['rank', 'rrank', 'ord1']
    inputs = InputBuilder()\
        .append_all('log_price', suffixes)\
        .append_all('log_price_diff', suffixes)\
        .append_all('log_price_diff_diff', suffixes)\
        .append_all('log_price_diff_diff_diff', suffixes)\
        .build()

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
