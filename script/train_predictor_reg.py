import tensorflow as tf

from prophet.predictor.tf_predictor import TfPredictor
from prophet.utils.input_builder import InputBuilder
from prophet.utils.metric import Metric
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':

    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    suffixes = ['rank', 'rrank', 'ordered']
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

    model.compile(optimizer='adam', loss='mse', metrics=[Metric.me, Metric.r2])

    predictor = TfPredictor(model, 1000, 1000, 'val_loss', 3, verbose=True)

    symbols = play_ground.storage.get_symbols(lambda s: s[0] == '3' and s <= '300800')

    play_ground.train(symbols, '2010-01-01', '2021-01-01', '2022-01-01', predictor, 0)

    predictor.save('models/temp_reg')
