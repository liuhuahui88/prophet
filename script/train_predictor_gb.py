from prophet.predictor.sk_predictor import SkPredictor
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':

    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    prefixes = ['log_price', 'log_price_diff', 'log_price_diff', 'log_price_diff']
    suffixes = ['rank', 'rrank', 'ordered', 'mean', 'std', 'skew', 'kurt', 'max', 'min', 'spread']
    feature_names = [p + '_' + s for p in prefixes for s in suffixes]

    predictor = SkPredictor(feature_names, ['next_log_price_diff'], HistGradientBoostingRegressor())

    symbols = play_ground.storage.get_symbols(lambda s: s[0] == '3' and s <= '300800')

    play_ground.train(symbols, '2010-01-01', '2021-01-01', '2022-01-01', predictor)

    predictor.save('models/temp_gb')
