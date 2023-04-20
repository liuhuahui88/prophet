from prophet.data.data_predictor import DataPredictor
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':
    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    symbols = play_ground.storage.get_symbols(lambda s: s[0] == '3' and s <= '300010')
    predictors = {
        'TEMP': DataPredictor.load('models/temp'),
        'SOTA_RANK': DataPredictor.load('models/sota_rank'),
        'SOTA_REG': DataPredictor.load('models/sota_reg'),
        'SOTA_CLS': DataPredictor.load('models/sota_cls'),
    }
    delta_free_list = ['SOTA_RANK', 'SOTA_REG', 'SOTA_CLS']

    for symbol in symbols:
        result = play_ground.test_smart_agent(
            symbol, '2022-01-01', '2023-01-01',
            predictors, delta_free_list, with_baseline=True)

        result.print()
        result.plot()
