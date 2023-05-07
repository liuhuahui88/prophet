from prophet.predictor.tf_predictor import TfPredictor
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':
    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    symbols = play_ground.storage.get_symbols(lambda s: s[0] == '3' and s <= '300200')
    predictors = {
        'TEMP': TfPredictor.load('models/temp'),
        'SOTA_RANK': TfPredictor.load('models/sota_rank'),
        'SOTA_REG': TfPredictor.load('models/sota_reg'),
        'SOTA_CLS': TfPredictor.load('models/sota_cls'),
    }
    delta_free_list = ['SOTA_RANK', 'SOTA_REG', 'SOTA_CLS']

    result = play_ground.test(
        symbols, '2022-01-01', '2023-01-01',
        predictors, delta_free_list, top_k=20, with_baseline=True)
    result.print()
    result.plot()
