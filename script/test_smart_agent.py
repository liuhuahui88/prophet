from prophet.predictor.sk_predictor import SkPredictor
from prophet.predictor.tf_predictor import TfPredictor
from prophet.utils.play_ground import PlayGround

if __name__ == '__main__':
    play_ground = PlayGround(
        name_file_path='../data/chinese_stock_codes.csv',
        history_file_path='../data/history',
        commission_rate=0.0)

    symbols = play_ground.storage.get_symbols(lambda s: s[0] == '3' and s <= '300200')
    predictors = {
        'SOTA_PLS': SkPredictor.load('models/sota_pls'),
        'SOTA_GB': SkPredictor.load('models/sota_gb'),
        'SOTA_RANK': TfPredictor.load('models/sota_rank'),
        'SOTA_REG': TfPredictor.load('models/sota_reg'),
        'SOTA_CLS': TfPredictor.load('models/sota_cls'),
    }
    delta_free_list = ['SOTA_RANK', 'SOTA_REG', 'SOTA_CLS']

    result = play_ground.test(
        symbols, '2022-01-01', '2023-01-01', predictors, 0, delta_free_list,
        global_threshold=-10000, local_threshold=-10000, top_k=20, weighted=False,
        with_baseline=True, verbose=True)
    result.print()
    result.plot()
