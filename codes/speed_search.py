from codes.config import *
from scipy.stats.stats import pearsonr
from codes.market import market
from codes.base import plot_simi_stock, norm, weighted_distance

def _speed_search(code=None, pattern=None, targets=None):
    if pattern is None:
        all_data, pattern, targets = market.get_historical_data(start_date=config.start_date, code=code)

    ALPHA = config.alpha
    BETA = config.beata

    targets['fft_deg'] = 0
    for i in range(config.fft_level):
        index = str(i + 1)
        p_fft = pattern['fft' + index].tail(1).values
        p_deg = pattern['deg' + index].tail(1).values

        targets['fft_' + index] = (targets['fft' + index] - p_fft).abs() * ALPHA[i]
        targets['deg_' + index] = (targets['deg' + index] - p_deg).abs() * BETA[i]

        targets['fft_deg'] += targets['fft_' + index] + targets['deg_' + index]

    sorted_std_diff = targets.sort_values(ascending=True, by=['fft_deg'])
    sorted_std_diff = sorted_std_diff.head(config.nb_similar_of_each_stock)

    distances = []
    for _, ith in sorted_std_diff.iterrows():
        result = targets[(targets['CODE'] == ith['CODE']) & (targets['DATE'] <= ith['DATE'])].tail(config.pattern_length)
        distances.append(weighted_distance(norm(result['CLOSE'], result[config.market_ratio_type]),
                                           norm(pattern['CLOSE'], pattern[config.market_ratio_type]),
                                           config.pattern_length))

    sorted_std_diff[config.similarity_method] = distances
    tops = sorted_std_diff.sort_values(ascending=True, by=[config.similarity_method])

    tops['pattern'] = code
    return tops

def parallel_speed_search(code):
    all_data, pattern, targets = market.get_historical_data(start_date=config.start_date, code=code)
    tops = _speed_search(pattern=pattern, targets=targets)
    tops = tops.head(config.nb_similar)

    if config.plot_simi_stock:
        plot_simi_stock(tops, all_data, pattern, code + '_simi_result', codes=code)

    tops = tops[['CODE', 'DATE', config.similarity_method]]

    pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20 = 0, 0, 0, 0

    # top n similar stocks about a stock
    for _, top in tops.iterrows():
        pred = market.get_data(start_date=top['DATE'], code=top['CODE'])
        pred_ratios1 += (pred.iloc[1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']
        pred_ratios5 += (pred.iloc[5]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']
        pred_ratios10 += (pred.iloc[10]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']
        pred_ratios20 += (pred.iloc[20]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']

    return [code, pred_ratios1/tops.shape[0], pred_ratios5/tops.shape[0],
            pred_ratios10/tops.shape[0], pred_ratios20/tops.shape[0]]

if __name__ == '__main__':

    all_data, pattern, targets = market.get_historical_data(config.start_date, code=config.code)

    tops = _speed_search(pattern, targets, config.code)
    tops = tops.head(config.nb_similar)
    plot_simi_stock(tops, all_data, pattern, 'speed_search_' + config.speed_method, codes=config.code)