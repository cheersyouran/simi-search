from codes.config import *
from scipy.stats.stats import pearsonr
from codes.market import market
from codes.base import plot_simi_stock, norm, weighted_distance


def find_similar_of_a_stock(code):
    all_data, pattern, targets = market.get_historical_data(end_date=config.start_date, code=code)

    if pattern is None:
        print('---Pattern is None for ', code)
        return None

    ALPHA = config.alpha
    BETA = config.beata

    targets['fft_deg'] = 0
    for i in range(config.fft_level):
        index = str(i + 1)
        p_fft = pattern['fft' + index].tail(1).values
        p_deg = pattern['deg' + index].tail(1).values

        targets['fft' + index] = (targets['fft' + index] - p_fft).abs() * ALPHA[i]
        targets['deg' + index] = (targets['deg' + index] - p_deg).abs() * BETA[i]

        targets['fft_deg'] += targets['fft' + index] + targets['deg' + index]

    sorted_std_diff = targets.sort_values(ascending=True, by=['fft_deg'])
    sorted_std_diff = sorted_std_diff.head(config.nb_similar_of_each_stock)

    distances = []
    tmp = all_data[all_data['DATE'] < market.current_date.date()]
    for _, ith in sorted_std_diff.iterrows():
        result = tmp[(tmp['CODE'] == ith['CODE']) & (tmp['DATE'] <= ith['DATE'])].tail(config.pattern_length)
        distances.append(weighted_distance(norm(result['RET'], result['300_RATIO']),
                                           norm(pattern['RET'], pattern['300_RATIO']),
                                           config.pattern_length))

    sorted_std_diff[config.similarity_method] = distances
    tops = sorted_std_diff.sort_values(ascending=True, by=[config.similarity_method])

    tops['pattern'] = code
    if config.plot_simi_stock:
        plot_simi_stock(tops.head(config.nb_similar_make_prediction), all_data, pattern, code + '_simi_result', codes=code)

    print('---Finish searching : ', code)
    return tops


def predict_stock_base_on_similars(code):
    tops = find_similar_of_a_stock(code)

    if tops is None:
        return None

    tops = tops.head(config.nb_similar_make_prediction)
    pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20 = 0, 0, 0, 0

    for _, top in tops.iterrows():
        pred = market.get_data(start_date=top['DATE'], code=top['CODE'])

        if config.speed_method in ['rm_market_vr_fft']:
            pred_market_ratios1 = market.get_span_market_ratio(pred, 1)
            pred_market_ratios5 = market.get_span_market_ratio(pred, 5)
            pred_market_ratios10 = market.get_span_market_ratio(pred, 10)
            pred_market_ratios20 = market.get_span_market_ratio(pred, 20)
        else:
            pred_market_ratios1, pred_market_ratios5, pred_market_ratios10, pred_market_ratios20 = 0, 0, 0, 0

        pred_ratios1 += (pred.iloc[1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios1
        pred_ratios5 += (pred.iloc[5]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios5
        pred_ratios10 += (pred.iloc[10]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios10
        pred_ratios20 += (pred.iloc[20]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios20

    size = tops.shape[0]
    return [code, pred_ratios1/size, pred_ratios5/size, pred_ratios10/size, pred_ratios20/size]


if __name__ == '__main__':

    all_data, pattern, targets = market.get_historical_data(config.start_date, code=config.code)

    tops = find_similar_of_a_stock(pattern, targets, config.code)
    tops = tops.head(config.nb_similar)
    plot_simi_stock(tops, all_data, pattern, 'speed_search_' + config.speed_method, codes=config.code)