from codes.config import *
from scipy.stats.stats import pearsonr
from codes.market import market
from codes.base import plot_simi_stock, norm, weighted_distance
import time
import psutil
from memory_profiler import profile

def _speed_search(pattern, targets, code=config.code, col='CLOSE'):

    if config.speed_method in ['fft_euclidean', 'wave_fft_euclidean', 'fft']:
        ALPHA = np.multiply([1, 1, 1, 1, 1], 100)
        BETA = np.multiply([1, 1, 1, 1, 1], 1)
    elif config.speed_method in ['value_ratio_fft_euclidean']:
        ALPHA = np.multiply([1, 1, 1, 1, 1], 100)
        BETA = np.multiply([1, 1, 1, 1, 1], 1)

    targets['fft_deg'] = 0
    for i in range(config.fft_level):
        index = str(i + 1)
        p_fft = pattern['fft' + index].tail(1).values
        p_deg = pattern['deg' + index].tail(1).values

        targets['fft_' + index] = (targets['fft' + index] - p_fft).abs() * ALPHA[i]
        targets['deg_' + index] = (targets['deg' + index] - p_deg).abs() * BETA[i]

        targets['fft_deg'] += targets['fft_' + index] + targets['deg_' + index]

    sorted_std_diff = targets.sort_values(ascending=True, by=['fft_deg'])

    sorted_std_diff = sorted_std_diff[sorted_std_diff['VOLUME'] != '0.0000']
    sorted_std_diff = sorted_std_diff[sorted_std_diff['VOLUME'] != 0]
    sorted_std_diff = sorted_std_diff[sorted_std_diff['VOLUME'] != '0']

    sorted_std_diff = sorted_std_diff.head(200)

    sorted_std_diff[config.similarity_method] = -1

    for i in sorted_std_diff.index.values:
        ith = sorted_std_diff.loc[i]
        result = targets[(targets['CODE'] == ith['CODE']) & (targets['DATE'] <= ith['DATE'])].tail(config.pattern_length)
        sorted_std_diff.loc[i, config.similarity_method] = weighted_distance(norm(result[col]), norm(pattern[col]), config.pattern_length)
        # sorted_std_diff.loc[i, similarity_method] = pearsonr(result['CLOSE'], pattern['CLOSE'])[0]

    tops = sorted_std_diff.sort_values(ascending=True, by=[config.similarity_method]).head(config.nb_similarity)

    return tops

def parallel_speed_search(code):
    all_data, pattern, targets, col = market.get_historical_data(start_date=config.start_date, code=code)
    tops = _speed_search(pattern, targets, code, col)
    tops = tops.head(15)[['CODE', 'DATE', config.similarity_method]]
    tops['pattern'] = code
    # queue.put([tops, pattern, code])

    pred_ratios = 0
    # top 20 similar stocks about this stock
    for _, top in tops.iterrows():
        pred = market.get_data(start_date=top['DATE'], code=top['CODE']).head(6)
        pred_ratios += (pred.iloc[-1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']

    print(os.getpid(), 'Memory in used:', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 'M')

    return [code, pred_ratios/tops.shape[0]]

if __name__ == '__main__':

    time_start = time.time()
    all_data, pattern, targets, col = market.get_historical_data(config.start_date, code=config.code)
    tops = _speed_search(pattern, targets, config.code, col)

    time_end = time.time()
    print('Part Time is:', time_end - time_start)

    plot_simi_stock(tops, all_data, pattern, 'speed_search_' + config.speed_method, codes=config.code)

    if config.speed_method not in ['value_ratio_fft_euclidean']:
        config.speed_method = 'changed'
        plot_simi_stock(tops, all_data, pattern, 'std_nav')