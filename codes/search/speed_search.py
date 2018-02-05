from codes.config import *
from scipy.stats.stats import pearsonr
from codes.base import plot_simi_stock, norm, compare_plot, weighted_distance
from codes.market import get_historical_data, pass_a_day, get_data
import time

def result_check(tops):
    top1 = tops.iloc[0]
    if config.speed_method in ['value_ratio_fft_euclidean']:
        length = config.pattern_length + 1
        end = pass_a_day(top1['DATE'])
        start = config.start_date
    else:
        length = config.pattern_length
        end = top1['DATE']
        start = pass_a_day(config.start_date)

    _, pred = get_data(code=top1['CODE'], end_date=end, pattern_length=length)
    income_ratio = (pred.iloc[-1]['CLOSE'] - pred.iloc[-2]['CLOSE']) / pred.iloc[-2]['CLOSE']
    _, act = get_data(code=config.code, start_date=start, pattern_length=length)
    act_income_ratio = (act.iloc[-1]['CLOSE'] - act.iloc[-2]['CLOSE']) / act.iloc[-2]['CLOSE']

    # print(norm(pred['CLOSE'].values))
    # print(norm(act['CLOSE'].values))
    print('Predict:', income_ratio)
    print('Actual:', act_income_ratio)

    compare_plot(norm(pred['CLOSE'].values), norm(act['CLOSE'].values))


def speed_search(pattern, targets, col='CLOSE'):

    if config.speed_method in ['fft_euclidean', 'wave_fft_euclidean', 'fft']:
        # ALPHA = np.multiply([1, 1.2, 1.5, 1.5], 100)
        # BETA = np.multiply([1, 1.2, 1.5, 1.5], 1)
        ALPHA = np.multiply([1, 1, 1, 1, 1], 100)
        BETA = np.multiply([1, 1, 1, 1, 1], 1)
    elif config.speed_method in ['value_ratio_fft_euclidean']:
        # ALPHA = np.multiply([1, 1.2, 1.5, 1.7], 2000)
        # BETA = np.multiply([1, 1, 1, 1], 1)
        ALPHA = np.multiply([1, 1, 1, 1, 1], 100)
        BETA = np.multiply([1, 1, 1, 1, 1], 1)

    targets['fft_deg'] = 0
    for i in range(5):
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

    # print(sorted_std_diff.iloc[:10, 7:])

    tops = sorted_std_diff.sort_values(ascending=True, by=[config.similarity_method]).head(config.nb_similarity)

    print(tops.iloc[:, 7:16])
    print('@@@')
    print(pattern.iloc[-1, 7:16].values)

    print(tops.iloc[:, 16:])

    result_check(tops)

    return tops

if __name__ == '__main__':

    time_start = time.time()

    all_data, pattern, targets, col = get_historical_data(config.start_date)
    tops = speed_search(pattern, targets, col)

    time_end = time.time()
    print('Part Time is:', time_end - time_start)

    plot_simi_stock(tops, all_data, pattern, 'speed_search_' + config.speed_method)

    if config.speed_method not in ['value_ratio_fft_euclidean']:
        config.speed_method = 'changed'
        plot_simi_stock(tops, all_data, pattern, '标准化-盈利率')