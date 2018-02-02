from codes.config import *
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
from codes.base import plot_simi_stock, norm
from codes.market import get_historical_data

def speed_search(pattern, targets, col='CLOSE'):

    # 舍弃0值
    # std_data = std_data[(std_data['CLOSE'] != 0)].any().reset_index()
    if speed_method in ['fft_euclidean', 'wave_fft_euclidean', 'fft']:
        ALPHA = np.multiply([1, 1, 1], 1)
        BETA = np.multiply([1, 1, 1], 100)
    elif speed_method in ['value_ratio_fft_euclidean']:
        ALPHA = np.multiply([1, 1, 1], 5000)
        BETA = np.multiply([1, 1, 1], 1)

    targets['fft_deg'] = 0
    for i in range(3):
        index = str(i+1)
        p_fft = pattern['fft' + index].tail(1).values
        p_deg = pattern['deg' + index].tail(1).values

        targets['fft_' + index] = (targets['fft' + index] - p_fft).abs() * ALPHA[i]
        targets['deg_' + index] = (targets['deg' + index] - p_deg).abs() * BETA[i]

        targets['fft_deg'] += targets['fft_' + index] + targets['deg_' + index]

    sorted_std_diff = targets.sort_values(ascending=True, by=['fft_deg'])
    sorted_std_diff = sorted_std_diff.head(200)

    sorted_std_diff[similarity_method] = -1

    for i in sorted_std_diff.index.values:
        ith = sorted_std_diff.loc[i]
        result = targets[(targets['CODE'] == ith['CODE']) & (targets['DATE'] <= ith['DATE'])].tail(pattern_length)
        sorted_std_diff.loc[i, similarity_method] = distance.euclidean(norm(result[col]), norm(pattern[col]))
        # sorted_std_diff.loc[i, similarity_method] = pearsonr(result['CLOSE'], pattern['CLOSE'])[0]

    tops = sorted_std_diff.sort_values(ascending=True, by=[similarity_method]).head(nb_similarity)

    print(tops.iloc[:, 8:])
    print(pattern.iloc[-1, 8:])

    return tops


if __name__ == '__main__':

    time_start = time.time()

    all_data, pattern, targets, col = get_historical_data(start_date)
    tops = speed_search(pattern, targets, col)

    time_end = time.time()
    print('Part Time is:', time_end - time_start)

    plot_simi_stock(tops, all_data, pattern, 'speed_search')