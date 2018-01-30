from codes.config import *
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
from codes.base import plot_simi_stock, norm

all_data = None

def part_search(pattern, targets, col='CLOSE'):

    # 舍弃0值
    # std_data = std_data[(std_data['CLOSE'] != 0)].any().reset_index()
    if speed_method == 'fft_euclidean' or speed_method == 'wave_fft_euclidean' or speed_method == 'fft':
        targets['fft_deg'] = 0
        for i in range(1, 4):
            index = str(i)
            p_fft = pattern['fft' + index].tail(1).values
            p_deg = pattern['deg' + index].tail(1).values

            targets['fft' + index] = (targets['fft' + index] - p_fft).abs()
            targets['deg' + index] = (targets['deg' + index] - p_deg).abs() * 100

            targets['fft_deg'] += targets['fft' + index] + targets['deg' + index]

        sorted_std_diff = targets.sort_values(ascending=True, by=['fft_deg'])
        sorted_std_diff = sorted_std_diff.head(200)

    else:
        std = pattern['std'].tail(1).values
        targets['std'] = (targets['std'] - std).abs()
        sorted_std_diff = targets.sort_values(ascending=True, by=['std']).head(500)

    sorted_std_diff[similarity_method] = -1

    for i in sorted_std_diff.index.values:
        ith = sorted_std_diff.loc[i]
        result = targets[(targets['CODE'] == ith['CODE']) & (targets['DATE'] <= ith['DATE'])].tail(pattern_length)
        # sorted_std_diff.loc[i, similarity_method] = pearsonr(result['CLOSE'], pattern['CLOSE'])[0]
        sorted_std_diff.loc[i, similarity_method] = distance.euclidean(norm(result[col]), norm(pattern[col]))

    if speed_method == 'fft':
        tops = sorted_std_diff.head(2)
    else:
        tops = sorted_std_diff.sort_values(ascending=True, by=[similarity_method]).head(2)

    return tops

def load_data(start_date=start_date):
    col = 'CLOSE'
    if speed_method == 'fft_euclidean' or speed_method == 'fft':
        file = ZZ800_FFT_DATA
    elif speed_method == 'wave_fft_euclidean':
        file = ZZ800_WAVE_FFT_DATA
        col = 'denoise_CLOSE'
    else:
        file = ZZ800_STD_DATA
    global all_data
    if all_data is None:
        all_data = pd.read_csv(file, parse_dates=['DATE'])

    targets = all_data[all_data['CODE'] != code].reset_index(drop=True)
    pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] >= start_date)].head(pattern_length)

    return all_data, pattern, targets, col

if __name__ == '__main__':

    time_start = time.time()

    all_data, pattern, targets, col = load_data(start_date)
    tops = part_search(pattern, targets, col)

    time_end = time.time()
    print('Part Time is:', time_end - time_start)

    plot_simi_stock(tops, std_data, pattern, 'part_simi_search')