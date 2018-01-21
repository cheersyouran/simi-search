from codes.config import *
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
from codes.base import plot_simi_stock, norm
from codes.market import load_all_data

def part_search():
    std_data = pd.read_csv(ZZ800_FFT_DATA, parse_dates=['DATE'])

    pattern = std_data[(std_data['CODE'] == code) & (std_data['DATE'] >= start_date)].head(pattern_length)
    targets = std_data[std_data['CODE'] != code].reset_index(drop=True)

    std = pattern['std'].tail(1).values
    targets['std'] = (targets['std'] - std).abs()
    sorted_std_diff = targets.sort_values(ascending=True, by=['std']).head(800)

    sorted_std_diff[similarity_method] = -1
    for i in sorted_std_diff.index.values:
        ith = sorted_std_diff.loc[i]
        result = targets[(targets['CODE'] == ith['CODE']) & (targets['DATE'] <= ith['DATE'])].tail(pattern_length)
        # sorted_std_diff.loc[i, similarity_method] = pearsonr(result['CLOSE'], pattern['CLOSE'])[0]
        sorted_std_diff.loc[i, similarity_method] = distance.euclidean(result['CLOSE'], pattern['CLOSE'])

    # sorted_std_diff.loc[20887, similarity_method] = 1
    tops = sorted_std_diff.sort_values(ascending=True, by=[similarity_method]).head(2)

    return tops, pattern

def fft_part_search():
    std_data = pd.read_csv(ZZ800_FFT_DATA, parse_dates=['DATE'])

    pattern = std_data[(std_data['CODE'] == code) & (std_data['DATE'] >= start_date)].head(pattern_length)
    targets = std_data[std_data['CODE'] != code].reset_index(drop=True)

    fft = pattern['fft'].tail(1).values
    deg = pattern['deg'].tail(1).values

    targets['fft'] = (targets['fft'] - fft).abs()
    targets['deg'] = (targets['deg'] - deg).abs()
    targets['fft_deg'] = targets['fft'] * 100 * targets['deg']

    sorted_std_diff = targets.sort_values(ascending=True, by=['fft_deg'])
    sorted_std_diff = sorted_std_diff.head(500)

    sorted_std_diff[similarity_method] = -1
    for i in sorted_std_diff.index.values:
        ith = sorted_std_diff.loc[i]
        result = targets[(targets['CODE'] == ith['CODE']) & (targets['DATE'] <= ith['DATE'])].tail(pattern_length)
        # sorted_std_diff.loc[i, similarity_method] = pearsonr(result['CLOSE'], pattern['CLOSE'])[0]
        sorted_std_diff.loc[i, similarity_method] = distance.euclidean(norm(result['CLOSE']), norm(pattern['CLOSE']))

    tops = sorted_std_diff.sort_values(ascending=True, by=[similarity_method]).head(2)

    return tops, pattern


if __name__ == '__main__':

    all_data = load_all_data()

    time_start = time.time()
    tops, pattern = fft_part_search()
    time_end = time.time()

    print('Part Time is:', time_end - time_start)

    plot_simi_stock(tops, all_data, pattern, 'part_simi_search')