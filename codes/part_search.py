from codes.config import *
from scipy.stats.stats import pearsonr
from scipy.spatial import distance
from codes.base import plot_simi_stock, norm
from codes.market import load_all_data

def part_search(speed_method='fft', start_date=start_date):

    file = ZZ800_FFT_DATA if speed_method == 'fft' else ZZ800_STD_DATA
    std_data = pd.read_csv(file, parse_dates=['DATE'])
    # 舍弃0值
    # std_data = std_data[(std_data['CLOSE'] != 0)].any().reset_index()

    pattern = std_data[(std_data['CODE'] == code) & (std_data['DATE'] >= start_date)].head(pattern_length)
    targets = std_data[std_data['CODE'] != code].reset_index(drop=True)

    if speed_method == 'fft':
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
        if result.shape[0] == 28:
            print("")
        sorted_std_diff.loc[i, similarity_method] = distance.euclidean(norm(result['CLOSE']), norm(pattern['CLOSE']))

    tops = sorted_std_diff.sort_values(ascending=True, by=[similarity_method]).head(2)

    return tops, pattern, targets, std_data

if __name__ == '__main__':

    all_data = load_all_data()

    time_start = time.time()
    tops, pattern, targets, std_data = part_search('nav_std')
    time_end = time.time()

    print('Part Time is:', time_end - time_start)

    plot_simi_stock(tops, all_data, pattern, 'part_simi_search')