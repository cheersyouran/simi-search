from scipy.stats.stats import pearsonr
import tushare as ts
from codes.config import *

if __name__ == '__main__':
    col = ['CODE', 'DATE', 'P1', 'P2', 'P3', 'P4']
    data = pd.read_csv(config.rootPath + '/result_analyse/pred2018-05-16_rm_market_vr_fft_20.csv', names=col)
    data = data.dropna()
    codes = data['CODE'].values

    act = []
    for code in codes:
        p = ts.get_k_data(code.split('.')[0], start='2018-05-15', end='2018-05-16')
        a = p['close'] / p['close'].shift(1) - 1
        act.append(a.values[1])

    data['A1'] = act
    data.sort_values(ascending=False, by=['P1'])

    p1 = pearsonr(data['P1'].values, act)[0]

    print(p1)