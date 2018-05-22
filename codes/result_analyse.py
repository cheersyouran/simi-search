from scipy.stats.stats import pearsonr
import tushare as ts
from codes.config import *

if __name__ == '__main__':
    data = pd.read_csv(config.rootPath + '/result_analyse/pred_2018-05-18.csv')
    data.columns = ['CODE', 'DATE', 'P1', 'P2', 'P3', 'P4', 'P5', 'P10', 'P20']
    data = data.dropna()
    codes = data['CODE'].values

    act = []
    for code in codes:
        p = ts.get_k_data(code.split('.')[0], start='2018-05-17', end='2018-05-18')
        a = p['close'] / p['close'].shift(1) - 1
        act.append(a.values[1])

    data['A1'] = act
    data = data.sort_values(ascending=True, by=['P1'])

    for i in range(5, 30):
        percentage = round(data.shape[0] * 0.01 * i)

        long = data.head(percentage)
        short = data.tail(percentage)

        pos = long['A1'].mean()
        neg = short['A1'].mean()

        print('\npos: ', pos)
        print('neg: ', neg)

    p1 = pearsonr(data['P1'].values, act)[0]
    print(p1)