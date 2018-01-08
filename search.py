from scipy.stats.stats import pearsonr
from datetime import timedelta
from scipy.spatial import distance
import base
from base import norm
import pandas as pd
pd.set_option('display.width', 1000)
import numpy as np


code = '000001.SZ'
time_period = 30
start_date = pd.to_datetime('2017-01-01')
nb_similarity = 2
nb_sample = 50000
ascending_sort = False

def t_rol_aply(t, p):
    result = pearsonr(t, p)[0]

    # result = distance.euclidean(norm(t), norm(p))
    return result

def target_apply(t, p):
    p_close = p['CLOSE']
    t_close = t['CLOSE']
    t['pearson'] = t_close.rolling(window=time_period).apply(func=t_rol_aply, args=(p_close, ))
    result = t.dropna(axis=0, how='any').sort_values(by=['pearson'], ascending=ascending_sort).head(1)
    return result

def load_and_process_data():
    quote = base.load_data()
    quote = quote.head(nb_sample)
    quote = quote.loc[(quote != 0).any(axis=1)]

    pattern = quote[quote['CODE'] == code].reset_index(drop=True)
    target = quote[quote['CODE'] != code].reset_index(drop=True)

    return quote, pattern, target



if __name__ == '__main__':
    quote, pattern, target = load_and_process_data()
    p = pattern[(pattern['DATE'] >= start_date)].head(time_period)

    a = target.groupby(['CODE']).apply(lambda x: x['CODE'].iloc[0])

    result = target.groupby(['CODE']).apply(func=target_apply, p=p)
    sorted_result = result.sort_values(by=['pearson'], ascending=ascending_sort)

    # 选出n项最匹配的
    most = sorted_result.head(nb_similarity)

    # 初始化绘图的数据
    plot_codes = np.append(most['CODE'].values, code)
    plot_dates = np.append(most['DATE'].values, None)
    plot_prices = np.zeros([nb_similarity + 1, time_period])
    plot_legend = list()

    # 构造相似数据
    for i in range(nb_similarity):
        plot_quote = quote[quote['CODE'] == plot_codes[i]]
        plot_quote = plot_quote[plot_quote['DATE'] <= pd.to_datetime(plot_dates[i])].tail(time_period)
        plot_prices[i] = plot_quote['CLOSE'].values
        plot_legend.append(str(plot_codes[i]) + ":" + str(most[most['CODE'] == plot_codes[i]]['pearson'].values))

    # 构造原数据
    plot_prices[-1] = p['CLOSE'].values
    plot_dates[-1] = p['DATE'].iloc[-1]
    plot_legend.append(str(plot_codes[-1]))

    for i in range(nb_similarity):
        print(t_rol_aply(plot_prices[i], plot_prices[-1]))
        assert t_rol_aply(plot_prices[i], plot_prices[-1]) == most[most['CODE'] == plot_codes[i]]['pearson'].values, 'calcu error!'

    # 调用绘图函数
    base.plot_stocks_price_plot(plot_codes, plot_prices, plot_legend)

    print('Finish')
