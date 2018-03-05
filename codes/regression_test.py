import sys
import os
import psutil

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

from codes.config import config
if 'Youran' in config.rootPath:
    config.nb_codes = 3
    config.plot_simi_stock = False
    config.nb_similar_of_each_stock = 200
    config.nb_similar = 5

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import OrderedDict
from codes.speed_search import parallel_speed_search, _speed_search
from codes.market import market
from codes.base import plot_nav_curve, norm
from scipy.stats.stats import pearsonr

def get_daily_action_parallel():

    pool = Pool(processes=os.cpu_count())

    avg_results = pool.map(parallel_speed_search, market.codes)

    pool.close()

    pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20 = [], [], [], []
    act_ratios1, act_ratios2, act_ratios3, act_ratios4, codes = [], [], [], [], []
    for avg_result in avg_results:

        codes.append(avg_result[0])
        pred_ratios1.append(avg_result[1])
        pred_ratios5.append(avg_result[2])
        pred_ratios10.append(avg_result[3])
        pred_ratios20.append(avg_result[4])

        act = market.get_data(start_date=market.current_date, code=avg_result[0])
        act1 = act.head(2)
        act_ratios1.append((act1.iloc[-1]['CLOSE'] - act1.iloc[0]['CLOSE']) / act1.iloc[0]['CLOSE'])

        act2 = act.head(6)
        act_ratios2.append((act2.iloc[-1]['CLOSE'] - act2.iloc[0]['CLOSE']) / act2.iloc[0]['CLOSE'])

        act3 = act.head(11)
        act_ratios3.append((act3.iloc[-1]['CLOSE'] - act3.iloc[0]['CLOSE']) / act3.iloc[0]['CLOSE'])

        act4 = act.head(21)
        act_ratios4.append((act4.iloc[-1]['CLOSE'] - act4.iloc[0]['CLOSE']) / act4.iloc[0]['CLOSE'])

    pred_act_result = pd.DataFrame(OrderedDict({'CODE': codes, 'CURRENT_DATE': market.current_date,
                                     'PRED1': pred_ratios1, 'PRED5': pred_ratios5,
                                     'PRED10': pred_ratios10, 'PRED20': pred_ratios20,
                                     'ACT1': act_ratios1, 'ACT5': act_ratios2,
                                     'ACT10': act_ratios3, 'ACT20': act_ratios4}))

    pred_act_result.to_csv(config.PRDT_AND_ACT_RESULT, mode='a', header=False, index=False)

    p1 = pearsonr(pred_ratios1, act_ratios1)[0]
    p2 = pearsonr(pred_ratios5, act_ratios2)[0]
    p3 = pearsonr(pred_ratios10, act_ratios3)[0]
    p4 = pearsonr(pred_ratios20, act_ratios4)[0]

    pearson = pd.DataFrame(OrderedDict({'CURRENT_DATE': [market.current_date], 'P1': [p1], 'P2': [p2], 'P3': [p3], 'P4': [p4]}))
    pearson.to_csv(config.PEARSON_CORR_RESLUT, mode='a', header=False, index=False)

    if config.weekily_regression:
        pred_ratio = np.sum(pred_act_result['PRED5']) * (1 / pred_act_result.shape[0])
        act_ratio = np.sum(pred_act_result['ACT5']) * (1 / pred_act_result.shape[0])
    else:
        pred_ratio = np.sum(pred_act_result['PRED1']) * (1 / pred_act_result.shape[0])
        act_ratio = np.sum(pred_act_result['ACT1']) * (1 / pred_act_result.shape[0])

    market_ratio = float(market.ratios[market.ratios['DATE'] == market.pass_days(market.current_date, 1)][config.market_ratio_type]) / 100

    if pred_ratio > 0:
        action = 1
    else:
        action = -1

    print('[Correlation] ', p1)
    print('[Correlation] ', p2)
    print('[Correlation] ', p3)
    print('[Correlation] ', p4)

    print('[Predict]:', pred_ratio)
    print('[Actual ]:', act_ratio)
    print('[Market ]:', market_ratio)

    return action, pred_ratio, act_ratio, market_ratio

def get_daily_action_parallel_rm_vr():

    pool = Pool(processes=os.cpu_count())

    tops = pool.map(_speed_search, market.codes)

    pool.close()

    tops = pd.concat(tops).sort_values(ascending=True, by=[config.similarity_method])
    tops = tops.head(100)

    def apply(x):
        pred_ratio1, pred_ratio5, pred_ratio10, pred_ratio20 = 0, 0, 0, 0
        for _, top in x.iterrows():
            pred = market.get_data(start_date=top['DATE'], code=top['CODE'])
            pred1 = pred.head(2)
            pred_ratio1 += (pred1.iloc[-1]['CLOSE'] - pred1.iloc[0]['CLOSE']) / pred1.iloc[0]['CLOSE']

            pred2 = pred.head(6)
            pred_ratio5 += (pred2.iloc[-1]['CLOSE'] - pred2.iloc[0]['CLOSE']) / pred2.iloc[0]['CLOSE']

            pred3 = pred.head(11)
            pred_ratio10 += (pred3.iloc[-1]['CLOSE'] - pred3.iloc[0]['CLOSE']) / pred3.iloc[0]['CLOSE']

            pred4 = pred.head(21)
            pred_ratio20 += (pred4.iloc[-1]['CLOSE'] - pred4.iloc[0]['CLOSE']) / pred4.iloc[0]['CLOSE']

        size = tops.shape[0]
        return [pred_ratio1/size, pred_ratio5/size, pred_ratio10/size, pred_ratio20/size]

    result = tops.groupby(['pattern']).apply(func=apply)

    act_ratios1, act_ratios2, act_ratios3, act_ratios4 = [], [], [], []
    pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20 = [], [], [], []
    codes = []

    for i, avg_result in result.iteritems():

        act = market.get_data(start_date=market.current_date, code=i)
        act1 = act.head(2)
        act_ratios1.append((act1.iloc[-1]['CLOSE'] - act1.iloc[0]['CLOSE']) / act1.iloc[0]['CLOSE'])

        act2 = act.head(6)
        act_ratios2.append((act2.iloc[-1]['CLOSE'] - act2.iloc[0]['CLOSE']) / act2.iloc[0]['CLOSE'])

        act3 = act.head(11)
        act_ratios3.append((act3.iloc[-1]['CLOSE'] - act3.iloc[0]['CLOSE']) / act3.iloc[0]['CLOSE'])

        act4 = act.head(21)
        act_ratios4.append((act4.iloc[-1]['CLOSE'] - act4.iloc[0]['CLOSE']) / act4.iloc[0]['CLOSE'])

        pred_ratios1.append(avg_result[0])
        pred_ratios5.append(avg_result[1])
        pred_ratios10.append(avg_result[2])
        pred_ratios20.append(avg_result[3])
        codes.append(i)

    pred_act_result = pd.DataFrame(OrderedDict({'CODE': codes, 'CURRENT_DATE': market.current_date,
                                     'PRED1': pred_ratios1, 'PRED5': pred_ratios5,
                                     'PRED10': pred_ratios10, 'PRED20': pred_ratios20,
                                     'ACT1': act_ratios1, 'ACT5': act_ratios2,
                                     'ACT10': act_ratios3, 'ACT20': act_ratios4}))

    pred_act_result.to_csv(config.PRDT_AND_ACT_RESULT, mode='a', header=False, index=False)

    p1 = pearsonr(pred_ratios1, act_ratios1)[0]
    p2 = pearsonr(pred_ratios5, act_ratios2)[0]
    p3 = pearsonr(pred_ratios10, act_ratios3)[0]
    p4 = pearsonr(pred_ratios20, act_ratios4)[0]

    pearson = pd.DataFrame(OrderedDict({'CURRENT_DATE': [market.current_date], 'P1': [p1], 'P2': [p2], 'P3': [p3], 'P4': [p4]}))
    pearson.to_csv(config.PEARSON_CORR_RESLUT, mode='a', header=False, index=False)

    if config.weekily_regression:
        pred_ratio = np.sum(pred_act_result['PRED5']) * (1 / pred_act_result.shape[0])
        act_ratio = np.sum(pred_act_result['ACT5']) * (1 / pred_act_result.shape[0])
    else:
        pred_ratio = np.sum(pred_act_result['PRED1']) * (1 / pred_act_result.shape[0])
        act_ratio = np.sum(pred_act_result['ACT1']) * (1 / pred_act_result.shape[0])

    market_ratio = float(market.ratios[market.ratios['DATE'] == market.pass_days(market.current_date, 1)][config.market_ratio_type]) / 100

    if pred_ratio > 0:
        action = 1
    else:
        action = -1

    print('[Correlation] ', p1)
    print('[Correlation] ', p2)
    print('[Correlation] ', p3)
    print('[Correlation] ', p4)

    print('[Predict]:', pred_ratio)
    print('[Actual ]:', act_ratio)
    print('[Market ]:', market_ratio)

    return action, pred_ratio, act_ratio, market_ratio

def regression_test(get_daily_action):

    strategy_net_values = [1.0]
    act_net_values = [1.0]
    market_net_values = [1.0]
    dates = [market.current_date.date()]
    turnover_rate = 0
    last_action = 0
    while config.start_date <= config.regression_end_date:

        print('\n[Start Date]: ' + str(config.start_date.date()))
        print('[Current Date]: ' + str(market.current_date.date()))

        action, pred_ratio, act_ratio, market_ratios = get_daily_action()

        if action == 1:
            print('[Action]: Buy in!')
            strategy_net_values.append(strategy_net_values[-1] * (1 + market_ratios))
            if last_action == 0:
                turnover_rate += 1
            last_action = action
        elif action == -1:
            print('[Action]: Keep Empty!')
            strategy_net_values.append(strategy_net_values[-1])
            if last_action == -1:
                turnover_rate += 1
            last_action = action
        else:
            raise Exception()

        act_net_values.append(act_net_values[-1] * (1 + act_ratio))
        market_net_values.append(market_net_values[-1] * (1 + market_ratios))

        if config.weekily_regression == False:
            market._pass_a_day()
        else:
            market._pass_a_week()

        dates.append(market.current_date.date())
        plot_nav_curve(strategy_net_values, act_net_values, market_net_values, dates, turnover_rate)

def result_check(tops, name, pred_ratio, act_ratio):
    def compare_plot(x1, x2, name):
        plt.plot(x1)
        plt.plot(x2)
        plt.grid(True)
        plt.ioff()
        plt.savefig('../pic/' + name + '.jpg')
        plt.close()

    top1 = tops.iloc[0]
    length = config.pattern_length + 1

    pred = market.get_data(code=top1['CODE'], end_date=market.pass_days(top1['DATE'], 1), pattern_length=length)
    act = market.get_data(code=config.code, start_date=config.start_date, pattern_length=length)

    pred_ratio1 = (pred.iloc[-1]['CLOSE'] - pred.iloc[-2]['CLOSE']) / pred.iloc[-2]['CLOSE']
    act_ratio1 = (act.iloc[-1]['CLOSE'] - act.iloc[-2]['CLOSE']) / act.iloc[-2]['CLOSE']

    assert pred_ratio1 == pred_ratio, 'calcu error!'
    assert act_ratio1 == act_ratio, 'calcu error!'

    compare_plot(norm(pred['CLOSE'].values), norm(act['CLOSE'].values), name)

if __name__ == '__main__':
    time_start = time.time()

    print('\n#####################################')
    print('Cpu Core Num: ', os.cpu_count())
    print('Memory in all :', psutil.virtual_memory().total / 1024 / 1024 / 1024, 'G')
    print('Start Date: ' + str(config.start_date))
    print('Similar NB: ' + str(config.nb_similar))
    print('Market Ind: ' + str(config.market_index))
    print('Speed Meth: ' + str(config.speed_method))
    print('#####################################')

    if config.speed_method == 'rm_vrfft_euclidean':
        regression_test(get_daily_action_parallel_rm_vr)
    else:
        regression_test(get_daily_action_parallel)

    time_end = time.time()
    print('Search Time:', time_end - time_start)