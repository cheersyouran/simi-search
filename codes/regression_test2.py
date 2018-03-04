import sys
import os
import psutil

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import OrderedDict
from codes.config import config
from codes.speed_search import _speed_search
from codes.market import market
from codes.base import plot_nav_curve, norm
from scipy.stats.stats import pearsonr
from codes.regression_test1 import regression_test

def get_daily_action_parallel():

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

if __name__ == '__main__':
    print('Cpu Core Num: ', os.cpu_count())

    config.nb_codes = 3
    regression_test(get_daily_action_parallel)