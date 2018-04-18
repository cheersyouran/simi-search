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
    # config.plot_simi_stock = True
    config.nb_similar_of_each_stock = 100
    config.nb_similar_make_prediction = 5
    config.nb_similar_of_all_similar = 15
    config.cores = 4

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from multiprocessing import Pool
from collections import OrderedDict
from codes.speed_search import predict_stock_base_on_similars, find_similar_of_a_stock
from codes.market import market
from codes.base import plot_nav_curve, norm
from scipy.stats.stats import pearsonr

def make_index_prediction():

    pool = Pool(processes=config.cores)
    all_stocks_avg_pred_results = pool.map(predict_stock_base_on_similars, market.codes)
    pool.close()

    pred1, pred5, pred10, pred20 = 0, 0, 0, 0
    for result in all_stocks_avg_pred_results:
        if result is None:
            continue
        pred1 += result[1] / len(all_stocks_avg_pred_results)
        pred5 += result[2] / len(all_stocks_avg_pred_results)
        pred10 += result[3] / len(all_stocks_avg_pred_results)
        pred20 += result[4] / len(all_stocks_avg_pred_results)

    m = market.get_data(start_date=market.current_date)
    act1 = (m['800_MARKET'].iloc[1] - m['800_MARKET'].iloc[0]) / m['800_MARKET'].iloc[0]
    act5 = (m['800_MARKET'].iloc[5] - m['800_MARKET'].iloc[0]) / m['800_MARKET'].iloc[0]
    act10 = (m['800_MARKET'].iloc[10] - m['800_MARKET'].iloc[0]) / m['800_MARKET'].iloc[0]
    act20 = (m['800_MARKET'].iloc[20] - m['800_MARKET'].iloc[0]) / m['800_MARKET'].iloc[0]

    if pred5 > 0:
        action = 1
    else:
        action = -1

    return action, pred5, act5, act5

# 汇总800*20支相似股票
def make_prediction():

    pool = Pool(processes=config.cores)
    all_stocks_avg_pred_results = pool.map(predict_stock_base_on_similars, market.codes)
    pool.close()
    all_stocks_avg_pred_results = [x for x in all_stocks_avg_pred_results if x is not None]

    pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20 = [], [], [], []
    act_ratios1, act_ratios5, act_ratios10, act_ratios20, codes = [], [], [], [], []

    for avg_pred_result in all_stocks_avg_pred_results:
        codes.append(avg_pred_result[0])

        pred_ratios1.append(avg_pred_result[1])
        pred_ratios5.append(avg_pred_result[2])
        pred_ratios10.append(avg_pred_result[3])
        pred_ratios20.append(avg_pred_result[4])

        act = market.get_data(start_date=market.current_date, code=avg_pred_result[0])

        if config.speed_method in ['rm_market_vr_fft']:
            act_market_ratios1 = market.get_span_market_ratio(act, 1)
            act_market_ratios5 = market.get_span_market_ratio(act, 5)
            act_market_ratios10 = market.get_span_market_ratio(act, 10)
            act_market_ratios20 = market.get_span_market_ratio(act, 20)
        else:
            act_market_ratios1, act_market_ratios5, act_market_ratios10, act_market_ratios20 = 0, 0, 0, 0

        act_ratios1.append((act.iloc[1]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios1)
        act_ratios5.append((act.iloc[5]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios5)
        act_ratios10.append((act.iloc[10]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios10)
        act_ratios20.append((act.iloc[20]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios20)

    action, pred_ratio, act_ratio, market_ratio = \
        get_action_and_calcu_corr(codes, pred_ratios1, pred_ratios5,pred_ratios10, pred_ratios20,
                                  act_ratios1, act_ratios5, act_ratios10,act_ratios20)

    return action, pred_ratio, act_ratio, market_ratio

# 先对每支股票的找200支相似，然后汇总800*200后取前4000
def make_prediction2():

    pool = Pool(processes=config.cores)
    tops = pool.map(find_similar_of_a_stock, market.codes)
    pool.close()

    tops = [top for top in tops if top is not None]
    tops = pd.concat(tops).sort_values(ascending=True, by=[config.similarity_method])
    tops = tops[tops[config.similarity_method] > 0]
    tops = tops.head(config.nb_similar_of_all_similar)

    def apply(x):
        x_ = x.head(config.nb_similar_make_prediction)
        pattern_code = x_['pattern'].values[0]

        pred_ratio1, pred_ratio5, pred_ratio10, pred_ratio20 = 0, 0, 0, 0
        for index, top in x_.iterrows():
            pred = market.get_data(start_date=top['DATE'], code=top['CODE'])

            if pred.shape[0] < 30:
                print(pattern_code)
                print(pred.values)

            if config.speed_method in ['rm_market_vr_fft']:
                pred_market_ratios1 = market.get_span_market_ratio(pred, 1)
                pred_market_ratios5 = market.get_span_market_ratio(pred, 5)
                pred_market_ratios10 = market.get_span_market_ratio(pred, 10)
                pred_market_ratios20 = market.get_span_market_ratio(pred, 20)
            else:
                pred_market_ratios1, pred_market_ratios5, pred_market_ratios10, pred_market_ratios20 = 0, 0, 0, 0

            pred_ratio1 += (pred.iloc[1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios1
            pred_ratio5 += (pred.iloc[5]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios5
            pred_ratio10 += (pred.iloc[10]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios10
            pred_ratio20 += (pred.iloc[20]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'] - pred_market_ratios20

        if ~config.is_regression_test:
        # if pd.to_datetime(config.update_end) < config.start_date:
            print('正在进行实际预测, 无实际值...')
        else:
            act = market.get_data(start_date=market.current_date, code=pattern_code)
            if config.speed_method in ['rm_market_vr_fft']:
                act_market_ratios1 = market.get_span_market_ratio(pred, 1)
                act_market_ratios5 = market.get_span_market_ratio(pred, 5)
                act_market_ratios10 = market.get_span_market_ratio(pred, 10)
                act_market_ratios20 = market.get_span_market_ratio(pred, 20)
            else:
                act_market_ratios1, act_market_ratios5, act_market_ratios10, act_market_ratios20 = 0, 0, 0, 0

            act_ratios1.append((act.iloc[1]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios1)
            act_ratios5.append((act.iloc[5]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios5)
            act_ratios10.append((act.iloc[10]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios10)
            act_ratios20.append((act.iloc[20]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'] - act_market_ratios20)

        size = tops.shape[0]

        pred_ratios1.append(pred_ratio1 / size)
        pred_ratios5.append(pred_ratio5 / size)
        pred_ratios10.append(pred_ratio10 / size)
        pred_ratios20.append(pred_ratio20 / size)

        codes.append(pattern_code)

    act_ratios1, act_ratios5, act_ratios10, act_ratios20 = [], [], [], []
    pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20 = [], [], [], []
    codes = []

    tops.groupby(['pattern']).apply(func=apply)
    print('[Codes left] ', len(codes))

    if ~config.is_regression_test:
        return get_action(codes, pred_ratios1, pred_ratios5,pred_ratios10, pred_ratios20)
    else:
        return get_action_and_calcu_corr(codes, pred_ratios1, pred_ratios5,pred_ratios10, pred_ratios20,
                              act_ratios1, act_ratios5, act_ratios10, act_ratios20)


def get_action_and_calcu_corr(codes, pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20,
                              act_ratios1, act_ratios5, act_ratios10, act_ratios20):

    pred_act_result = pd.DataFrame(
        OrderedDict({'CODE': codes, 'CURRENT_DATE': market.current_date,'PRED1': pred_ratios1,
                     'PRED5': pred_ratios5, 'PRED10': pred_ratios10, 'PRED20': pred_ratios20,
                     'ACT1': act_ratios1, 'ACT5': act_ratios5, 'ACT10': act_ratios10, 'ACT20': act_ratios20}))

    pred_act_result.to_csv(config.PRDT_AND_ACT_RESULT, mode='a', header=False, index=False)

    p1 = pearsonr(pred_ratios1, act_ratios1)[0]
    p2 = pearsonr(pred_ratios5, act_ratios5)[0]
    p3 = pearsonr(pred_ratios10, act_ratios10)[0]
    p4 = pearsonr(pred_ratios20, act_ratios20)[0]

    pearson = pd.DataFrame(
        OrderedDict({'CURRENT_DATE': [market.current_date], 'P1': [p1], 'P2': [p2], 'P3': [p3], 'P4': [p4]}))
    pearson.to_csv(config.PEARSON_CORR_RESLUT, mode='a', header=False, index=False)

    pred_ratio = np.sum(pred_act_result['PRED5']) * (1 / pred_act_result.shape[0])
    act_ratio = np.sum(pred_act_result['ACT5']) * (1 / pred_act_result.shape[0])
    market_ratio = market.get_data(start_date=market.current_date)[config.market_ratio_type].iloc[5]

    market_ratio /= 100

    if pred_ratio > 0:
        action = 1
    else:
        action = -1

    print('[Correlation] ', p1)
    print('[Correlation] ', p2)
    print('[Correlation] ', p3)
    print('[Correlation] ', p4)

    return action, pred_ratio, act_ratio, market_ratio

def get_action(codes, pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20):
    pred_act_result = pd.DataFrame(
        OrderedDict({'CODE': codes, 'CURRENT_DATE': market.current_date, 'PRED1': pred_ratios1,
                     'PRED5': pred_ratios5, 'PRED10': pred_ratios10, 'PRED20': pred_ratios20}))

    pred_act_result = pred_act_result.sort_values(ascending=False, by=['PRED5'])
    pred_act_result.to_csv(config.PRDT_AND_ACT_RESULT, mode='a', header=False, index=False)

    return -1, 0, 0, 0

def regression_test(get_daily_action):

    strategy_net_values = [1.0]
    act_net_values = [1.0]
    market_net_values = [1.0]
    dates = [market.current_date.date()]
    turnover_rate = 0
    last_action = -1
    while config.start_date <= config.regression_end_date:
        time_start = time.time()

        print('\n[Start Date]: ' + str(config.start_date.date()))
        print('[Current Date]: ' + str(market.current_date.date()))

        action, pred_ratio, act_ratio, market_ratio = get_daily_action()

        print('[Predict]:', pred_ratio)
        print('[Actual ]:', act_ratio)
        print('[Market ]:', market_ratio)

        if action == 1:
            print('[Action]: Buy in!')
            strategy_net_values.append(strategy_net_values[-1] * (1 + market_ratio))
        elif action == -1:
            print('[Action]: Keep Empty!')
            strategy_net_values.append(strategy_net_values[-1])
        else:
            raise Exception()

        if last_action != action:
            turnover_rate += 1
        last_action = action

        act_net_values.append(act_net_values[-1] * (1 + act_ratio))
        market_net_values.append(market_net_values[-1] * (1 + market_ratio))

        if config.weekily_regression == False:
            market._pass_a_day()
        else:
            market._pass_a_week()

        dates.append(market.current_date.date())

        plot_nav_curve(strategy_net_values, act_net_values, market_net_values, dates, turnover_rate)
        time_end = time.time()
        print('Search Time:', time_end - time_start)

if __name__ == '__main__':

    time_start = time.time()

    print('\n#####################################')
    print('Cpu Core Num: ', os.cpu_count())
    print('Memory in all :', psutil.virtual_memory().total / 1024 / 1024 / 1024, 'G')
    print('Start Date: ' + str(config.start_date))
    print('Similar NB: ' + str(config.nb_similar_make_prediction))
    print('Market Ind: ' + str(config.market_index))
    print('Speed Meth: ' + str(config.speed_method))
    print('#####################################')

    regression_test(make_prediction2)
    # regression_test(make_prediction)
    # regression_test(make_index_prediction)

    time_end = time.time()
    print('Search Time:', time_end - time_start)