import sys
import os
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

from codes.config import config
if 'Youran/Projects/' in config.rootPath:
    print('Using Test Config!')
    config.nb_codes = 20
    config.cores = 4
    config.plot_simi_stock = False
    # config.nb_similar_of_each_stock = 100
    # config.nb_similar_make_prediction = 5
    # config.nb_similar_of_all_similar = 15
    # config.slide_window = 100

import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from multiprocessing import Pool
from collections import OrderedDict
from codes.search import predict_stock_base_on_similars, find_similar_of_a_stock
from codes.market import market
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

    preds = [pred_ratios1, pred_ratios5, pred_ratios10, pred_ratios20]
    acts = [act_ratios1, act_ratios5, act_ratios10, act_ratios20]

    get_prediction_and_calcu_corr(codes, preds, acts)


# 先对每支股票的找200支相似，然后汇总800*200后取前4000
def make_prediction2():

    time_start = time.time()
    pool = Pool(processes=config.cores)
    tops = []
    for codes in np.split(market.codes, min(20, os.cpu_count()), axis=0):
        result = pool.map(find_similar_of_a_stock, codes)
        tops = tops + result

    pool.close()
    time_end = time.time()
    print('Search Time:', time_end - time_start)

    print("---tops:", len(tops))
    tops = [top for top in tops if top is not None and top.shape[0] != 0]
    print("---tops:", len(tops))

    tops = pd.concat(tops).sort_values(ascending=True, by=[config.similarity_method])
    tops.to_csv(config.rootPath + '/output/' + str(market.current_date.date()) + '_16K_similars.csv', index=False)

    tops = tops[tops[config.similarity_method] >= 0]
    tops = tops.head(config.nb_similar_of_all_similar)

    tops.to_csv(config.rootPath + '/output/' + str(market.current_date.date()) + '_4K_similars.csv', index=False)

    print("---tops:", len(tops))

    def apply(x):
        x_ = x.head(config.nb_similar_make_prediction).copy()
        pattern_code = x_['pattern'].values[0]

        pred_ratio1, pred_ratio2, pred_ratio3, pred_ratio4, pred_ratio5, pred_ratio10, pred_ratio20 = 0, 0, 0, 0, 0, 0, 0
        size = 0
        for index, top in x_.iterrows():
            pred = market.get_data(start_date=top['DATE'], code=top['CODE'])

            if pred.shape[0] != 30:
                print('---Cannot found 30 ', top['CODE'], ' for ', pattern_code, ' start form ', str(top['DATE']))
                continue
            size += 1
            pred_ratio1 += market.get_span_ret(pred, 1)/market.get_span_market_ratio(pred, 1) - 1
            pred_ratio2 += market.get_span_ret(pred, 2)/market.get_span_market_ratio(pred, 2) - 1
            pred_ratio3 += market.get_span_ret(pred, 3)/market.get_span_market_ratio(pred, 3) - 1
            pred_ratio4 += market.get_span_ret(pred, 4)/market.get_span_market_ratio(pred, 4) - 1
            pred_ratio5 += market.get_span_ret(pred, 5)/market.get_span_market_ratio(pred, 5) - 1
            pred_ratio10 += market.get_span_ret(pred, 10)/market.get_span_market_ratio(pred, 10) - 1
            pred_ratio20 += market.get_span_ret(pred, 20)/market.get_span_market_ratio(pred, 20) - 1

        if config.is_regression_test:
            act = market.get_data(start_date=market.current_date, code=pattern_code)
            act_ratios1.append(market.get_span_ret(act, 1)/market.get_span_market_ratio(act, 1) - 1)
            act_ratios2.append(market.get_span_ret(act, 2)/market.get_span_market_ratio(act, 2) - 1)
            act_ratios3.append(market.get_span_ret(act, 3)/market.get_span_market_ratio(act, 3) - 1)
            act_ratios4.append(market.get_span_ret(act, 4)/market.get_span_market_ratio(act, 4) - 1)
            act_ratios5.append(market.get_span_ret(act, 5)/market.get_span_market_ratio(act, 5) - 1)
            act_ratios10.append(market.get_span_ret(act, 10)/market.get_span_market_ratio(act, 10) - 1)
            act_ratios20.append(market.get_span_ret(act, 20)/market.get_span_market_ratio(act, 20) - 1)
        # else:
        #     print('正在进行实际预测, 无实际值...', pattern_code)

        pred_ratios1.append(pred_ratio1 / size)
        pred_ratios2.append(pred_ratio2 / size)
        pred_ratios3.append(pred_ratio3 / size)
        pred_ratios4.append(pred_ratio4 / size)
        pred_ratios5.append(pred_ratio5 / size)
        pred_ratios10.append(pred_ratio10 / size)
        pred_ratios20.append(pred_ratio20 / size)

        codes.append(pattern_code)

    act_ratios1, act_ratios2, act_ratios3, act_ratios4, act_ratios5, act_ratios10, act_ratios20 = [], [], [], [], [], [], []
    pred_ratios1, pred_ratios2, pred_ratios3, pred_ratios4, pred_ratios5, pred_ratios10, pred_ratios20 = [], [], [], [], [], [], []
    codes = []

    tops.groupby(['pattern']).apply(func=apply)
    print('[Codes left] ', len(codes))

    preds = [pred_ratios1, pred_ratios2, pred_ratios3, pred_ratios4, pred_ratios5, pred_ratios10, pred_ratios20]
    acts = [act_ratios1, act_ratios2, act_ratios3, act_ratios4, act_ratios5, act_ratios10, act_ratios20]

    return get_prediction_and_calcu_corr(codes, preds, acts) if config.is_regression_test else get_prediction(codes, preds)


def get_prediction_and_calcu_corr(codes, pred, act):

    pred_act_result = pd.DataFrame(
        OrderedDict({'CODE': codes, 'CURRENT_DATE': market.current_date, 'PRED1': pred[0],
                     'PRED2': pred[1], 'PRED3': pred[2], 'PRED4': pred[3],
                     'PRED5': pred[4], 'PRED10': pred[5], 'PRED20': pred[6],
                     'ACT1': act[0], 'ACT2': act[1], 'ACT3': act[2], 'ACT4': act[3],
                     'ACT5': act[4], 'ACT10': act[5], 'ACT20': act[6]}))
    pred_act_result = pred_act_result.dropna()
    path = config.rootPath + '/output/pred_' + str(market.current_date.date()) + '.csv'
    pred_act_result.to_csv(path, index=False)

    p1 = pearsonr(pred[0], act[0])[0]
    p2 = pearsonr(pred[1], act[1])[0]
    p3 = pearsonr(pred[2], act[2])[0]
    p4 = pearsonr(pred[3], act[3])[0]
    p5 = pearsonr(pred[4], act[4])[0]
    p10 = pearsonr(pred[5], act[5])[0]
    p20 = pearsonr(pred[6], act[6])[0]

    pearson = pd.DataFrame(
        OrderedDict({'CURRENT_DATE': [market.current_date],
                     'P1': [p1], 'P2': [p2], 'P3': [p3], 'P4': [p4],
                     'P4': [p5], 'P10': [p10], 'P20': [p20]}))
    pearson.to_csv(config.PEARSON_CORR_RESLUT, mode='a', header=False, index=False)

    print('[Correlation1] ', p1)
    print('[Correlation2] ', p2)
    print('[Correlation3] ', p3)
    print('[Correlation4] ', p4)
    print('[Correlation5] ', p5)
    print('[Correlation10] ', p10)
    print('[Correlation20] ', p20)


def get_prediction(codes, pred):
    pred_result = pd.DataFrame(OrderedDict({'CODE': codes, 'CURRENT_DATE': market.current_date,
                                            'PRED1': pred[0], 'PRED2': pred[1], 'PRED3': pred[2],
                                            'PRED4': pred[3], 'PRED5': pred[4], 'PRED10': pred[5],
                                            'PRED20': pred[6]}))

    pred_result = pred_result.dropna()
    pred_result = pred_result.sort_values(ascending=False, by=['PRED5'])

    path = config.rootPath + '/output/pred_' + str(market.current_date.date()) + '.csv'
    pred_result.to_csv(path, index=False)


if __name__ == '__main__':

    time_start = time.time()

    print('\n#####################################')
    print('Cpu Core Num: ', os.cpu_count())
    print('Cpu Core Ocp: ', config.cores)
    print('Start Date: ' + str(config.start_date))
    print('Codes NB: ' + str(config.nb_codes))
    print('Similar NB: ' + str(config.nb_similar_make_prediction))
    print('#####################################')

    while config.start_date <= config.regression_end_date:
        print('\n[Current Date]: ' + str(market.current_date.date()))
        make_prediction2()
        # make_prediction()
        # make_index_prediction()

        if config.is_regression_test:
            market._pass_a_week()
        else:
            market._pass_a_day()

    time_end = time.time()
    print('Total Search Time:', time_end - time_start)