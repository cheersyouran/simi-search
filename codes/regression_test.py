import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from codes.config import config
from codes.speed_search import speed_search
from codes.market import market
from codes.base import plot_nav_curve, norm

def map_search(code):
    all_data, pattern, targets, col = market.get_historical_data(start_date=config.start_date, code=code)
    tops = speed_search(pattern, targets, code, col)
    top1 = tops.head(1)[['CODE', 'DATE', config.similarity_method]].values.flatten()
    top1 = np.hstack((top1, code))
    # queue.put([tops, pattern, code])
    return top1

def get_daily_action_serial():
    all_data, pattern, target, col = market.get_historical_data(start_date=config.start_date)

    if config.speed_method is None:
        from codes.all_search import all_search
        tops = all_search(pattern, target, config.nb_similarity)
    else:
        from codes.speed_search import speed_search
        tops = speed_search(pattern, target, col)

    top1 = tops.iloc[0]

    # 预测值
    pred = market.get_data(start_date=top1['DATE'], code=top1['CODE']).head(2)
    pred_ratio = (pred.iloc[1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']

    # 实际值
    act = market.get_data(start_date=market.current_date, code=config.code).head(2)
    act_ratio = (act.iloc[1]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE']

    # 大盘实际值
    # act_ratio = float(market.ratios[market.ratios['DATE'] == pattern['DATE'].tail(1).values[0]]['ratio']) / 100
    result_check(tops, 'speed_' + config.code + '_' + str(market.current_date.date()), pred_ratio, act_ratio)
    # 根据income_ratio来做决策
    if pred_ratio > 0:
        action = 1
    elif pred_ratio < 0:
        action = -1
    else:
        action = 0

    print('[Predict]:', pred_ratio)
    print('[Actual ]:', act_ratio)
    return action, pred_ratio, act_ratio

def get_daily_action_parallel():
    # 找到所有序列 各自最相似的
    pool = Pool(processes=os.cpu_count())
    top1s = pool.map(map_search, market.codes)
    top1s = pd.DataFrame(columns=['CODE', 'DATE', config.similarity_method, 'ORG_CODE'], data=top1s)

    # 计算预测盈利和实际盈利
    pred_ratios = []
    act_ratios = []
    for _, top1 in top1s.iterrows():
        # tops, pattern, code = queue.get()
        # plot_simi_stock(tops, market.all_data, pattern, 'speed_' + config.speed_method + '_' + code, codes=code)
        pred = market.get_data(start_date=top1['DATE'], code=top1['CODE']).head(2)
        pred_ratios.append((pred.iloc[1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE'])

        act = market.get_data(start_date=market.current_date, code=config.code).head(2)
        act_ratios.append((act.iloc[1]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE'])

    top1s['pred_ratio'] = pred_ratios
    top1s['act_ratios'] = act_ratios

    # top1s = top1s[top1s['pred_ratio'] > 0]
    # top1s_df.sort_values(ascending=False, by=['pred_ratio']).head(config.nb_to_make_action)

    pred_ratio = -1 if top1s.shape[0] == 0 else np.sum(top1s['pred_ratio']) * (1 / top1s.shape[0])
    act_ratio = -1 if top1s.shape[0] == 0 else np.sum(top1s['act_ratios']) * (1 / top1s.shape[0])
    # act_ratio = float(market.ratios[market.ratios['DATE'] == pattern['DATE'].tail(1).values[0]]['ratio']) / 100

    if pred_ratio > 0:
        action = 1
    else:
        action = -1

    print('[Predict]:', pred_ratio)
    print('[Actual ]:', act_ratio)

    return action, pred_ratio, act_ratio

def regression_test(func, name):

    strategy_net_values = [1.0]
    act_net_values = [1.0]
    dates = [market.current_date.date()]
    while config.start_date <= config.end_date:

        print('\n[Current Date]: ' + str(market.current_date.date()))

        action, pred_ratio, act_ratio = func()

        if action == 1:
            print('[Action]: Buy in!')
            strategy_net_values.append(strategy_net_values[-1] * (1 + act_ratio))
        else:
            print('[Action]: Keep Empty')
            strategy_net_values.append(strategy_net_values[-1])

        act_net_values.append(act_net_values[-1] * (1 + act_ratio))

        market._pass_a_day()
        dates.append(market.current_date.date())
        plot_nav_curve(strategy_net_values, act_net_values, dates, name)

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

    pred = market.get_data(code=top1['CODE'], end_date=market.pass_days(top1['DATE'], 2), pattern_length=length)
    act = market.get_data(code=config.code, start_date=config.start_date, pattern_length=length)

    pred_ratio1 = (pred.iloc[-1]['CLOSE'] - pred.iloc[-2]['CLOSE']) / pred.iloc[-2]['CLOSE']
    act_ratio1 = (act.iloc[-1]['CLOSE'] - act.iloc[-2]['CLOSE']) / act.iloc[-2]['CLOSE']

    assert pred_ratio1 == pred_ratio, 'calcu error!'
    assert act_ratio1 == act_ratio, 'calcu error!'

    # compare_plot(norm(pred['CLOSE'].values), norm(act['CLOSE'].values), name)

if __name__ == '__main__':
    print('Cpu Core Num: ', os.cpu_count())
    time_start = time.time()

    # regression_test(get_daily_action_serial, 'serial_regression_result.jpg')

    # queue = Manager().Queue()
    regression_test(get_daily_action_parallel, 'parallel_regression_result.jpg')

    time_end = time.time()
    print('Total Time is:', time_end - time_start)