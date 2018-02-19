import time
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from os import listdir, path
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from multiprocessing import Pool, Manager

class Config:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        print('Init config!', os.getpid())
        self.rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

        self.RAW_DATA_DIR = self.rootPath + '/raw_data'
        self.DATA = self.rootPath + '/data/data.csv'

        self.ZZ800_CODES = self.rootPath + '/data/800_codes.csv'
        self.ZZ800_DATA = self.rootPath + '/data/800_data.csv'
        # self.ZZ800_WAVE_DATA = self.rootPath + '/data/800_wave_data.csv'
        self.ZZ800_FFT_DATA = self.rootPath + '/data/800_fft_data.csv'
        self.ZZ800_WAVE_FFT_DATA = self.rootPath + '/data/800_wave_fft_data.csv'
        self.ZZ800_VALUE_RATIO_FFT_DATA = self.rootPath + '/data/800_value_ratio_fft_data.csv'
        self.ZZ800_TRAINING_DAY = self.rootPath + '/data/800_training_day.csv'

        self.code = '000001.SZ'
        self.nb_codes = 4

        self.pattern_length = 30
        self.regression_days = 30
        self.start_date = pd.to_datetime('2017-02-24')
        # self.start_date = pd.to_datetime('2016-01-01')
        self.end_date = self.start_date + timedelta(days=self.regression_days)

        self.parallel = True
        self.speed_method = 'fft_euclidean' # 'value_ratio_fft_euclidean'
        self.fft_level = 3
        self.similarity_method = 'euclidean' #'pearsonr'

        self.nb_similarity = 2
        self.nb_to_make_action = 2
        self.nb_data = 0

        self.weighted_dist = True

config = Config()

class Market:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self.all_data = None
        self._init_all_data(config.speed_method)

        self.codes = None
        self._init_codes()

        self.ratios = None
        self._init_ratios()

        self.trading_days = None
        self._init_trading_days()

        self.pattern = None
        self.targets = None
        self.current_date = self.pass_days(config.start_date, config.pattern_length)
        self.col = 'CLOSE'

    def _init_all_data(self, speed_method=config.speed_method):

        if speed_method == 'fft_euclidean':
            file = config.ZZ800_FFT_DATA
        elif speed_method == 'wave_fft_euclidean':
            file = config.ZZ800_WAVE_FFT_DATA
            self.col = 'denoise_CLOSE'
        elif speed_method == 'value_ratio_fft_euclidean':
            file = config.ZZ800_VALUE_RATIO_FFT_DATA

        if self.all_data is None:
            print('Init all data! ', os.getpid())
            self.all_data = pd.read_csv(file, parse_dates=['DATE'], low_memory=False)

    def _init_ratios(self):
        self.ratios = pd.read_csv('../data/800_ratio.csv', parse_dates=['DATE'])

    def _init_codes(self):
        self.codes = pd.read_csv(config.ZZ800_CODES).head(config.nb_codes).values.flatten()

    def _init_trading_days(self):
        self.trading_days = pd.read_csv(config.ZZ800_TRAINING_DAY, parse_dates=['DATE'])

    def _pass_a_day(self):
        self.current_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > self.current_date].head(1).values[0][0])
        config.start_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > config.start_date].head(1).values[0][0])

    def get_historical_data(self, start_date=None, end_date=None, speed_method=config.speed_method, code=config.code):

        self.targets = self.all_data[self.all_data['CODE'] != code].reset_index(drop=True)

        if start_date == None and end_date != None:
            self.pattern = self.all_data[(self.all_data['CODE'] == code) & (self.all_data['DATE'] <= end_date)].tail(config.pattern_length)
        elif start_date != None and end_date == None:
            self.pattern = self.all_data[(self.all_data['CODE'] == code) & (self.all_data['DATE'] >= start_date)].head(config.pattern_length)
        elif start_date != None and end_date != None:
            self.pattern = self.all_data[(self.all_data['CODE'] == code) & (self.all_data['DATE'] <= end_date) & (self.all_data['DATE'] >= start_date)]

        self.pattern = self.pattern.reset_index(drop=True)

        if config.nb_data != 0:
            self.targets = self.targets.head(config.nb_data)

        return self.all_data, self.pattern, self.targets, self.col

    def get_data(self, start_date=None, end_date=None, code=None, pattern_length=config.pattern_length):

        all_data = self.all_data

        if code is not None:
            if start_date == None and end_date != None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date)].tail(pattern_length)
            elif start_date != None and end_date == None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] >= start_date)].head(pattern_length)
            elif start_date != None and end_date != None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]
        else:
            if start_date == None and end_date != None:
                pattern = all_data[(all_data['DATE'] <= end_date)].tail(pattern_length)
            elif start_date != None and end_date == None:
                pattern = all_data[(all_data['DATE'] >= start_date)].head(pattern_length)
            elif start_date != None and end_date != None:
                pattern = all_data[(all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]

        pattern = pattern.reset_index(drop=True)

        return pattern

    def pass_days(self, date, nb_day):
        date_ = pd.to_datetime(self.trading_days[self.trading_days['DATE'] >= date].head(nb_day).tail(1).values[0][0])
        return date_


market = Market()

def weighted_distance(x, y, length):
    if config.weighted_dist == True:
        weight = np.arange(1, 2, 1 / length)
        dist = np.abs(np.multiply(pd.DataFrame(x).values - pd.DataFrame(y).values, weight.reshape(30, 1)))
        return np.sum(dist)
    else:
        return euclidean(x, y)

def norm(X):
    if config.speed_method in ['value_ratio_fft_euclidean', 'changed']:
        result = X / pd.DataFrame(X).iloc[0][0]
    else:
        result = preprocessing.scale(X)
    return result

def speed_search(pattern, targets, code=config.code, col='CLOSE'):

    if config.speed_method in ['fft_euclidean', 'wave_fft_euclidean', 'fft']:
        ALPHA = np.multiply([1, 1, 1, 1, 1], 100)
        BETA = np.multiply([1, 1, 1, 1, 1], 1)
    elif config.speed_method in ['value_ratio_fft_euclidean']:
        ALPHA = np.multiply([1, 1, 1, 1, 1], 100)
        BETA = np.multiply([1, 1, 1, 1, 1], 1)

    targets['fft_deg'] = 0
    for i in range(config.fft_level):
        index = str(i + 1)
        p_fft = pattern['fft' + index].tail(1).values
        p_deg = pattern['deg' + index].tail(1).values

        targets['fft_' + index] = (targets['fft' + index] - p_fft).abs() * ALPHA[i]
        targets['deg_' + index] = (targets['deg' + index] - p_deg).abs() * BETA[i]

        targets['fft_deg'] += targets['fft_' + index] + targets['deg_' + index]

    sorted_std_diff = targets.sort_values(ascending=True, by=['fft_deg'])

    sorted_std_diff = sorted_std_diff[sorted_std_diff['VOLUME'] != '0.0000']
    sorted_std_diff = sorted_std_diff[sorted_std_diff['VOLUME'] != 0]
    sorted_std_diff = sorted_std_diff[sorted_std_diff['VOLUME'] != '0']

    sorted_std_diff = sorted_std_diff.head(200)

    sorted_std_diff[config.similarity_method] = -1

    for i in sorted_std_diff.index.values:
        ith = sorted_std_diff.loc[i]
        result = targets[(targets['CODE'] == ith['CODE']) & (targets['DATE'] <= ith['DATE'])].tail(config.pattern_length)
        sorted_std_diff.loc[i, config.similarity_method] = weighted_distance(norm(result[col]), norm(pattern[col]), config.pattern_length)
        # sorted_std_diff.loc[i, similarity_method] = pearsonr(result['CLOSE'], pattern['CLOSE'])[0]

    tops = sorted_std_diff.sort_values(ascending=True, by=[config.similarity_method]).head(config.nb_similarity)

    return tops

def parallel_speed_search(code):
    all_data, pattern, targets, col = market.get_historical_data(start_date=config.start_date, code=code)
    tops = speed_search(pattern, targets, code, col)
    top1 = tops.head(1)[['CODE', 'DATE', config.similarity_method]].values.flatten()
    top1 = np.hstack((top1, code))
    # queue.put([tops, pattern, code])
    return top1

def get_daily_action_parallel():

    pool = Pool(processes=os.cpu_count())
    top1s = pool.map(parallel_speed_search, market.codes)
    top1s = pd.DataFrame(columns=['CODE', 'DATE', config.similarity_method, 'ORG_CODE'], data=top1s)

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

    top1s = top1s[top1s['pred_ratio'] > 0]
    # top1s_df.sort_values(ascending=False, by=['pred_ratio']).head(config.nb_to_make_action)

    pred_ratio = -1 if top1s.shape[0] == 0 else np.sum(top1s['pred_ratio']) * (1 / top1s.shape[0])
    act_ratio = -1 if top1s.shape[0] == 0 else np.sum(top1s['act_ratios']) * (1 / top1s.shape[0])
    market_ratio = float(market.ratios[market.ratios['DATE'] == market.current_date]['ratio']) / 100

    if pred_ratio > 0:
        action = 1
    else:
        action = -1

    print('[Predict]:', pred_ratio)
    print('[Actual ]:', act_ratio)
    print('[Market ]:', market_ratio)

    return action, pred_ratio, act_ratio, market_ratio

def plot_simi_stock(top, data, pattern, filename, codes=config.code):
    print('plot simi stock of ', codes, ' ...')
    def init_plot_data():
        plot_codes = np.append(top['CODE'].values, codes)
        plot_dates = list(top['DATE'].values)
        plot_dates.append(None)
        plot_prices = np.zeros([config.nb_similarity + 1, config.pattern_length])
        plot_legend = list()
        return plot_codes, plot_dates, plot_prices, plot_legend

    plot_codes, plot_dates, plot_prices, plot_legend = init_plot_data()

    for i in range(config.nb_similarity):
        plot_quote = data[data['CODE'] == plot_codes[i]]
        plot_quote = plot_quote[plot_quote['DATE'] <= pd.to_datetime(plot_dates[i])].tail(config.pattern_length)
        plot_prices[i] = plot_quote['CLOSE'].values
        plot_legend.append(
            str(plot_codes[i]) + "," + config.similarity_method + ":" +
            str(top.iloc[i][config.similarity_method]))

    plot_prices[-1] = pattern['CLOSE'].values
    plot_dates[-1] = pattern['DATE'].iloc[-1]
    plot_legend.append(str(plot_codes[-1]))
    norm_plot_prices = [norm(plot_prices[i]) for i in range(config.nb_similarity + 1)]

    # print('股票价格:')
    # print(plot_prices)

    # print('股票norm:')
    # print(norm_plot_prices)

    # 验证结果
    from codes.all_search import t_rol_aply
    if config.speed_method not in ['changed']:
        for i in range(config.nb_similarity):
            a = t_rol_aply(plot_prices[i], plot_prices[-1])
            b = top.iloc[i][config.similarity_method]
            # print(a, b)
            assert a == b, 'calcu error!'

    line_styles = ['k--', 'k:', 'k-.', 'k--', 'k:', 'k-.', 'k:', 'k-.', 'k--', 'k:', 'k-.']
    for i in range(plot_codes.size):
        if i == plot_codes.size - 1:
            plt.plot(norm_plot_prices[i], 'r-', label=norm_plot_prices[i], linewidth=1.5)
        else:
            plt.plot(norm_plot_prices[i], line_styles[i], label=norm_plot_prices[i], linewidth=1.2)

    plt.xlabel('Time')
    if config.speed_method == 'value_ratio_fft_euclidean':
        plt.ylabel('NAV')
    else:
        plt.ylabel('Close Price')
    plt.legend(plot_legend, loc='upper left')
    plt.title("Similarity Search[" + plot_codes[-1] + "]\n")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=45)
    plt.ioff()
    # plt.show()
    plt.savefig('../pic/' + filename+'.jpg')
    plt.close()

def plot_nav_curve(strategy_net_value, act_net_value, market_net_value, dates, name):
    print('\nplot nav curve...')
    plt.plot(dates, strategy_net_value, 'r-', label=strategy_net_value, linewidth=1.5)
    plt.plot(dates, act_net_value, 'k-', label=act_net_value, linewidth=1.5)
    plt.plot(dates, market_net_value, 'g-', label=market_net_value, linewidth=1.5)

    plt.xlabel('Time')
    plt.ylabel('Net Asset Value')
    plt.legend(['strategy', 'baseline', 'market'], loc='upper left')
    plt.title("Net Asset Value")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=20)
    plt.xticks(fontsize=8)
    plt.ioff()
    # plt.show()
    plt.savefig('../pic/' + name)
    plt.close()

def compare_plot(x1, x2):
    plt.plot(x1)
    plt.plot(x2)
    plt.grid(True)
    plt.ioff()
    plt.savefig('../pic/' + 'compare_result.jpg')
    plt.close()

def regression_test(func, name):

    strategy_net_values = [1.0]
    act_net_values = [1.0]
    market_net_values = [1.0]
    dates = [market.current_date.date()]
    while config.start_date <= config.end_date:

        print('\n[Current Date]: ' + str(market.current_date.date()))

        action, pred_ratio, act_ratio, market_ratios = func()

        if action == 1:
            print('[Action]: Buy in!')
            strategy_net_values.append(strategy_net_values[-1] * (1 + act_ratio))
        else:
            print('[Action]: Keep Empty')
            strategy_net_values.append(strategy_net_values[-1])

        act_net_values.append(act_net_values[-1] * (1 + act_ratio))
        market_net_values.append(market_net_values[-1] * (1 + market_ratios))

        market._pass_a_day()
        dates.append(market.current_date.date())
        plot_nav_curve(strategy_net_values, act_net_values, market_net_values, dates, name)

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

    if config.parallel:
        # queue = Manager().Queue()
        regression_test(get_daily_action_parallel, 'parallel_regression_result.jpg')

    time_end = time.time()
    print('Total Time is:', time_end - time_start)