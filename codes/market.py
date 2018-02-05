from datetime import timedelta

from codes.base import plot_nav_curve
from codes.config import *

all_data = None

def get_historical_data(start_date=None, end_date=None, speed_method=config.speed_method,
                        code=config.code, data=None, pattern_length=config.pattern_length):

    """
    :param speed_method: 'fft_euclidean', 'wave_fft_euclidean' ,'value_ratio_fft_euclidean'or others
    :param start_date: give a specific date to down-scope Pattern.(包含这一天，即预测下一天)
    :param end_date: give a specific date to up-scope Pattern.
    :return: all_data, pattern, targets
    """

    if speed_method == 'fft_euclidean':
        file = config.ZZ800_FFT_DATA
        col = 'CLOSE'
    elif speed_method == 'wave_fft_euclidean':
        file = config.ZZ800_WAVE_FFT_DATA
        col = 'denoise_CLOSE'
    elif speed_method == 'value_ratio_fft_euclidean':
        file = config.ZZ800_VALUE_RATIO_FFT_DATA
        col = 'CLOSE'

    global all_data
    if all_data is None:
        all_data = pd.read_csv(file, parse_dates=['DATE'])

    if data != None:
        all_data = data

    all_data = all_data.loc[(all_data != 0).any(axis=1)]
    targets = all_data[all_data['CODE'] != code].reset_index(drop=True)

    if start_date == None and end_date != None:
        pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date)].tail(pattern_length)
    elif start_date != None and end_date == None:
        pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] >= start_date)].head(pattern_length)
    elif start_date != None and end_date != None:
        pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]

    pattern = pattern.reset_index(drop=True)

    if config.nb_data != 0:
        targets = targets.head(config.nb_data)

    return all_data, pattern, targets, col

def get_data(start_date=None, end_date=None, speed_method=config.speed_method,
                        code=config.code, data=None, pattern_length=config.pattern_length):

    if speed_method == 'fft_euclidean':
        file = config.ZZ800_FFT_DATA
    elif speed_method == 'wave_fft_euclidean':
        file = config.ZZ800_WAVE_FFT_DATA
    elif speed_method == 'value_ratio_fft_euclidean':
        file = config.ZZ800_VALUE_RATIO_FFT_DATA

    global all_data
    if all_data is None:
        all_data = pd.read_csv(file, parse_dates=['DATE'])

    if data != None:
        all_data = data

    if start_date == None and end_date != None:
        pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date)].tail(pattern_length)
    elif start_date != None and end_date == None:
        pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] >= start_date)].head(pattern_length)
    elif start_date != None and end_date != None:
        pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]

    pattern = pattern.reset_index(drop=True)

    return all_data, pattern,

# 获取每日的操作
def get_daily_action(start_date):

    if config.speed_method == None:
        from codes.search.all_search import all_search
        data, pattern, target, col = get_historical_data(start_date=start_date)
        tops = all_search(pattern, target, config.nb_similarity)
    else:
        from codes.search.speed_search import speed_search
        data, pattern, target, col = get_historical_data(start_date=start_date)
        tops = speed_search(pattern, target, col)

    top1 = tops.iloc[0]

    # 预测值
    pred = data[(data['CODE'] == top1['CODE']) & (data['DATE'] >= top1['DATE'])].head(2)
    income_ratio = (pred.iloc[1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']

    # 实际值
    act = data[(data['CODE'] == config.code) & (data['DATE'] >= start_date)].head(config.pattern_length + 1).tail(2)
    act_income_ratio = (act.iloc[1]['CLOSE'] - act.iloc[0]['CLOSE']) / act.iloc[0]['CLOSE']

    # 根据income_ratio来做决策
    if income_ratio > 0:
        action = 1
    elif income_ratio < 0:
        action = -1
    else:
        action = 0

    print('##############################################')
    print('Date: ' + str(start_date.date()))
    print('Predict Ratio:', income_ratio)
    print('Actual  Ratio:', act_income_ratio)
    return action, income_ratio, act_income_ratio

def get_recent_weekdays(date):
    dayofweek = date.weekday_name
    if dayofweek not in ['Saturday', 'Sunday']:
        return date
    else:
        while dayofweek not in ['Saturday', 'Sunday']:
            date = date + timedelta(days=1)
            dayofweek = date.weekday_name
    return date

def get_next_weekdays(date):
    date_ = date + timedelta(days=1)
    dayofweek = date_.weekday_name
    if dayofweek not in ['Saturday', 'Sunday']:
        return date_
    else:
        while dayofweek in ['Saturday', 'Sunday']:
            date_ = date_ + timedelta(days=1)
            dayofweek = date_.weekday_name
    return date_

def pass_a_day(date):
    date_ = get_recent_weekdays(date)
    date_ = get_next_weekdays(date_)
    return date_

def regression_test(start_date, end_date):
    state = 0 # 当前持股状态，0未持股，1持股
    date = start_date
    strategy_net_value = [1.0]
    act_net_value = [1.0]
    dates = [date]
    while date <= end_date:

        action, strategy_income_ratio, act_income_ratio = get_daily_action(date, )
        if (state == 0) & (action == 1):
            print('[Action]: Buy in!')
            state = 1
            strategy_net_value.append(strategy_net_value[-1])
        elif (state == 0) & (action != 1):
            print('[Action]: Keep empty')
            strategy_net_value.append(strategy_net_value[-1])
        elif (state == 1) & (action == -1):
            print('[Action]: Sold out')
            state = 0
            strategy_net_value.append(strategy_net_value[-1] * (1 + act_income_ratio))
        elif (state == 1) & (action != -1):
            print('[Action]: Keep full')
            strategy_net_value.append(strategy_net_value[-1] * (1 + act_income_ratio))
        else:
            raise Exception('No Such Combination! state:' + state + ', action:' + action)

        act_net_value.append(act_net_value[-1] * (1 + act_income_ratio))

        date = pass_a_day(date)
        dates.append(date)

        plot_nav_curve(strategy_net_value, act_net_value, dates)

    return strategy_net_value, act_net_value, dates

if __name__ == '__main__':
    start_date = get_recent_weekdays(config.start_date)
    pred_net_value, act_net_value, dates = regression_test(start_date, start_date + timedelta(days=config.regression_days))
