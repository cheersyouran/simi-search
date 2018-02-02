from codes.config import *
from codes.base import plot_nav_curve

all_data = None

def get_historical_data(start_date=None, end_date=None, speed_method=speed_method, code=code, data=None):

    """
    :param speed_method: 'fft_euclidean', 'wave_fft_euclidean' ,'value_ratio_fft_euclidean'or others
    :param start_date: give a specific date to down-scope Pattern.(包含这一天，即预测下一天)
    :param end_date: give a specific date to up-scope Pattern.
    :return: all_data, pattern, targets
    """

    if speed_method == 'fft_euclidean':
        file = ZZ800_FFT_DATA
        col = 'CLOSE'
    elif speed_method == 'wave_fft_euclidean':
        file = ZZ800_WAVE_FFT_DATA
        col = 'denoise_CLOSE'
    elif speed_method == 'value_ratio_fft_euclidean':
        file = ZZ800_VALUE_RATIO_FFT_DATA
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

    if nb_data != 0:
        targets = targets.head(nb_data)

    return all_data, pattern, targets, col

# 获取每日的操作
def get_daily_action(start_date):

    if speed_method == None:
        from codes.all_search import load_and_process_data, all_search
        data, pattern, target = load_and_process_data(start_date)
        tops = all_search(pattern, target, nb_similarity)
    else:
        from codes.speed_search import speed_search
        data, pattern, target, col = get_historical_data(start_date=start_date)
        tops = speed_search(pattern, target, col)

    top1 = tops.iloc[0]

    # 预测值
    pred = data[(data['CODE'] == top1['CODE']) & (data['DATE'] >= top1['DATE'])].head(2)
    income_ratio = (pred.iloc[1]['CLOSE'] - pred.iloc[0]['CLOSE']) / pred.iloc[0]['CLOSE']

    # 实际值
    act = data[(data['CODE'] == code) & (data['DATE'] >= start_date)].head(pattern_length + 1).tail(2)
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
    dayofweek = date.dayofweek
    if dayofweek <= 5:
        return date
    else:
        while dayofweek > 5:
            date = date + timedelta(days=1)
            dayofweek = date.dayofweek
    return date

def pass_a_day(date):
    date = get_recent_weekdays(date)
    date = date + timedelta(days=1)
    return date

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
    start_date = get_recent_weekdays(start_date)
    pred_net_value, act_net_value, dates = regression_test(start_date, start_date + timedelta(days=regression_days))