from codes.config import *
from codes.base import plot_nav_curve

# 加载数据

# 获取每日的操作
def get_daily_action(start_date):

    if speed_method == None:
        from codes.all_search import load_and_process_data, find_tops_similar
        data, pattern, target = load_and_process_data(start_date)
        tops = find_tops_similar(pattern, target, nb_similarity)
    else:
        from codes.part_search import part_search, load_data
        data, pattern, target, col = load_data(start_date)
        tops = part_search(pattern, target, col)

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