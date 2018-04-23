import tushare as ts
import pandas as pd
import numpy as np
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

from codes.base import norm
from codes.config import config

def get_traiding_day(start=None, end=None):
    all_trading_day = ts.trade_cal()

    all_trading_day['calendarDate'] = all_trading_day['calendarDate'].apply(lambda x: pd.to_datetime(x))
    all_trading_day = all_trading_day[all_trading_day['isOpen'] == 1]
    if start is None:
        trading_day = all_trading_day[all_trading_day['calendarDate'] <= end]
    elif end is None:
        trading_day = all_trading_day[all_trading_day['calendarDate'] >= start]
    else:
        trading_day = all_trading_day[(all_trading_day['calendarDate'] >= start) & (all_trading_day['calendarDate'] <= end)]
    all_trading_day.columns = [['DATE', 'OPEN']]
    return trading_day, all_trading_day

def _gen_zz800_stock_list():
    '''
    :return: 生成中证800股票代码，[code, name, weight, date]
    '''
    zz500 = ts.get_zz500s()
    hs300 = ts.get_hs300s()
    zz800 = zz500.append(hs300)
    zz800 = zz800[['code', 'name', 'weight', 'date']]
    zz800.columns = zz800.columns.str.upper()
    zz800 = zz800.sort_values(ascending=True, by=['CODE'])
    zz800.to_csv(config.ZZ800_CODES, index=False)
    return zz800

def get_zz800_hist_data(trading_day):

    market_start_date = str(trading_day.iloc[0]['calendarDate'].date())
    start_date = str(trading_day.iloc[1]['calendarDate'].date())
    end_date = str(trading_day.iloc[-1]['calendarDate'].date())

    codes = _gen_zz800_stock_list()['CODE'].head(10)
    if codes.empty:
        raise Exception('No codes file found!')

    datas = []
    for code in codes:
        data = ts.get_k_data(code, start=start_date, end=end_date)
        if data.size == 0:
            print(code)
            continue
        else:
            data['300_market'] = 0
            data['300_ratio'] = 0
            data = data.reset_index()
            datas.append(data[['close', 'code', 'date', '300_market', '300_ratio']])

    hist_data = pd.concat(datas)
    hist_data.columns = hist_data.columns.str.upper()

    market = ts.get_k_data('000906', start=market_start_date, end=end_date, index=True)[['date', 'close']]
    market.columns = ['DATE', '800_MARKET']

    def rolling_aply(input):
        return (input[1] - input[0]) / input[0]

    market['800_RATIO'] = market['800_MARKET'].rolling(window=2).apply(func=rolling_aply)
    market = market.dropna()
    hist_data = hist_data.merge(market, on=['DATE'], how='left')
    hist_data = _format_code_and_date(hist_data)

    return hist_data

def _update_800_RM_VR_fft_data(data):
    def apply(data, freq, method):
        result = data.copy()
        fft = []
        for i in range(data.shape[0]):
            if i < 30:
                fft.append(None)
            else:
                close = data['CLOSE'].iloc[i - 30: i].values
                market = data['800_RATIO'].iloc[i - 30: i].values
                x_ = norm(close, market)
                ffts = np.fft.fft(x_) / len(x_)
                if method == 'fft':
                    fft.append(np.abs(ffts[freq]))
                elif method == 'deg':
                    fft.append(np.rad2deg(np.angle(ffts[freq])))

        result['CLOSE'] = fft
        result['800_RATIO'] = data['800_RATIO']
        return result

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft' + ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, freq=i, method='fft')['CLOSE'].values
        data['deg' + ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, freq=i, method='deg')['CLOSE'].values
        print(ind)

    return data

def update_data():
    start = config.update_start
    end = config.update_end

    zz800_data = pd.read_csv(config.ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    trading_day, all_trading_day = get_traiding_day(start, end)
    recently_zz800_data = get_zz800_hist_data(trading_day)
    new_zz800_data = zz800_data.append(recently_zz800_data)
    sorted_new_zz800_data = new_zz800_data.sort_values(ascending=True, by=['CODE', 'DATE'])

    zz800_fft_data = pd.read_csv(config.ZZ800_RM_VR_FFT, parse_dates=['DATE'], low_memory=False)
    date, _ = get_traiding_day(start=None, end=start)
    date = date.tail(30).head(1).values[0][0]
    new_fft_zz800 = zz800_fft_data[zz800_fft_data['DATE'] >= date]
    new_fft_zz800 = new_fft_zz800.append(recently_zz800_data)
    new_fft_zz800 = new_fft_zz800.sort_values(ascending=True, by=['CODE', 'DATE'])
    rm_vr_data = _gen_800_RM_VR_fft_data(new_fft_zz800)
    new_zz800_rm_vr_data = zz800_fft_data.append(rm_vr_data)
    sorted_new_zz800_rm_vr_data = new_zz800_rm_vr_data.sort_values(ascending=True, by=['CODE', 'DATE'])

    sorted_new_zz800_data.drop_duplicates(subset=['CODE', 'DATE'], inplace=True)
    sorted_new_zz800_rm_vr_data.drop_duplicates(subset=['CODE', 'DATE'], inplace=True)

    return sorted_new_zz800_rm_vr_data, sorted_new_zz800_data, all_trading_day

def init_dataset_matrix():
    '''
    将 800_raw_data.csv 映射到 800_data.csv。
    后者是一个按calendar day排列的字典，空值补Nan
    '''

    zz800_raw_data = pd.read_csv(config.ZZ800_RAW_DATA, dtype={'SecuCode': str}, parse_dates=['date'])
    zz800_raw_data = zz800_raw_data[['SecuCode', 'date', 'ret', 'close']]
    zz800_raw_data.columns = ['CODE', 'DATE', 'RET', 'CLOSE']

    def _format_code(data):

        def apply(x):
            if int(x[0]) >= 6:
                return x + '.SH'
            else:
                return x + '.SZ'

        data['CODE'] = data['CODE'].apply(func=apply)
        return data

    zz800_raw_data = _format_code(zz800_raw_data)

    trading_day = pd.read_csv(config.TRAINING_DAY, parse_dates=['DATE'])
    trading_day = trading_day[(trading_day['DATE'] > '2007-01-01') & (trading_day['DATE'] < '2018-01-01')]

    def _mapping_to_dataset():
        data = []
        def apply(x):
            ret = trading_day.merge(x, on=['DATE'], how='left')
            ret['CODE'] = ret['CODE'].fillna('placeholder')
            data.append(ret)

        zz800_raw_data.groupby(['CODE']).apply(func=apply)
        dataset = pd.concat(data)

        index_ratio = pd.read_csv(config.MARKET_RATIO, parse_dates=['DATE'])
        dataset = dataset.merge(index_ratio, on=['DATE'], how='left')
        return dataset

    dataset = _mapping_to_dataset()

    dataset.to_csv(config.ZZ800_DATA, index=False)

def gen_800_RM_VR_fft_data():
    print('gen 800 remove-market-ratio fft data...')
    print(config.speed_method)

    def apply(data, freq, method):
        result = data.copy()
        fft = []
        for i in range(data.shape[0]):
            if i < 30:
                fft.append(None)
            else:
                ret = data['RET'].iloc[i - 30: i].values
                market = data['800_RATIO'].iloc[i - 30: i].values
                x_ = norm(ret, market)
                ffts = np.fft.fft(x_) / len(x_)
                if method == 'fft':
                    fft.append(np.abs(ffts[freq]))
                elif method == 'deg':
                    fft.append(np.rad2deg(np.angle(ffts[freq])))

        result['RET'] = fft
        return result

    data = pd.read_csv(config.ZZ800_DATA, low_memory=False)

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft' + ind] = data.groupby(['CODE'])[['CODE', '800_RATIO', 'RET']].apply(func=apply, freq=i, method='fft')['RET'].values
        data['deg' + ind] = data.groupby(['CODE'])[['CODE', '800_RATIO', 'RET']].apply(func=apply, freq=i, method='deg')['RET'].values
        print(ind)

    data.to_csv(config.ZZ800_RM_VR_FFT, index=False)

if __name__ == '__main__':

    # init_dataset_matrix()
    gen_800_RM_VR_fft_data()

    print('')
