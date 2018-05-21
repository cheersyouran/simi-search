'''
工具类，用于生成新的fft数据
'''

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


def _format_code(data):
    def apply(x):
        if int(x[0]) >= 6:
            return x + '.SH'
        else:
            return x + '.SZ'

    data['CODE'] = data['CODE'].apply(func=apply)
    return data


def get_traiding_day(start=None, end=None):
    all_trading_day = pd.read_csv(config.TRAINING_DAY)
    all_trading_day.columns = ['DATE', 'OPEN']
    all_trading_day = all_trading_day[all_trading_day['OPEN'] == 1]

    if start is None:
        trading_day = all_trading_day[all_trading_day['DATE'] <= end]
    elif end is None:
        trading_day = all_trading_day[all_trading_day['DATE'] >= start]
    else:
        trading_day = all_trading_day[(all_trading_day['DATE'] >= start) & (all_trading_day['DATE'] <= end)]
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


def _get_new_zz800_data(trading_day):

    start_date = trading_day.iloc[0]['DATE']
    end_date = trading_day.iloc[-1]['DATE']
    nb_date = trading_day.shape[0]

    codes = _gen_zz800_stock_list()['CODE']
    if codes.empty:
        raise Exception('No codes file found!')

    market_ratio = ts.get_hist_data('hs300', start=start_date, end=end_date).reset_index()
    datas = []
    for code in codes:
        data = ts.get_hist_data(code, start=start_date, end=end_date)
        if data is None or data.shape[0] != nb_date:
            data = ts.get_k_data(code, start='2010-01-01', end=end_date).reset_index().tail(nb_date + 1)
            if data is None or data.shape[0] != nb_date + 1:
                print('停牌或找不到：', code)
                continue
            data['p_change'] = data['close'] / data['close'].shift(1) - 1

            data = data.tail(nb_date)
        data['800_ratio'] = 0
        data['code'] = code
        data = data.reset_index()
        datas.append(data[['close', 'code', 'date', 'p_change', '800_ratio']])

    hist_data = pd.concat(datas)
    hist_data.columns = ['close', 'code', 'date', 'ret', '800_ratio']
    hist_data = hist_data.merge(market_ratio[['date', 'p_change']], on=['date'], how='left')
    hist_data.columns = ['close', 'code', 'date', 'ret', '800_ratio', '300_ratio']
    hist_data.columns = hist_data.columns.str.upper()

    hist_data = _format_code(hist_data)
    return hist_data


def update_data(new_dataset=None):

    def update_trading_date():
        trading_day = ts.trade_cal()
        trading_day['calendarDate'] = trading_day['calendarDate'].apply(lambda x: pd.to_datetime(x))
        trading_day = trading_day[trading_day['isOpen'] == 1]
        trading_day.columns = [['DATE', 'OPEN']]
        trading_day.to_csv(config.TRAINING_DAY, index=False)

    def _update_quotation_data():

        zz800_dataset = pd.read_csv(config.ZZ800_DATA, low_memory=False)
        new_zz800_data = _get_new_zz800_data(trading_day)

        codes = np.unique(np.hstack((new_zz800_data['CODE'].unique(), zz800_dataset['CODE'].unique())))
        dates = trading_day['DATE'].values

        item = []
        for code in codes:
            for date in dates:
                item.append([code, date])
        dataset = pd.DataFrame(item, columns=['CODE', 'DATE'])

        update_dataset = dataset.merge(new_zz800_data, how='left', on=['CODE', 'DATE'])
        new_dataset = zz800_dataset.append(update_dataset)
        new_dataset = new_dataset.drop_duplicates(['CODE', 'DATE'])
        new_dataset = new_dataset.sort_values(ascending=True, by=['CODE', 'DATE'])
        return new_dataset

    def _update_RMVR_fft_data(new_dataset):
        date, _ = get_traiding_day(start=None, end=start)
        date = date.tail(30).head(1).values[0][0]

        df = new_dataset[new_dataset['DATE'] >= date]
        new_fft_dataset = init_800_RMVR_fft_data(df)

        new_fft_dataset = new_fft_dataset[new_fft_dataset['DATE'] >= start]

        fft_dataset = pd.read_csv(config.ZZ800_RM_VR_FFT, low_memory=False)
        new_fft_dataset = fft_dataset.append(new_fft_dataset)
        new_fft_dataset = new_fft_dataset.sort_values(ascending=True, by=['CODE', 'DATE'])
        new_fft_dataset = new_fft_dataset.drop_duplicates(['CODE', 'DATE'])
        return new_fft_dataset

    update_trading_date()
    trading_day, all_trading_day = get_traiding_day(start, end)
    if new_dataset is None:
        new_dataset = _update_quotation_data()
    zz800_fft_dataset = _update_RMVR_fft_data(new_dataset)

    return new_dataset, zz800_fft_dataset


def init_dataset_matrix():
    '''
    将 800_raw_data.csv 映射到 800_data.csv。
    后者是一个按calendar day排列的字典，空值补Nan
    '''

    zz800_raw_data = pd.read_csv(config.ZZ800_RAW_DATA, dtype={'SecuCode': str})
    zz800_raw_data = zz800_raw_data[['SecuCode', 'date', 'ret', 'close']]
    zz800_raw_data.columns = ['CODE', 'DATE', 'RET', 'CLOSE']

    zz800_raw_data = _format_code(zz800_raw_data)

    trading_day = pd.read_csv(config.TRAINING_DAY)
    trading_day = trading_day[(trading_day['DATE'] > '2007-01-01') & (trading_day['DATE'] < '2018-01-01')]

    def _mapping_to_dataset():
        data = []
        def apply(x):
            ret = trading_day.merge(x, on=['DATE'], how='left')
            ret['CODE'] = x['CODE'].values[0]
            data.append(ret)

        zz800_raw_data.groupby(['CODE']).apply(func=apply)
        dataset = pd.concat(data)

        index_ratio = pd.read_csv(config.MARKET_RATIO)
        dataset = dataset.drop_duplicates()
        dataset = dataset.merge(index_ratio, on=['DATE'], how='left')
        return dataset

    dataset = _mapping_to_dataset()
    dataset.to_csv(config.ZZ800_DATA, index=False)


def init_800_RMVR_fft_data(input_data=None):
    print('init 800 remove-market-ratio fft data...')

    def apply(data, freq, method):
        result = data.copy()
        fft = []
        for i in range(data.shape[0]):
            if i < 30:
                fft.append(None)
            else:
                ret = data['RET'].iloc[i - 30: i].values
                market = data['300_RATIO'].iloc[i - 30: i].values
                x_ = norm(ret, market)
                ffts = np.fft.fft(x_) / len(x_)
                if method == 'fft':
                    fft.append(np.abs(ffts[freq]))
                elif method == 'deg':
                    fft.append(np.rad2deg(np.angle(ffts[freq])))

        result['RET'] = fft
        return result

    if input_data is None:
        data = pd.read_csv(config.ZZ800_DATA, low_memory=False)
    else:
        data = input_data

    for i in range(config.fft_level):
        ind = str(i+1)
        data.loc[:, 'fft' + ind] = data.groupby(['CODE'])[['CODE', '300_RATIO', 'RET']].apply(func=apply, freq=i, method='fft')['RET'].values
        data.loc[:, 'deg' + ind] = data.groupby(['CODE'])[['CODE', '300_RATIO', 'RET']].apply(func=apply, freq=i, method='deg')['RET'].values
        print('Calculating fft: ', ind)
    if input_data is None:
        data.to_csv(config.ZZ800_RM_VR_FFT, index=False)
    else:
        return data

if __name__ == '__main__':

    init_dataset_matrix()
    init_800_RMVR_fft_data()

    # start = config.update_start
    # end = config.update_end
    #
    # zz800_dataset, zz800_fft_dataset = update_data()
    # zz800_fft_dataset.to_csv(config.ZZ800_RM_VR_FFT, index=False)
    # zz800_dataset.to_csv(config.ZZ800_DATA, index=False)

    print('')
