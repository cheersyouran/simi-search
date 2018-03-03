#encoding:utf-8
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

from os import listdir, path
from codes.config import *
from codes.base import norm

count = 0

def merge_raw_data():
    print('merge raw data...')
    col = ['CODE', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    merged_csv = pd.concat([pd.read_csv(path.join(config.RAW_DATA_DIR, f), header=None, names=col) for f in listdir(config.RAW_DATA_DIR)], axis=0)
    merged_csv = merged_csv.drop_duplicates()
    merged_csv.to_csv(config.DATA, index=False)
    merged_csv = pd.read_csv(config.DATA, parse_dates=['DATE'])
    merged_csv = merged_csv.sort_values(ascending=True, by=['CODE', 'DATE'])
    merged_csv.to_csv(config.DATA, index=False)

def gen_trading_days():
    df = pd.read_csv(config.ZZ800_DATA)
    pd.DataFrame(df['DATE'].unique()).to_csv(config.ZZ800_TRAINING_DAY, index=False)

def gen_800_data():
    print('gen 800 data...')
    codes = pd.read_csv(config.ZZ800_CODES)
    data = pd.read_csv(config.DATA)
    data = data[data['CODE'].isin(codes.values)]
    data.to_csv(config.ZZ800_DATA, index=False)

def gen_800_fft_data():
    print('gen 800 fft data...')
    print(config.speed_method)

    def rolling_aply_fft(data, freq, method):
        data_ = norm(data)
        ffts = np.fft.fft(data_)/len(data_)
        if method == 'fft':
            return np.abs(ffts[freq])
        elif method == 'deg':
            return np.rad2deg(np.angle(ffts[freq]))

    def apply(data, rolling_aply, freq, method):
        result = data.rolling(window=config.pattern_length).apply(func=rolling_aply, args=(freq, method))
        return result

    data = pd.read_csv(config.ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    data = data.dropna()

    ratio_300 = pd.read_csv(config.HS300_MARKET_RATIO, parse_dates=['DATE'])
    data = data.merge(ratio_300, on=['DATE'], how='left')

    ratio_800 = pd.read_csv(config.ZZ800_MARKET_RATIO, parse_dates=['DATE'])
    data = data.merge(ratio_800, on=['DATE'], how='left')

    assert data['800_RATIO'].isnull().any() == False

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft'+ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='fft')
        data['deg'+ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='deg')

    if config.speed_method == 'fft_euclidean':
        data.to_csv(config.ZZ800_FFT_DATA, index=False)
    elif config.speed_method == 'value_ratio_fft_euclidean':
        data.to_csv(config.ZZ800_VALUE_RATIO_FFT_DATA, index=False)
    elif config.speed_method == 'rm_vrfft_euclidean':
        data.to_csv(config.ZZ800_RM_VR_FFT, index=False)

def gen_800_RM_VR_fft_data():
    print('gen 800 remove-market-ratio fft data...')
    print(config.speed_method)

    def rolling_aply_fft(x, freq, method, data):
        global count
        x_test = data['CLOSE'].iloc[count:count+config.pattern_length].values
        if ~((x_test - x) == 0).any():
            raise Exception('Error!')
        ratio = data['800_RATIO'].iloc[count:count + config.pattern_length].values
        x_ = norm(x, ratio)
        count += 1

        ffts = np.fft.fft(x_)/len(x_)
        if method == 'fft':
            return np.abs(ffts[freq])
        elif method == 'deg':
            return np.rad2deg(np.angle(ffts[freq]))

    def apply(data, rolling_aply, freq, method):
        global count
        count = 0
        result = data.copy()
        result['CLOSE'] = data['CLOSE'].rolling(window=config.pattern_length).apply(func=rolling_aply, args=(freq, method, data))
        result['800_RATIO'] = data['800_RATIO']
        return result

    data = pd.read_csv(config.ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    data = data.dropna()

    ratio_800 = pd.read_csv(config.ZZ800_MARKET_RATIO, parse_dates=['DATE'])
    data = data.merge(ratio_800, on=['DATE'], how='left')

    ratio_300 = pd.read_csv(config.HS300_MARKET_RATIO, parse_dates=['DATE'])
    data = data.merge(ratio_300, on=['DATE'], how='left')

    assert data['800_RATIO'].isnull().any() == False

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft' + ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='fft')['CLOSE'].values
        data['deg' + ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='deg')['CLOSE'].values

    data.to_csv(config.ZZ800_RM_VR_FFT, index=False)

def gen_new_800_data():
    df1 = pd.read_csv('1.csv')
    df2 = pd.read_csv('2.csv')
    df = df1.append(df2).set_index(['DATE'])

    result = pd.DataFrame(columns=['DATE', 'CODE', 'CLOSE'])
    for _, item in df.iteritems():
        i = pd.DataFrame()
        i['DATE'] = item.index.values
        i['CODE'] = item.name
        i['CLOSE'] = item.values
        result = result.append(i)

    result.to_csv('800_data.csv', index=False)

def gen_300_fft_from_800_fft():

    codes_300 = pd.read_csv(config.HS300_CODES)['CODE'].values

    if config.speed_method == 'fft_euclidean':
        data = pd.read_csv(config.ZZ800_FFT_DATA)
    elif config.speed_method == 'value_ratio_fft_euclidean':
        data = pd.read_csv(config.ZZ800_VALUE_RATIO_FFT_DATA)

    data = data[data['CODE'].isin(codes_300)]

    if config.speed_method == 'fft_euclidean':
        data.to_csv(config.HS300_FFT_DATA, index=False)
    elif config.speed_method == 'value_ratio_fft_euclidean':
        data.to_csv(config.HS300_VALUE_RATIO_FFT_DATA, index=False)

if __name__ == '__main__':
    # config.speed_method = 'rm_vrfft_euclidean'
    # gen_800_fft_data()

    # insert_market_ratios_to_data()
    # gen_new_800_data()

    data = pd.read_csv(config.ZZ800_FFT_DATA, parse_dates=['DATE'], low_memory=False)

    data.isnull().any()

