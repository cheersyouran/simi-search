#encoding:utf-8
import pandas as pd
from os import listdir, path
from codes.config import *
from codes.base import norm

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

    def rolling_aply_fft(data, freq, method):
        data_ = norm(data)
        ffts = np.fft.fft(data_)/len(data_)
        if method == 'fft':
            return np.abs(ffts[freq])
        elif method == 'deg':
            return np.rad2deg(np.angle(ffts[freq]))

    def apply(data, rolling_aply, freq, method):
        result = data.rolling(window=config.pattern_length).apply(func=rolling_aply, args=(freq, method,))
        return result

    data = pd.read_csv(config.ZZ800_DATA, parse_dates=['DATE'])
    data = data.dropna()

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft'+ind] = data.groupby(['CODE'])['CLOSE'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='fft')
        data['deg'+ind] = data.groupby(['CODE'])['CLOSE'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='deg')

    if config.speed_method == 'fft_euclidean':
        data.to_csv(config.ZZ800_FFT_DATA, index=False)
    else:
        data.to_csv(config.ZZ800_VALUE_RATIO_FFT_DATA, index=False)

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

    codes_300 = pd.read_csv(config.HS300_CODES)['code'].values

    if config.speed_method == 'fft_euclidean':
        data = pd.read_csv(config.ZZ800_FFT_DATA)
    else:
        data = pd.read_csv(config.ZZ800_VALUE_RATIO_FFT_DATA)

    data = data[data['CODE'].isin(codes_300)]

    if config.speed_method == 'fft_euclidean':
        data.to_csv(config.HS300_FFT_DATA, index=False)
    else:
        data.to_csv(config.HS300_VALUE_RATIO_FFT_DATA, index=False)

if __name__ == '__main__':

    df = pd.read_csv(config.HS300_VALUE_RATIO_FFT_DATA)
    df['CODE'].unique()

    df = pd.read_csv(config.ZZ800_VALUE_RATIO_FFT_DATA)
    df['CODE'].unique()
    # gen_new_800_data()
    # gen_800_fft_data()
