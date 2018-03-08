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

def gen_800_fft_data(path):
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

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft'+ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='fft')
        data['deg'+ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='deg')

    data.to_csv(path, index=False)

def gen_800_RM_VR_fft_data(path):
    print('gen 800 remove-market-ratio fft data...')
    print(config.speed_method)

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

    data = pd.read_csv(config.ZZ800_DATA, low_memory=False)

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft' + ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, freq=i, method='fft')['CLOSE'].values
        data['deg' + ind] = data.groupby(['CODE'])['CLOSE', '800_RATIO'].apply(func=apply, freq=i, method='deg')['CLOSE'].values
        print(ind)

    data.to_csv(path, index=False)

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

def add_market_ratio():
    data = pd.read_csv(config.ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    index_ratio = pd.read_csv(config.MARKET_RATIO, parse_dates=['DATE'])
    data = data.merge(index_ratio, on=['DATE'], how='left')
    data.to_csv(config.ZZ800_DATA, index=False)

def read_excel():
    df = pd.read_csv(config.rootPath + '/data/800_data_merged.csv').set_index(['Date'])
    result = pd.DataFrame(columns=['DATE', 'CODE', 'CLOSE'])

    df1 = df.iloc[:, 0:800]
    df2 = df.iloc[:, 800:802]
    for _, item in df1.iteritems():
        i = pd.DataFrame()
        i['DATE'] = item.index.values
        i['CODE'] = item.name
        i['CLOSE'] = item.values
        result = result.append(i)

    df2 = df2.reset_index()
    data = result.merge(df2, left_on=['DATE'], right_on=['Date'], how='left')
    data = data[['CLOSE', 'CODE', 'DATE', '000906.SH', '000300.SH']]
    data.columns = ['CLOSE', 'CODE', 'DATE', '800_MARKET', '300_MARKET']
    data.to_csv('800_data.csv', index=False)

if __name__ == '__main__':
    # add_market_ratio()
    # read_excel()
    # gen_800_fft_data(config.ZZ800_FFT_DATA)
    # gen_800_fft_data(config.ZZ800_VALUE_RATIO_FFT_DATA)

    # gen_800_RM_VR_fft_data(config.ZZ800_RM_FFT)
    gen_800_RM_VR_fft_data(config.rootPath + '/data/800_rm_vr_fft1.csv')
