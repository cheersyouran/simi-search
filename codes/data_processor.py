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

def gen_trading_days():
    df = pd.read_csv(config.ZZ800_DATA)
    pd.DataFrame(df['DATE'].unique()).to_csv(config.ZZ800_TRAINING_DAY, index=False)

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
        data['fft'+ind] = data.groupby(['CODE'])['CLOSE'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='fft')
        data['deg'+ind] = data.groupby(['CODE'])['CLOSE'].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='deg')
        print(ind)

    data.to_csv(path, index=False)

if __name__ == '__main__':
    # add_market_ratio()
    # read_excel()
    config.speed_method = 'fft_euclidean'
    gen_800_fft_data(config.rootPath + '/data/800_fft_data1.csv')

    config.speed_method = 'value_ratio_fft_euclidean'
    gen_800_fft_data(config.rootPath + '/data/800_value_ratio_fft_data1.csv')

    # gen_800_RM_VR_fft_data(config.ZZ800_RM_FFT)
    # gen_800_RM_VR_fft_data(config.rootPath + '/data/800_rm_vr_fft1.csv')

    print('')
