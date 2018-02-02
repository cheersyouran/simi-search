import pandas as pd
import time
import numpy as np
from datetime import timedelta
pd.set_option('display.width', 1000)

RAW_DATA_DIR = '../raw_data'
DATA = '../data/data.csv'

ZZ800_CODES = '../data/800_codes.csv'
ZZ800_DATA = '../data/800_data.csv'
ZZ800_WAVE_DATA = '../data/800_wave_data.csv'
ZZ800_FFT_DATA = '../data/800_fft_data.csv'
ZZ800_WAVE_FFT_DATA = '../data/800_wave_fft_data.csv'
ZZ800_VALUE_RATIO_FFT_DATA = '../data/800_value_ratio_fft_data.csv' # 尤老师方法

# 'fft_euclidean', 'wave_fft_euclidean' ,'value_ratio_fft_euclidean'
speed_method = 'value_ratio_fft_euclidean'
# speed_method = 'fft_euclidean'

code = '000001.SZ' # 被搜索股票的code

pattern_length = 30 # 被搜索股票的长度
start_date = pd.to_datetime('2016-01-01') # 被搜索股票的起始时间

regression_days = 250

# similarity_method = 'pearsonr'
similarity_method = 'euclidean'
nb_similarity = 2 # 返回target票数目
nb_data = 0 # 若设置为0，则targets用全部数据集

if __name__ == '__main__':
    std_data = pd.read_csv(DATA)
