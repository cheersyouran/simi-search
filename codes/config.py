import pandas as pd
import time
import numpy as np
from datetime import timedelta
pd.set_option('display.width', 1000)

RAW_DATA_DIR = '../raw_data'
DATA = '../data/data.csv'
STD_DATA = '../data/std_data.csv'
ZZ800_DATA = '../data/800_data.csv'
ZZ800_STD_DATA = '../data/800_std_data.csv'
ZZ800_CODES = '../data/800_codes.csv'

code = '000001.SZ' # 被搜索股票的code

pattern_length = 30 # 被搜索股票的长度
start_date = pd.to_datetime('2016-01-01') # 被搜索股票的起始时间

regression_days = 60

# similarity_method = 'pearsonr'
similarity_method = 'euclidean'
nb_similarity = 2 # 返回target票数目
nb_data = 0 # 若设置为0，则targets用全部数据集

ascending_sort = True
WAVE = True