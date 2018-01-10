import pandas as pd
import numpy as np

RAW_DATA_DIR = './raw_data'
DATA = './data/data.csv'


code = '000001.SZ' # 被搜索股票的code
pattern_length = 30 # 被搜索股票的长度
start_date = pd.to_datetime('2017-01-03') # 被搜索股票的起始时间

regression_days = 30

# similarity_method = 'pearsonr'
similarity_method = 'euclidean'
nb_similarity = 2 # 返回target票数目
nb_data = 500000 # 若设置为0，则targets用全部数据集

ascending_sort = False