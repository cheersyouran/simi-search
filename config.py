import pandas as pd

code = '000001.SZ' # 被搜索股票的code
pattern_length = 30 # 被搜索股票的长度
start_date = pd.to_datetime('2017-01-01') # 被搜索股票的起始时间

nb_similarity = 2 # 返回相似股票数目
similarity_method = 'pearsonr'
nb_data = 10000 # 若设置为0，则用全部数据集

ascending_sort = False