import pandas as pd
import numpy as np

pd.set_option('display.width', 1200)
pd.set_option('precision', 3)
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

class Config:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self.RAW_DATA_DIR = '../raw_data'
        self.DATA = '../data/data.csv'

        self.ZZ800_CODES = '../data/800_codes.csv'
        self.ZZ800_DATA = '../data/800_data.csv'
        # self.ZZ800_WAVE_DATA = '../data/800_wave_data.csv'
        self.ZZ800_FFT_DATA = '../data/800_fft_data.csv'
        self.ZZ800_WAVE_FFT_DATA = '../data/800_wave_fft_data.csv'
        self.ZZ800_VALUE_RATIO_FFT_DATA = '../data/800_value_ratio_fft_data.csv'  # 尤老师方法

        self.code = '000001.SZ'

        self.pattern_length = 30
        self.start_date = pd.to_datetime('2015-01-01')  # 被搜索股票的起始时间

        self.speed_method = 'value_ratio_fft_euclidean'
        # self.speed_method = 'fft_euclidean'

        self.regression_days = 250
        self.regression_days = 0

        # self.similarity_method = 'pearsonr'
        self.similarity_method = 'euclidean'

        self.nb_similarity = 2  # 返回相似序列的数目
        self.nb_data = 0  # 若为0，则targets用全部数据集

        self.weighted_dist = False
        self.weighted_dist = True

config = Config()

if __name__ == '__main__':
    std_data = pd.read_csv(config.DATA)
