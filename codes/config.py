import pandas as pd
import numpy as np
import os
from datetime import timedelta

pd.set_option('display.width', 1200)
pd.set_option('precision', 3)
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)

class Config:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        print('Init config!', os.getpid())
        self.RAW_DATA_DIR = '../raw_data'
        self.DATA = '../data/data.csv'

        self.ZZ800_CODES = '../data/800_codes.csv'
        self.ZZ800_DATA = '../data/800_data.csv'
        # self.ZZ800_WAVE_DATA = '../data/800_wave_data.csv'
        self.ZZ800_FFT_DATA = '../data/800_fft_data.csv'
        self.ZZ800_WAVE_FFT_DATA = '../data/800_wave_fft_data.csv'
        self.ZZ800_VALUE_RATIO_FFT_DATA = '../data/800_value_ratio_fft_data.csv'  # 尤老师方法
        self.ZZ800_TRAINING_DAY = '../data/800_training_day.csv'

        self.code = '000001.SZ'
        self.nb_codes = 1

        self.pattern_length = 30
        self.regression_days = 250
        self.start_date = pd.to_datetime('2017-02-24')
        # self.start_date = pd.to_datetime('2016-01-01')
        self.end_date = self.start_date + timedelta(days=self.regression_days)

        self.speed_method = 'fft_euclidean' # 'value_ratio_fft_euclidean'
        self.fft_level = 3
        self.similarity_method = 'euclidean' #'pearsonr'

        self.nb_similarity = 2  # 返回相似序列的数目
        self.nb_to_make_action = 2 # 用于做action的codes的数目
        self.nb_data = 0  # 若为0，则targets用全部数据集

        self.weighted_dist = True

config = Config()

if __name__ == '__main__':
    std_data = pd.read_csv(config.DATA)
