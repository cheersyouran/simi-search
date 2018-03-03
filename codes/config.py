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
        self.rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]

        self.RAW_DATA_DIR = self.rootPath + '/data/raw_data'
        self.DATA = self.rootPath + '/data/data.csv'

        self.ZZ800_DATA = self.rootPath + '/data/800_data.csv'
        self.ZZ800_FFT_DATA = self.rootPath + '/data/800_fft_data.csv'
        self.ZZ800_VALUE_RATIO_FFT_DATA = self.rootPath + '/data/800_value_ratio_fft_data.csv'
        self.ZZ800_MARKET_RATIO = self.rootPath + '/data/800_ratio.csv'
        self.ZZ800_CODES = self.rootPath + '/data/800_codes.csv'
        self.ZZ800_RM_VR_FFT = self.rootPath + '/data/800_rm_vr_fft.csv'

        self.HS300_MARKET_RATIO = self.rootPath + '/data/300_ratio.csv'
        self.HS300_VALUE_RATIO_FFT_DATA = self.rootPath + '/data/300_value_ratio_fft_data.csv'
        self.HS300_FFT_DATA = self.rootPath + '/data/300_fft_data.csv'
        self.HS300_CODES = self.rootPath + '/data/300_codes.csv'

        self.TRAINING_DAY = self.rootPath + '/data/training_day.csv'

        self.market_index = 800
        self.market_ratio_type = '300_RATIO' if self.market_index == 300 else '800_RATIO'

        self.nb_codes = 300
        self.code = '000001.SZ'
        self.nb_similar = 5 # avergae them as result
        self.nb_similar_of_each_stock = 200

        self.pattern_length = 30
        self.regression_days = 1000
        self.start_date = pd.to_datetime('2014-06-04')
        self.regression_end_date = self.start_date + timedelta(days=self.regression_days)

        self.speed_method = 'fft_euclidean'
        self.speed_method = 'value_ratio_fft_euclidean'
        self.speed_method = 'rm_vrfft_euclidean'

        self.fft_level = 3
        self.similarity_method = 'euclidean' #'pearsonr'

        self.nb_data = 0
        self.above_ratio = 0.00

        self.weighted_dist = True
        self.weekily_regression = True

        name = str(self.market_index) + '_' + str(self.speed_method) + '_' + str(self.nb_similar)
        self.PEARSON_CORR_RESLUT = self.rootPath + '/output/corr' + name + '.csv'
        self.PRDT_AND_ACT_RESULT = self.rootPath + '/output/pred' + name +'.csv'
        self.regression_result = self.rootPath + '/pic/para_' + name + '.png'


config = Config()

if __name__ == '__main__':
    std_data = pd.read_csv(config.DATA)
