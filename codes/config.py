import pandas as pd
import numpy as np
import os
from datetime import timedelta
import math

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
        self.ZZ800_RAW_DATA = self.rootPath + '/data/800_raw_data.csv'
        self.ZZ800_FFT_DATA = self.rootPath + '/data/800_fft_data.csv'
        self.ZZ800_VALUE_RATIO_FFT_DATA = self.rootPath + '/data/800_value_ratio_fft_data.csv'
        self.ZZ800_CODES = self.rootPath + '/data/800_codes.csv'
        self.ZZ800_RM_FFT = self.rootPath + '/data/800_rm_fft.csv'
        self.ZZ800_RM_VR_FFT = self.rootPath + '/data/800_rm_vr_fft.csv'

        self.MARKET_RATIO = self.rootPath + '/data/index_ratio.csv'
        self.HS300_VALUE_RATIO_FFT_DATA = self.rootPath + '/data/300_value_ratio_fft_data.csv'
        self.HS300_FFT_DATA = self.rootPath + '/data/300_fft_data.csv'
        self.HS300_CODES = self.rootPath + '/data/300_codes.csv'
        self.TRAINING_DAY = self.rootPath + '/data/trading_day.csv'

        self.speed_method = 'fft_euclidean' # 800and300; normalization
        self.speed_method = 'value_ratio_fft_euclidean' # 沪深300指数预测
        # self.speed_method = 'rm_market_fft'
        self.speed_method = 'rm_market_vr_fft' #value_ratio 除去市场
        self.market_index = 800

        self.code = '000001.SZ'
        self.nb_similar_make_prediction = 20 # avergae them as a pred
        self.nb_similar_of_all_similar = 4000 # 从所有股票的相似票中选择top N
        self.nb_similar_of_each_stock = 200

        self.pattern_length = 30
        self.regression_days = 3000

        # 之前的最后一天
        self.update_start = '2017-12-29'
        self.update_end = '2018-03-07'
        self.start_date = pd.to_datetime('2018-03-05')
        self.regression_end_date = self.start_date + timedelta(days=self.regression_days)

        self.fft_level = 5
        self.similarity_method = 'euclidean' #'pearsonr'

        self.market_ratio_type = '300_RATIO' if self.market_index == 300 else '800_RATIO'
        self.nb_codes = 300 if self.market_index == 300 else 800

        self.auto_update = False
        self.plot_simi_stock = False
        self.is_regression_test = False
        self.weighted_dist = True
        self.weight_a = 1
        self.weight_b = 2
        self.alpha = np.multiply([1, 1, 1, 1, 1], 40)
        self.beata = np.multiply([1, 1, 1, 1, 1], math.pi / 180)
        self.cores = 100

        name = str(self.start_date.date()) + '_' + str(self.speed_method) + '_' + \
               str(self.market_index) + '_' + str(self.nb_similar_make_prediction)

        self.PEARSON_CORR_RESLUT = self.rootPath + '/output/corr' + name + '.csv'
        self.PRDT_AND_ACT_RESULT = self.rootPath + '/output/pred' + name +'.csv'
        self.regression_result = self.rootPath + '/pic/para_' + name + '.png'

config = Config()

if __name__ == '__main__':
    std_data = pd.read_csv(config.DATA)
