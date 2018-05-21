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
        print('Init Config!', os.getpid())

        # 文件路径相关参数
        self.rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
        self.ZZ800_DATA = self.rootPath + '/data/800_data.csv'
        self.ZZ800_RAW_DATA = self.rootPath + '/data/800_raw_data.csv'
        self.ZZ800_CODES = self.rootPath + '/data/800_codes.csv'
        self.ZZ800_RM_VR_FFT = self.rootPath + '/data/800_rm_vr_fft.csv'
        self.MARKET_RATIO = self.rootPath + '/data/index_ratio.csv'
        self.TRAINING_DAY = self.rootPath + '/data/trading_day.csv'

        # self.speed_method = 'value_ratio_fft_euclidean' # for 沪深300指数预测
        self.speed_method = 'rm_market_vr_fft' # for 沪深800选股

        self.update_start = '2018-01-01'  # 更新数据的开始时间
        self.update_end = '2018-05-18'  # 更新数据的结束时间

        self.start_date = pd.to_datetime('2018-05-16') #回测的开始时间。 比如'2018-01-01'，则从'2018-01-02'开始做预测
        self.regression_days = 5
        self.regression_end_date = self.start_date + timedelta(days=self.regression_days) # 回测结束时间

        self.auto_update = False # 回测时是否自动更新数据
        self.plot_simi_stock = False # 是否画出相似股票
        self.is_regression_test = False # 是回测还是预测

        # 相似性查找参数
        self.pattern_length = 30

        self.nb_similar_make_prediction = 20  # avergae them as a pred
        self.nb_similar_of_all_similar = 4000  # 从所有股票的相似票中选择top N
        self.nb_similar_of_each_stock = 200

        self.slide_window = 1500

        self.weighted_dist = True
        self.weight_a = 1
        self.weight_b = 2
        self.alpha = np.multiply([1, 1, 1, 1, 1], 40)
        self.beata = np.multiply([1, 1, 1, 1, 1], math.pi / 180)

        self.fft_level = 5
        self.similarity_method = 'euclidean'  # or 'pearsonr'

        self.cores = 20
        self.nb_codes = 800

        # 输出文件地址
        name = str(self.start_date.date()) + '_' + str(self.speed_method) + '_' + str(self.nb_similar_make_prediction)
        self.PEARSON_CORR_RESLUT = self.rootPath + '/output/corr' + name + '.csv'
        self.PRDT_AND_ACT_RESULT = self.rootPath + '/output/pred' + name +'.csv'
        self.regression_result = self.rootPath + '/pic/para_' + name + '.png'

config = Config()

if __name__ == '__main__':
    std_data = pd.read_csv(config.DATA)
