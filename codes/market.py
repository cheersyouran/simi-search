from codes.config import config
import os
import pandas as pd

class Market:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        self.all_data = None
        self._init_all_data(config.speed_method)

        self.codes = None
        self._init_codes()

        self.ratios = None
        self._init_ratios()

        self.trading_days = None
        self._init_trading_days()

        self.pattern = None
        self.targets = None
        self.current_date = self.pass_days(config.start_date, config.pattern_length)
        self.col = 'CLOSE'

    def _init_all_data(self, speed_method=config.speed_method):

        if speed_method == 'fft_euclidean':
            file = config.ZZ800_FFT_DATA
        elif speed_method == 'wave_fft_euclidean':
            file = config.ZZ800_WAVE_FFT_DATA
            self.col = 'denoise_CLOSE'
        elif speed_method == 'value_ratio_fft_euclidean':
            file = config.ZZ800_VALUE_RATIO_FFT_DATA

        if self.all_data is None:
            print('Init all data! ', os.getpid())
            self.all_data = pd.read_csv(file, parse_dates=['DATE'], low_memory=False)

    def _init_ratios(self):
        self.ratios = pd.read_csv('../data/800_ratio.csv', parse_dates=['DATE'])

    def _init_codes(self):
        self.codes = pd.read_csv(config.ZZ800_CODES).head(config.nb_codes).values.flatten()

    def _init_trading_days(self):
        self.trading_days = pd.read_csv(config.ZZ800_TRAINING_DAY, parse_dates=['DATE'])

    def _pass_a_day(self):
        self.current_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > self.current_date].head(1).values[0][0])
        config.start_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > config.start_date].head(1).values[0][0])

    # 更新pattern, targets
    def get_historical_data(self, start_date=None, end_date=None, speed_method=config.speed_method, code=config.code):

        self.targets = self.all_data[self.all_data['CODE'] != code].reset_index(drop=True)

        if start_date == None and end_date != None:
            self.pattern = self.all_data[(self.all_data['CODE'] == code) & (self.all_data['DATE'] <= end_date)].tail(config.pattern_length)
        elif start_date != None and end_date == None:
            self.pattern = self.all_data[(self.all_data['CODE'] == code) & (self.all_data['DATE'] >= start_date)].head(config.pattern_length)
        elif start_date != None and end_date != None:
            self.pattern = self.all_data[(self.all_data['CODE'] == code) & (self.all_data['DATE'] <= end_date) & (self.all_data['DATE'] >= start_date)]

        self.pattern = self.pattern.reset_index(drop=True)

        if config.nb_data != 0:
            self.targets = self.targets.head(config.nb_data)

        return self.all_data, self.pattern, self.targets, self.col

    def get_data(self, start_date=None, end_date=None, code=None, pattern_length=config.pattern_length):

        all_data = self.all_data

        if code is not None:
            if start_date == None and end_date != None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date)].tail(pattern_length)
            elif start_date != None and end_date == None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] >= start_date)].head(pattern_length)
            elif start_date != None and end_date != None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]
        else:
            if start_date == None and end_date != None:
                pattern = all_data[(all_data['DATE'] <= end_date)].tail(pattern_length)
            elif start_date != None and end_date == None:
                pattern = all_data[(all_data['DATE'] >= start_date)].head(pattern_length)
            elif start_date != None and end_date != None:
                pattern = all_data[(all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]

        pattern = pattern.reset_index(drop=True)

        return pattern

    def pass_days(self, date, nb_day):
        date_ = pd.to_datetime(self.trading_days[self.trading_days['DATE'] >= date].head(nb_day).tail(1).values[0][0])
        return date_

market = Market()
