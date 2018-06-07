from codes.config import config
import os
import pandas as pd
import numpy as np
import math
from codes.data_generator import update_data


class Market:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        print('Init Market! ', os.getpid())
        self.all_data = None
        self._init_all_data(config.speed_method)

        self.codes = None
        self._init_codes()

        self.trading_days = None
        self._init_trading_days()

        self.pattern = None
        self.targets = None
        self.current_date = config.start_date

    def _init_all_data(self, speed_method=config.speed_method):

        if speed_method == 'value_ratio_fft_euclidean':
            file = config.ZZ800_VALUE_RATIO_FFT_DATA
        elif speed_method == 'rm_market_vr_fft':
            file = config.ZZ800_RM_VR_FFT

        if self.all_data is None:
            print('Init All Data! ', os.getpid())
            if config.auto_update:
                rm_vr_data, _, __ = update_data()
                self.all_data = rm_vr_data
            else:
                self.all_data = pd.read_csv(file, parse_dates=['DATE'], low_memory=False)

            self.all_data['OPEN'] = 1

    def _init_codes(self):
        def apply(x):
            if int(x[0]) >= 6:
                return x + '.SH'
            else:
                return x + '.SZ'

        codes = pd.read_csv(config.ZZ800_CODES, dtype={'CODE': str})
        codes['CODE'] = codes['CODE'].apply(func=apply)
        self.codes = codes['CODE'].head(config.nb_codes).values.flatten()

    def _init_trading_days(self):

        self.trading_days = pd.read_csv(config.TRAINING_DAY, parse_dates=['DATE'])

    def _pass_a_day(self):
        self.current_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > self.current_date].head(1).values[0][0])
        config.start_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > config.start_date].head(1).values[0][0])

    def _pass_a_week(self):
        self.current_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > self.current_date].head(5).tail(1).values[0][0])
        config.start_date = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > config.start_date].head(5).tail(1).values[0][0])

    def get_historical_data(self, end_date=None, code=None):

        targets = self.all_data[self.all_data['CODE'] != code].reset_index(drop=True)
        start = self.trading_days[self.trading_days['DATE'] <= end_date].tail(30).head(1).values[0][0]

        def apply(x):
            return x.tail(config.slide_window)

        targets = targets[targets['DATE'] < start].groupby(['CODE']).apply(func=apply)

        self.pattern = self.all_data[(self.all_data['CODE'] == code) &
                                     (self.all_data['DATE'] <= end_date) &
                                     (self.all_data['DATE'] >= start)]

        self.targets = targets.dropna()

        if self.pattern.shape[0] == 0:
            return self.all_data, None, self.targets

        self.pattern = self.pattern.reset_index(drop=True)
        return self.all_data, self.pattern, self.targets

    def get_data(self, start_date=None, end_date=None, code=None, pattern_length=config.pattern_length):

        all_data = self.all_data

        if code is not None:
            if start_date is None and end_date is not None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date)].tail(pattern_length)
            elif start_date is not None and end_date is None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] >= start_date)].head(pattern_length)
            elif start_date is not None and end_date is not None:
                pattern = all_data[(all_data['CODE'] == code) & (all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]
        else:
            if start_date is None and end_date is not None:
                pattern = all_data[(all_data['DATE'] <= end_date)].tail(pattern_length)
            elif start_date is not None and end_date is None:
                pattern = all_data[(all_data['DATE'] >= start_date)].head(pattern_length)
            elif start_date is not None and end_date is not None:
                pattern = all_data[(all_data['DATE'] <= end_date) & (all_data['DATE'] >= start_date)]

        return pattern

    def pass_days(self, date, nb_day):
        date_ = pd.to_datetime(self.trading_days[self.trading_days['DATE'] > date].head(nb_day).tail(1).values[0][0])
        return date_

    def get_span_market_ratio(self, df, n):
        array = np.cumprod(df['300_RATIO'] / 100 + 1).values
        return 1 if math.isnan(array[n-1]) else array[n]

    def get_span_ret(self, df, n):
        array = np.cumprod(df['RET'] / 100 + 1).values
        return 1 if math.isnan(array[n-1]) else array[n]


market = Market()