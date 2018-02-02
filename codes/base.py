#encoding:utf-8
from os import listdir, path
from codes.config import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pywt
import numpy as np
import math

def merge_raw_data():
    print('merge raw data...')
    col = ['CODE', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    merged_csv = pd.concat([pd.read_csv(path.join(RAW_DATA_DIR, f), header=None, names=col) for f in listdir(RAW_DATA_DIR)], axis=0)
    merged_csv = merged_csv.drop_duplicates()
    merged_csv.to_csv(DATA, index=False)
    merged_csv = pd.read_csv(DATA, parse_dates=['DATE'])
    merged_csv = merged_csv.sort_values(ascending=True, by=['CODE', 'DATE'])
    merged_csv.to_csv(DATA, index=False)

# 从所有数据中取出ZH800数据
def gen_800_data():
    print('gen 800 data...')
    codes = pd.read_csv(ZZ800_CODES)
    data = pd.read_csv(DATA)
    data = data[data['CODE'].isin(codes.values)]
    data.to_csv(ZZ800_DATA, index=False)

# 3级傅里叶变换(norm 处理)
def gen_800_fft_data(col='CLOSE'):
    print('gen 800 fft data...')

    def rolling_aply_fft(data, freq, method):
        data_ = norm(data)
        ffts = np.fft.fft(data_)/len(data_)
        if method == 'fft':
            return np.abs(ffts[freq])
        else:
            return np.rad2deg(np.angle(ffts[freq]))

    def apply(data, rolling_aply, freq, method):
        result = data.rolling(window=pattern_length).apply(func=rolling_aply, args=(freq, method,))
        return result

    data = pd.read_csv(ZZ800_DATA, parse_dates=['DATE'])
    for i in range(1, 4):
        ind = str(i)
        data['fft'+ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='fft')
        data['deg'+ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='deg')
    data.to_csv(ZZ800_FFT_DATA, index=False)

# 3级傅里叶变换(收益率 处理)
def gen_800_value_ratio_fft_data(col='CLOSE'):

    print('gen 800 value ratio fft data...')

    def rolling_aply_nav(data):
        global ratio_data
        if data.size < pattern_length:
            str_data = None
        else:
            data_ = data / data[0]
            str_data = np.array2string(data_, separator=',')[1:-1]
        ratio_data.append(str_data)
        return 1

    def ratio_apply(data, rolling_aply, min_periods=pattern_length):
        global ratio_data
        ratio_data = []
        data.rolling(window=pattern_length, min_periods=min_periods).apply(func=rolling_aply)
        result = pd.DataFrame(ratio_data, index=data.index)
        return result

    def fft_apply(data, freq, method):
        result = []
        for i, row in data.iteritems():
            if str(row) == 'nan':
                val = None
            else:
                data_ = np.fromstring(row, dtype=float, sep=',')
                ffts = np.fft.fft(data_) / len(data_)
                if method == 'fft':
                    val = np.abs(ffts[freq])
                else:
                    val = np.rad2deg(np.angle(ffts[freq]))
            result.append(val)
        res = pd.Series(result, index=data.index)
        return res

    data = pd.read_csv(ZZ800_DATA, parse_dates=['DATE'])
    ratio_data = []
    data['value_ratio'] = data.groupby(['CODE'])[col].apply(func=ratio_apply, rolling_aply=rolling_aply_nav, min_periods=1)
    data.to_csv(ZZ800_VALUE_RATIO_FFT_DATA, index=False)

    data = pd.read_csv(ZZ800_VALUE_RATIO_FFT_DATA, parse_dates=['DATE'])

    for i in range(1, 4):
        ind = str(i)
        data['fft'+ind] = data.groupby(['CODE'])['value_ratio'].apply(func=fft_apply, freq=i, method='fft')
        data['deg'+ind] = data.groupby(['CODE'])['value_ratio'].apply(func=fft_apply, freq=i, method='deg')

    data.to_csv(ZZ800_VALUE_RATIO_FFT_DATA, index=False)

# 先用小波变换给CLOSE降噪，再生成3级傅里叶变换结果
def gen_800_wave_fft_data(level, col='denoise_CLOSE'):
    print('gen 800 wave denoise data...')

    def apply(data):
        coeff = pywt.wavedec(data, 'db4', mode='sym', level=level)
        for i in range(0, level + 1):
            cD = coeff[i]
            if i in [0, level]:
                for j in range(len(cD)):
                    coeff[i][j] = 0
        waverec = pywt.waverec(coeff, 'db4')
        data_ = pd.Series(waverec[:-1], data.index)
        return data_

    data = pd.read_csv(ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    data['denoise_CLOSE'] = data.groupby(['CODE'])['CLOSE'].apply(func=apply)

    # 生成降噪后的数据
    data.to_csv(ZZ800_WAVE_DATA, index=False)

    # 用降噪后的数据生成3级傅里叶
    data = pd.read_csv(ZZ800_WAVE_DATA, parse_dates=['DATE'])
    for i in range(3):
        ind = str(i+1)
        data['fft' + ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=gen_800_fft_data.rolling_aply_fft, freq=i, method='fft')
        data['deg' + ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=gen_800_fft_data.rolling_aply_fft, freq=i, method='deg')
    data.to_csv(ZZ800_WAVE_FFT_DATA, index=False)

def norm(X):
    if speed_method in ['value_ratio_fft_euclidean', 'fft_euclidean']:
        result = X / pd.DataFrame(X).iloc[0][0]
    else:
        result = preprocessing.scale(X)
    return result

def norm1(X):
    # result = X - np.mean(X, axis=0)
    result = X / np.array(X)[0]
    return result

def plot_simi_stock(top, data, pattern, filename):
    print('plot simi stock...')
    def init_plot_data():
        plot_codes = np.append(top['CODE'].values, code)
        plot_dates = list(top['DATE'].values)
        plot_dates.append(None)
        plot_prices = np.zeros([nb_similarity + 1, pattern_length])
        plot_legend = list()
        return plot_codes, plot_dates, plot_prices, plot_legend

    plot_codes, plot_dates, plot_prices, plot_legend = init_plot_data()

    # 存储相似数据
    for i in range(nb_similarity):
        plot_quote = data[data['CODE'] == plot_codes[i]]
        plot_quote = plot_quote[plot_quote['DATE'] <= pd.to_datetime(plot_dates[i])].tail(pattern_length)
        plot_prices[i] = plot_quote['CLOSE'].values
        plot_legend.append(
            str(plot_codes[i]) + "," + similarity_method + ":" +
            str(top[top['CODE'] == plot_codes[i]][similarity_method].values))

    # 存储原数据
    plot_prices[-1] = pattern['CLOSE'].values
    plot_dates[-1] = pattern['DATE'].iloc[-1]
    plot_legend.append(str(plot_codes[-1]))
    norm_plot_prices = [norm(plot_prices[i]) for i in range(nb_similarity + 1)]

    print('股票价格:')
    print(plot_prices)

    print('股票norm:')
    print(norm_plot_prices)

    # 验证结果
    from codes.all_search import t_rol_aply
    for i in range(nb_similarity):
        a = t_rol_aply(plot_prices[i], plot_prices[-1])
        b = top.iloc[i][similarity_method]
        print(a, b)
        assert a == b, 'calcu error!'

    # 绘图
    line_styles = ['k--', 'k:', 'k-.', 'k--', 'k:', 'k-.', 'k:', 'k-.', 'k--', 'k:', 'k-.']
    for i in range(plot_codes.size):
        if i == plot_codes.size - 1:
            plt.plot(norm_plot_prices[i], 'r-', label=norm_plot_prices[i], linewidth=1.5)
        else:
            plt.plot(norm_plot_prices[i], line_styles[i], label=norm_plot_prices[i], linewidth=1.2)

    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend(plot_legend, loc='upper left')
    plt.title("Similarity Search[" + plot_codes[-1] + "]\n")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=45)
    plt.ioff()
    # plt.show()
    plt.savefig('../pic/' + filename+'.jpg')
    plt.close()

def plot_nav_curve(strategy_net_value, act_net_value, dates):
    print('plot nav curve...')
    plt.plot(dates, strategy_net_value, 'r-', label=strategy_net_value, linewidth=1.5)
    plt.plot(dates, act_net_value, 'k-', label=act_net_value, linewidth=1.5)

    plt.xlabel('Time')
    plt.ylabel('Net Asset Value')
    plt.legend(['strategy', 'baseline'], loc='upper left')
    plt.title("Net Asset Value")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=45)
    plt.xticks(fontsize=8)
    plt.ioff()
    # plt.show()
    plt.savefig('../pic/' + 'regression_result.jpg')
    plt.close()

if __name__ == '__main__':
    merge_raw_data()
    gen_800_data()
    gen_800_fft_data()
    # gen_800_wave_fft_data(level=2)
    # gen_800_value_ratio_fft_data()
    # data = pd.read_csv(ZZ800_VALUE_RATIO_FFT_DATA)
    # print('')
