#encoding:utf-8
from os import listdir, path
from codes.config import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pywt
import numpy as np

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
    codes = pd.read_csv(ZZ800_CODES)
    data = pd.read_csv(DATA)
    data = data[data['CODE'].isin(codes.values)]
    data.to_csv(ZZ800_DATA, index=False)

# 尤教授建议的
def gen_800_nav_std_data():
    print('pre process 800 wave std data...')

    def rolling_aply(data):
        std = np.std(data, axis=0)
        return std

    def apply(data):
        data_ = data / data.iloc[0]
        result = data_.rolling(window=pattern_length).apply(func=rolling_aply)
        return result

    data = pd.read_csv(ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    data['std'] = data.groupby(['CODE'])['CLOSE'].apply(func=apply)
    data.to_csv(ZZ800_NAV_STD_DATA, index=False)

def gen_800_wave_std_data():
    print('pre process 800 wave std data...')
    def rolling_aply(data):
        std = np.std(data, axis=0)
        return std

    def apply(data):
        result = data.rolling(window=pattern_length).apply(func=rolling_aply)
        return result

    data = pd.read_csv(ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    data['std'] = data.groupby(['CODE'])['CLOSE'].apply(func=apply)
    data.to_csv(ZZ800_STD_DATA, index=False)

# 3级傅里叶变换结果
def gen_800_fft_data(org_data=ZZ800_DATA, det_data=ZZ800_FFT_DATA, col='CLOSE'):
    print('pre process 800 fft data...')

    def rolling_aply_fft(data, freq, method):
        data_ = preprocessing.scale(data)
        ffts = np.fft.fft(data_)/len(data_)
        if method == 'fft':
            return np.abs(ffts[freq])
        else:
            return np.rad2deg(np.angle(ffts[freq]))

    def apply(data, rolling_aply, freq, method):
        result = data.rolling(window=pattern_length).apply(func=rolling_aply, args=(freq, method,))
        return result

    data = pd.read_csv(org_data, parse_dates=['DATE'])

    data['fft1'] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=1, method='fft')
    data['deg1'] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=1, method='deg')
    data['fft2'] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=2, method='fft')
    data['deg2'] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=2, method='deg')
    data['fft3'] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=3, method='fft')
    data['deg3'] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=3, method='deg')

    data.to_csv(det_data, index=False)

# 用小波变换给CLOSE降噪，并生成3级傅里叶变换结果
def gen_800_wave_fft_data(level):
    print('gen_800_wave_denoise_data...')
    level = 2
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
    gen_800_fft_data(org_data=ZZ800_WAVE_DATA, det_data=ZZ800_WAVE_FFT_DATA, col='denoise_CLOSE')

def norm(X):
    result = preprocessing.scale(X)
    # result = X - np.mean(X, axis=0)
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

    print('股票价格:')
    print(plot_prices)

    # 验证结果
    from codes import all_search
    for i in range(nb_similarity):
        a = all_search.t_rol_aply(plot_prices[i], plot_prices[-1])
        b = top[top['CODE'] == plot_codes[i]][similarity_method].values
        print(a, b)
        assert a == b, 'calcu error!'

    # 绘图
    line_styles = ['k--', 'k:', 'k-.', 'k--', 'k:', 'k-.']
    for i in range(plot_codes.size):
        plot_prices[i] = norm(plot_prices[i])
        if i == plot_codes.size - 1:
            plt.plot(plot_prices[i], 'r-', label=plot_prices[i], linewidth=1.5)
        else:
            plt.plot(plot_prices[i], line_styles[i], label=plot_prices[i], linewidth=1.2)

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
    plt.savefig('../pic/' + 'result.jpg')
    plt.close()

if __name__ == '__main__':
    merge_raw_data()
    gen_800_data()
    # gen_800_fft_data()
    # gen_800_nav_std_data()
    # gen_800_wave_std_data()

    gen_800_wave_fft_data(level=2)
