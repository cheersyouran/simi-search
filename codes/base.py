#encoding:utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from os import listdir, path
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from codes.config import *

def weighted_distance(x, y, length):
    if config.weighted_dist == True:
        weight = np.arange(1, 2, 1 / length)
        dist = np.abs(np.multiply(pd.DataFrame(x).values - pd.DataFrame(y).values, weight.reshape(30, 1)))
        return np.sum(dist)
    else:
        return euclidean(x, y)

def norm(X):
    if config.speed_method in ['value_ratio_fft_euclidean', 'changed']:
        result = X / pd.DataFrame(X).iloc[0][0]
    else:
        result = preprocessing.scale(X)
    return result

def merge_raw_data():
    print('merge raw data...')
    col = ['CODE', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    merged_csv = pd.concat([pd.read_csv(path.join(config.RAW_DATA_DIR, f), header=None, names=col) for f in listdir(config.RAW_DATA_DIR)], axis=0)
    merged_csv = merged_csv.drop_duplicates()
    merged_csv.to_csv(config.DATA, index=False)
    merged_csv = pd.read_csv(config.DATA, parse_dates=['DATE'])
    merged_csv = merged_csv.sort_values(ascending=True, by=['CODE', 'DATE'])
    merged_csv.to_csv(config.DATA, index=False)

def gen_trading_days():
    df = pd.read_csv(config.ZZ800_DATA)
    pd.DataFrame(df['DATE'].unique()).to_csv(config.ZZ800_TRAINING_DAY, index=False)

def gen_800_data():
    print('gen 800 data...')
    codes = pd.read_csv(config.ZZ800_CODES)
    data = pd.read_csv(config.DATA)
    data = data[data['CODE'].isin(codes.values)]
    data.to_csv(config.ZZ800_DATA, index=False)

def gen_800_fft_data(col='CLOSE'):
    print('gen 800 fft data...')

    def rolling_aply_fft(data, freq, method):
        data_ = norm(data)
        ffts = np.fft.fft(data_)/len(data_)
        if method == 'fft':
            return np.abs(ffts[freq])
        elif method == 'deg':
            return np.rad2deg(np.angle(ffts[freq]))

    def apply(data, rolling_aply, freq, method):
        result = data.rolling(window=config.pattern_length).apply(func=rolling_aply, args=(freq, method,))
        return result

    data = pd.read_csv(config.ZZ800_DATA, parse_dates=['DATE'])
    # if config.speed_method == 'fft_euclidean':
    #     data = pd.read_csv(config.ZZ800_FFT_DATA, parse_dates=['DATE'])
    # else:
    #     data = pd.read_csv(config.ZZ800_VALUE_RATIO_FFT_DATA, parse_dates=['DATE'])

    for i in range(config.fft_level):
        ind = str(i+1)
        data['fft'+ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='fft')
        data['deg'+ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=rolling_aply_fft, freq=i, method='deg')

    if config.speed_method == 'fft_euclidean':
        data.to_csv(config.ZZ800_FFT_DATA, index=False)
    else:
        data.to_csv(config.ZZ800_VALUE_RATIO_FFT_DATA, index=False)

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

    data = pd.read_csv(config.ZZ800_DATA, parse_dates=['DATE'], low_memory=False)
    data['denoise_CLOSE'] = data.groupby(['CODE'])['CLOSE'].apply(func=apply)

    # 生成降噪后的数据
    data.to_csv(config.ZZ800_WAVE_FFT_DATA, index=False)

    # 用降噪后的数据生成3级傅里叶
    data = pd.read_csv(config.ZZ800_WAVE_FFT_DATA, parse_dates=['DATE'])
    for i in range(3):
        ind = str(i+1)
        data['fft' + ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=gen_800_fft_data.rolling_aply_fft, freq=i, method='fft')
        data['deg' + ind] = data.groupby(['CODE'])[col].apply(func=apply, rolling_aply=gen_800_fft_data.rolling_aply_fft, freq=i, method='deg')
    data.to_csv(config.ZZ800_WAVE_FFT_DATA, index=False)

def plot_simi_stock(top, data, pattern, filename, codes=config.code):
    print('plot simi stock of ', codes, ' ...')
    def init_plot_data():
        plot_codes = np.append(top['CODE'].values, codes)
        plot_dates = list(top['DATE'].values)
        plot_dates.append(None)
        plot_prices = np.zeros([config.nb_similarity + 1, config.pattern_length])
        plot_legend = list()
        return plot_codes, plot_dates, plot_prices, plot_legend

    plot_codes, plot_dates, plot_prices, plot_legend = init_plot_data()

    for i in range(config.nb_similarity):
        plot_quote = data[data['CODE'] == plot_codes[i]]
        plot_quote = plot_quote[plot_quote['DATE'] <= pd.to_datetime(plot_dates[i])].tail(config.pattern_length)
        plot_prices[i] = plot_quote['CLOSE'].values
        plot_legend.append(
            str(plot_codes[i]) + "," + config.similarity_method + ":" +
            str(top.iloc[i][config.similarity_method]))

    plot_prices[-1] = pattern['CLOSE'].values
    plot_dates[-1] = pattern['DATE'].iloc[-1]
    plot_legend.append(str(plot_codes[-1]))
    norm_plot_prices = [norm(plot_prices[i]) for i in range(config.nb_similarity + 1)]

    # print('股票价格:')
    # print(plot_prices)

    # print('股票norm:')
    # print(norm_plot_prices)

    # 验证结果
    from codes.all_search import t_rol_aply
    if config.speed_method not in ['changed']:
        for i in range(config.nb_similarity):
            a = t_rol_aply(plot_prices[i], plot_prices[-1])
            b = top.iloc[i][config.similarity_method]
            # print(a, b)
            assert a == b, 'calcu error!'

    line_styles = ['k--', 'k:', 'k-.', 'k--', 'k:', 'k-.', 'k:', 'k-.', 'k--', 'k:', 'k-.']
    for i in range(plot_codes.size):
        if i == plot_codes.size - 1:
            plt.plot(norm_plot_prices[i], 'r-', label=norm_plot_prices[i], linewidth=1.5)
        else:
            plt.plot(norm_plot_prices[i], line_styles[i], label=norm_plot_prices[i], linewidth=1.2)

    plt.xlabel('Time')
    if config.speed_method == 'value_ratio_fft_euclidean':
        plt.ylabel('NAV')
    else:
        plt.ylabel('Close Price')
    plt.legend(plot_legend, loc='upper left')
    plt.title("Similarity Search[" + plot_codes[-1] + "]\n")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=45)
    plt.ioff()
    # plt.show()
    plt.savefig(config.rootPath + '/pic/' + filename+'.png')
    plt.close()

def plot_nav_curve(strategy_net_value, act_net_value, market_net_value, dates, name):
    plt.plot(dates, strategy_net_value, 'r-', label=strategy_net_value, linewidth=1.5)
    # plt.plot(dates, act_net_value, 'k-', label=act_net_value, linewidth=1.5)
    plt.plot(dates, market_net_value, 'g-', label=market_net_value, linewidth=1.5)

    plt.xlabel('Time')
    plt.ylabel('Net Asset Value')
    plt.legend(['strategy', 'baseline', 'market'], loc='upper left')
    plt.title("Net Asset Value")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=20)
    plt.xticks(fontsize=8)
    plt.ioff()
    # plt.show()
    plt.savefig(config.rootPath + '/pic/' + name + '.png')
    plt.close()

def compare_plot(x1, x2):
    plt.plot(x1)
    plt.plot(x2)
    plt.grid(True)
    plt.ioff()
    plt.savefig('../pic/' + 'compare_result.jpg')
    plt.close()

if __name__ == '__main__':
    # merge_raw_data()
    # gen_800_data()
    gen_trading_days()

    # config.speed_method = 'fft_euclidean'
    # gen_800_fft_data()

    # config.speed_method = 'value_ratio_fft_euclidean'
    # gen_800_fft_data()
    # gen_800_wave_fft_data(level=2)
    # print('')
