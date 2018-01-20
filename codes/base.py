#encoding:utf-8
from os import listdir, path
from codes.config import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pywt

def merge_raw_data():
    print('merge raw data...')
    col = ['CODE', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    mergeed_csv = pd.concat([pd.read_csv(path.join(RAW_DATA_DIR, f), header=None, names=col) for f in listdir(RAW_DATA_DIR)], axis=0)
    mergeed_csv.to_csv("./data/data.csv", index=False)

def gen_std_data():
    def rolling_aply(data):
        if WAVE == True:
            data_ = preprocessing.scale(data)
            coeff = pywt.wavedec(data_, 'db4', mode='sym', level=2)
            for i in range(3):
                cD = coeff[i]
                if i not in [0]:
                    for j in range(len(cD)):
                        coeff[i][j] = 0
            denoised_close = pywt.waverec(coeff, 'db4')
        else:
            denoised_close = data
        # mean = np.mean(data, axis=0)
        std = np.std(denoised_close, axis=0)
        return std

    def apply(data):
        result = data.rolling(window=pattern_length).apply(func=rolling_aply)
        return result

    print('pre-process data...')
    data = pd.read_csv(DATA, parse_dates=['DATE'], low_memory=False)
    # data['CLOSE'] = (data['CLOSE'] - np.min(data['CLOSE'])) / (np.max(data['CLOSE']) - np.min(data['CLOSE']))
    # data['std'] = data.groupby(['CODE'])['CLOSE'].apply(func=apply)

    data['CLOSE_NORM'] = data["CLOSE"]

    data['std'] = data.groupby(['CODE'])['CLOSE_NORM'].apply(func=apply)
    data = data.drop(['CLOSE_NORM'], axis=1)

    data.to_csv(STD_DATA, index=False)

def gen_800_data():
    codes = pd.read_csv(ZZ800_CODES)
    data = pd.read_csv(DATA)
    data = data[data['CODE'].isin(codes.values)]
    data.to_csv(ZZ800_DATA, index=False)

def gen_800_std_data():
    codes = pd.read_csv(ZZ800_CODES)
    std_data = pd.read_csv(STD_DATA)
    std_data = std_data[std_data['CODE'].isin(codes.values)]
    std_data.to_csv(ZZ800_STD_DATA, index=False)

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

    # 验证结果
    from codes import all_search
    for i in range(nb_similarity):
        a = all_search.t_rol_aply(plot_prices[i], plot_prices[-1])
        b = top[top['CODE'] == plot_codes[i]][similarity_method].values
        print(a, b)
        # assert a == b, 'calcu error!'

    # 绘图
    line_styles = ['k--', 'k:', 'k-.']
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
    # plt.xticks(fontsize=8, rotation=45)
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
    # plt.xticks(fontsize=8, rotation=45)
    plt.xticks(fontsize=8)
    plt.ioff()
    # plt.show()
    plt.savefig('../pic/' + 'result.jpg')
    plt.close()

if __name__ == '__main__':
    # merge_raw_data()
    # gen_800_data()

    gen_std_data()
    gen_800_std_data()
    print('base main function...')
