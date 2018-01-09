#encoding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from os import listdir, path
from config import *

RAW_DATA_DIR = './raw_data'
DATA = './data/data.csv'

def merge_raw_data():
    col = ['CODE', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    mergeed_csv = pd.concat([pd.read_csv(path.join(RAW_DATA_DIR, f), header=None, names=col) for f in listdir(RAW_DATA_DIR)], axis=0)
    mergeed_csv.to_csv("./data/data.csv", index=False)

def load_data():
    print('load data....')
    df = pd.read_csv(DATA, parse_dates=['DATE'], low_memory=False)
    return df

def norm(X):
    X = preprocessing.scale(X)
    # X -= np.mean(X, axis=0)
    return X

def test_data(date, time_span, df):
    return df[df['DATE'] < pd.to_datetime(date)].tail(time_span)

def plot_stocks_price_plot(most, data, pattern, normal=True):

    def init_plot_data():
        plot_codes = np.append(most['CODE'].values, code)
        plot_dates = np.append(most['DATE'].values, None)
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
            str(most[most['CODE'] == plot_codes[i]][similarity_method].values))

    # 存储原数据
    plot_prices[-1] = pattern['CLOSE'].values
    plot_dates[-1] = pattern['DATE'].iloc[-1]
    plot_legend.append(str(plot_codes[-1]))

    # 验证结果
    for i in range(nb_similarity):
        import search
        print(search.t_rol_aply(plot_prices[i], plot_prices[-1]), similarity_method)
        assert search.t_rol_aply(plot_prices[i], plot_prices[-1]) == most[most['CODE'] == plot_codes[i]][
            similarity_method].values, 'calcu error!'

    # 绘图
    line_styles = ['k--', 'k:', 'k-.']
    for i in range(plot_codes.size):
        if normal == True:
            plot_prices[i] = norm(plot_prices[i])
        if i == plot_codes.size - 1:
            plt.plot(plot_prices[i], 'r-', label=plot_prices[i], linewidth=1.5)
        else:
            plt.plot(plot_prices[i], line_styles[i], label=plot_prices[i], linewidth=1.2)

    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend(plot_legend, loc='upper right')
    plt.title("Similarity Search["+ plot_codes[-1] +"]\n")
    plt.grid(True)
    # plt.xticks(fontsize=8, rotation=45)
    plt.ioff()
    plt.show()
    plt.close()

if __name__ == '__main__':
    # merge_raw_data()
    df = load_data().head(50)
    print(df)
