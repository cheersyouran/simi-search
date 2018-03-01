#encoding:utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    elif config.speed_method in ['rm_vrfft_euclidean']:
        result = X / pd.DataFrame(X).iloc[0][0]
    elif config.speed_method in ['fft_euclidean']:
        result = preprocessing.scale(X)

    return result

def plot_simi_stock(top, data, pattern, filename, codes):
    print('plot simi stock of ', codes, ' ...')
    def init_plot_data():
        plot_codes = np.append(top['CODE'].values, codes)
        plot_dates = list(top['DATE'].values)
        plot_dates.append(None)
        plot_prices = np.zeros([config.nb_similar + 1, config.pattern_length])
        plot_legend = list()
        return plot_codes, plot_dates, plot_prices, plot_legend

    plot_codes, plot_dates, plot_prices, plot_legend = init_plot_data()

    for i in range(config.nb_similar):
        plot_quote = data[data['CODE'] == plot_codes[i]]
        plot_quote = plot_quote[plot_quote['DATE'] <= pd.to_datetime(plot_dates[i])].tail(config.pattern_length)
        plot_prices[i] = plot_quote['CLOSE'].values
        plot_legend.append(
            str(plot_codes[i]) + "," + config.similarity_method + ":" +
            str(top.iloc[i][config.similarity_method]))

    plot_prices[-1] = pattern['CLOSE'].values
    plot_dates[-1] = pattern['DATE'].iloc[-1]
    plot_legend.append(str(plot_codes[-1]))
    norm_plot_prices = [norm(plot_prices[i]) for i in range(config.nb_similar + 1)]

    if config.speed_method not in ['changed']:
        for i in range(config.nb_similar):
            a = weighted_distance(norm(plot_prices[i]), norm(plot_prices[-1]), config.pattern_length)
            b = top.iloc[i][config.similarity_method]
            print(a, b)
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
    plt.savefig(config.rootPath + '/pic/' + filename +'.png')
    plt.close()

def plot_nav_curve(strategy_net_value, act_net_value, market_net_value, dates, name):
    plt.plot(dates, strategy_net_value, 'r-', label=strategy_net_value, linewidth=1.5)
    plt.plot(dates, act_net_value, 'k-', label=act_net_value, linewidth=1.5)
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
    print('')
