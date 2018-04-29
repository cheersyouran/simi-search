#encoding:utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn import preprocessing
from codes.config import *

def weighted_distance(x, y, length=config.pattern_length):
    if config.weighted_dist == True:
        weight = np.arange(config.weight_a, config.weight_b, (config.weight_b - config.weight_a) / length)
        dist = np.abs(np.multiply(pd.DataFrame(x).values - pd.DataFrame(y).values, weight.reshape(length, 1)))
        return np.sum(dist)
    else:
        return euclidean(x, y)

def norm(X, ratio=None):
    if ratio is None:
        raise Exception('No Ratios!')
    ret_ = X + 1
    ratio_ = (ratio / 100) + 1
    r = ratio_ / pd.DataFrame(ratio_).iloc[0][0]
    result = np.divide(ret_, r)
    return result

def plot_simi_stock(top, data, pattern, filename, codes):
    print('plot simi stock of ', codes, ' ...')
    def init_plot_data():
        plot_codes = np.append(top['CODE'].values, codes)
        plot_dates = list(top['DATE'].values)
        plot_dates.append(None)
        plot_prices = np.zeros([config.nb_similar_make_prediction + 1, config.pattern_length])
        plot_legend = list()
        plot_market_ratio = np.zeros([config.nb_similar_make_prediction + 1, config.pattern_length])
        return plot_codes, plot_dates, plot_prices, plot_legend, plot_market_ratio

    plot_codes, plot_dates, plot_prices, plot_legend, plot_market_ratio = init_plot_data()

    for i in range(config.nb_similar_make_prediction):
        plot_quote = data[data['CODE'] == plot_codes[i]]
        plot_quote = plot_quote[plot_quote['DATE'] <= pd.to_datetime(plot_dates[i])].tail(config.pattern_length)
        plot_prices[i] = plot_quote['RET'].values
        plot_legend.append(
            str(plot_codes[i]) + "," +
            str(config.similarity_method) + ":" +
            str(top.iloc[i][config.similarity_method]))
        plot_market_ratio[i] = plot_quote[config.market_ratio_type].values

    plot_prices[-1] = pattern['RET'].values
    plot_market_ratio[-1] = pattern[config.market_ratio_type].values

    plot_dates[-1] = pattern['DATE'].iloc[-1]
    plot_legend.append(str(plot_codes[-1]))
    norm_plot_prices = [norm(plot_prices[i], plot_market_ratio[i]) for i in range(config.nb_similar_make_prediction + 1)]

    # assert for result checking
    for i in range(config.nb_similar_make_prediction):
        a = weighted_distance(norm_plot_prices[i], norm_plot_prices[-1], config.pattern_length)
        b = top.iloc[i][config.similarity_method]
        # print(a, b)
        assert a == b, 'calcu error!'

    for i in range(plot_codes.size):
        plt.plot(norm_plot_prices[i],  label=norm_plot_prices[i], linewidth=1.5)

    plt.xlabel('Time')
    plt.ylabel('NAV')
    plt.legend(plot_legend, loc='upper left')
    plt.title("Similarity Search[" + plot_codes[-1] + "]\n")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=45)
    plt.ioff()
    plt.savefig(config.rootPath + '/pic/' + filename +'.png')
    plt.close()

def plot_nav_curve(strategy_net_value, act_net_value, market_net_value, dates, turnover_rate):
    plt.plot(dates, strategy_net_value, 'r-', label=strategy_net_value, linewidth=1.5)
    plt.plot(dates, act_net_value, 'k-', label=act_net_value, linewidth=1.5)
    plt.plot(dates, market_net_value, 'g-', label=market_net_value, linewidth=1.5)

    plt.xlabel('Time')
    plt.ylabel('Net Asset Value')
    plt.legend(['strategy', 'baseline', 'market'], loc='upper left')
    plt.title("Turnover rate: " + str(turnover_rate))
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=20)
    plt.ioff()
    plt.savefig(config.regression_result)
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
