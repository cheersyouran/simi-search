#encoding:utf-8
import pandas as pd
from os import listdir, path
import matplotlib.pyplot as plt
from sklearn import preprocessing

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


def plot_stocks_price_plot(codes, prices, legend, n=True):
    styles = ['k--', 'k:', 'k-.']
    for i in range(codes.size):
        if n == True:
            prices[i] = norm(prices[i])
        if i == codes.size - 1:
            plt.plot(prices[i], 'r-', label=codes[i], linewidth=1.5)
        else:
            plt.plot(prices[i], styles[i], label=codes[i], linewidth=1.2)


    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend(legend, loc='upper right')
    plt.title("Similarity Search[" + codes[-1] +"]")
    plt.grid(True)
    plt.xticks(fontsize=8, rotation=45)
    plt.ioff()
    plt.show()
    plt.close()

if __name__ == '__main__':
    # merge_raw_data()
    df = load_data().head(50)
    print(df)
