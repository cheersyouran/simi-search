import sys
import os
from codes.regression_test import make_prediction2

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

from codes.config import config
if 'Youran' in config.rootPath:
    config.nb_codes = 3
    config.plot_simi_stock = False
    config.nb_similar_of_each_stock = 100
    config.nb_similar_make_prediction = 5
    config.nb_similar_of_all_similar = 15
    config.cores = 4

import time
import matplotlib
matplotlib.use('Agg')
from codes.market import market

if __name__ == '__main__':

    time_start = time.time()
    print('\n[Start Date]: ' + str(config.start_date.date()))
    print('[Current Date]: ' + str(market.current_date.date()))

    action, pred_ratio, act_ratio, market_ratio = make_prediction2()

    print('[Predict]:', pred_ratio)
    print('[Actual ]:', act_ratio)
    print('[Market ]:', market_ratio)

    time_end = time.time()
    print('Search Time:', time_end - time_start)