import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

from codes.config import config
import time
import matplotlib
matplotlib.use('Agg')
from codes.market import market
from codes.regression_test import make_prediction2

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