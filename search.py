from scipy.stats.stats import pearsonr
from datetime import timedelta
from scipy.spatial import distance
import base
from config import *
import pandas as pd
pd.set_option('display.width', 1000)

def t_rol_aply(target, pattern, method='pearsonr'):
    global ascending_sort
    if method == 'pearsonr':
        ascending_sort = False
        return pearsonr(target, pattern)[0]
    else:
        ascending_sort = True
        return distance.euclidean(base.norm(target), base.norm(pattern))

def target_apply(target, pattern):
    p_close = pattern['CLOSE']
    t_close = target['CLOSE']
    target[similarity_method] = t_close.rolling(window=pattern_length).apply(func=t_rol_aply, args=(p_close,))
    result = target.dropna(axis=0, how='any').sort_values(by=[similarity_method], ascending=ascending_sort).head(1)
    return result

def load_and_process_data(date=None, nb_data=10000):
    """
    :param date: give a specific date to down-scope Targets
    :param nb_data: if == 0, then use whole all_data.
    :return: all_data, pattern, targets
    """

    all_data = base.load_data()
    all_data = all_data.loc[(all_data != 0).any(axis=1)]

    pattern = all_data[all_data['CODE'] == code].reset_index(drop=True)
    pattern = pattern[(pattern['DATE'] >= start_date)].head(pattern_length)

    targets = all_data[all_data['CODE'] != code].reset_index(drop=True)
    if nb_data != 0:
        targets = targets.head(nb_data)

    if date != None:
        targets = targets[target['DATE'] >= date]

    return all_data, pattern, targets

def most_similar(pattern, targets, nb_similarity):

    """
    :return is the most similar stock(s).
        most['CODE', 'DATE', 'Similarity_Score']
    """
    result = targets.groupby(['CODE']).apply(func=target_apply, pattern=pattern)
    sorted_result = result.sort_values(by=[similarity_method], ascending=ascending_sort)

    result = sorted_result.head(nb_similarity)

    most = pd.DataFrame()
    most['CODE'] = result['CODE']
    most['DATE'] = result['DATE']
    most[similarity_method] = result[similarity_method]

    return most


if __name__ == '__main__':
    data, pattern, target = load_and_process_data()
    most = most_similar(pattern, target, nb_similarity)
    base.plot_stocks_price_plot(most, data, pattern,)

    print('Finish')
