from scipy.stats.stats import pearsonr
from scipy.spatial import distance
from base import norm, plot_simi_stock
from market import load_all_data
from config import *
pd.set_option('display.width', 1000)


def t_rol_aply(target, pattern, similarity_method):
    global ascending_sort
    if similarity_method == 'pearsonr':
        ascending_sort = False
        return pearsonr(target, pattern)[0]
    else:
        ascending_sort = True
        return distance.euclidean(norm(target), norm(pattern))

def target_apply(target, pattern):
    p_close = pattern['CLOSE']
    t_close = target['CLOSE']
    target[similarity_method] = t_close.rolling(window=pattern_length).apply(func=t_rol_aply, args=(p_close, similarity_method, ))
    result = target.dropna(axis=0, how='any').sort_values(by=[similarity_method], ascending=ascending_sort).head(1)
    return result

def load_and_process_data(start_date, date=None, nb_data=nb_data, code=code):
    """
    :param start_date: give a specific date to down-scope Pattern.(包含这一天，即预测下一天)
    :param date: give a specific date to down-scope Targets
    :param nb_data: if == 0, then use whole all_data.
    :return: all_data, pattern, targets
    """

    all_data = load_all_data()
    all_data = all_data.loc[(all_data != 0).any(axis=1)]

    pattern = all_data[all_data['CODE'] == code].reset_index(drop=True)
    pattern = pattern[(pattern['DATE'] >= start_date)].head(pattern_length)

    targets = all_data[all_data['CODE'] != code].reset_index(drop=True)
    if nb_data != 0:
        targets = targets.head(nb_data)

    if date is not None:
        targets = targets[target['DATE'] >= date]

    return all_data, pattern, targets

def find_tops_similar(pattern, targets, nb_similarity=nb_similarity, similarity_method=similarity_method):

    """
    :return is dataframe of the tops similar stock(s).
        colomns : ['CODE', 'DATE', 'Similarity_Score']
    """
    result = targets.groupby(['CODE']).apply(func=target_apply, pattern=pattern)
    sorted_result = result.sort_values(by=[similarity_method], ascending=ascending_sort)

    result = sorted_result.head(nb_similarity)

    tops = pd.DataFrame()
    tops['CODE'] = result['CODE']
    tops['DATE'] = result['DATE']
    tops[similarity_method] = result[similarity_method]

    return tops


if __name__ == '__main__':
    data, pattern, target = load_and_process_data(start_date)
    tops = find_tops_similar(pattern, target, nb_similarity)
    plot_simi_stock(tops, data, pattern, )

    print('finish search!')
