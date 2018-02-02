from codes.config import *
from scipy.spatial import distance
from scipy.stats.stats import pearsonr
from codes.market import get_historical_data
from codes.base import norm, plot_simi_stock

count = 0
data_set = None

def t_rol_aply(target, pattern):
    global ascending_sort
    if similarity_method == 'pearsonr':
        ascending_sort = False
        return pearsonr(target, pattern)[0]
    else:
        ascending_sort = True
        return distance.euclidean(norm(target), norm(pattern))

def target_apply(target, pattern):
    global count
    p_close = pattern['CLOSE']
    t_close = target['CLOSE']
    target[similarity_method] = t_close.rolling(window=pattern_length).apply(func=t_rol_aply, args=(p_close, ))
    result = target.dropna(axis=0, how='any').sort_values(by=[similarity_method], ascending=ascending_sort).head(1)
    count = count + 1
    return result

def all_search(pattern, targets):

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
    time_start = time.time()

    data, pattern, target = get_historical_data(start_date=start_date)
    tops = all_search(pattern, target)
    plot_simi_stock(tops, data, pattern, 'all_simi_search')

    time_end = time.time()
    print('All Time is:', time_end - time_start)

    print('finish search!')
