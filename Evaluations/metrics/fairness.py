import itertools
import numpy as np
import pandas as pd
from typing import List
from collections import Counter


def gceGlobalUserFairness(ground_truth, predictions, active_users):
    """
    Computes the Generalized Cross-Entropy (GCE) between repeat and explore
    users. The distinction between these types of users is done elsewhere.

    Note that this is a global metric: it is computed across ALL recommendation
    lists at once. It is not a per-user metric.
    """

    rg_u = [(u, len([x for x in predictions[u] if x in ground_truth[u]]))
            for u in ground_truth.keys()]
    rg_u = pd.DataFrame(rg_u, columns=['user_id', 'rg_u']).set_index('user_id')

    rg_u = rg_u.join(active_users[['active']], how='left').fillna(0)
    active_proportion = rg_u['active'].mean()
    Z_u = rg_u['rg_u'].sum()
    p_m = rg_u[['rg_u', 'active']].groupby('active').sum() / Z_u
    p_m['rg_fair'] = pd.Series(
        [(1 - active_proportion), active_proportion],
        index=[False, True]
    )
    print("[[ User precision ratios ]]")
    print(p_m)
    beta = 2
    p_m['product'] = (p_m['rg_u'] ** (1-beta)) * (p_m['rg_fair'] ** beta)

    gce_users = (1 / (beta * (1-beta))) * ((p_m['product'].sum()) - 1)
    return gce_users


def accuracyMetricByUserGroup(metric, users, active_users):
    """
    Given a list of metric scores (e.g. precision), users, and a distinction
    between active users, computes the average inside both groups.
    """
    results = {
        'active': [],
        'inactive': [],
    }
    for m, u in zip(metric, users):
        if active_users.loc[u, 'active']:
            results['active'].append(m)
        else:
            results['inactive'].append(m)

    results['active'] = np.mean(results['active'])
    results['inactive'] = np.mean(results['inactive'])
    return results


def gceGlobalItemFairness(ground_truth, predictions, k, checkin_counts):
    """
    Computes the Generalized Cross-Entropy (GCE) between short-head and long-tail
    items. The distinction between these types of items is done elsewhere.

    Note that this is a global metric: it is computed across ALL recommendation
    lists at once. It is not a per-item metric.
    """

    total_coverage_realestate = k * len(ground_truth)
    rg_i = Counter([x for u in ground_truth.keys() for x in predictions[u][:k]])
    rg_i = pd.DataFrame(rg_i.items(), columns=['poi_id', 'rg_i']).set_index('poi_id')
    rg_i = rg_i.join(checkin_counts[['checkins', 'short_head']], how='right').fillna(0)
    short_head_proportion = rg_i['short_head'].mean()
    Z_i = rg_i['rg_i'].sum()
    p_mi = rg_i[['rg_i', 'short_head']].groupby('short_head').sum() / Z_i
    rg_i_fair = len(checkin_counts) / total_coverage_realestate
    p_mi['rg_fair'] = pd.Series(
        [(1 - short_head_proportion), short_head_proportion],
        index=[False, True]
    )
    print("[[ Item coverage ratios ]]")
    print(p_mi)
    beta = 2
    p_mi['product'] = (p_mi['rg_i'] ** (1-beta)) * (p_mi['rg_fair'] ** beta)

    gce_items = (1 / (beta * (1-beta))) * ((p_mi['product'].sum()) - 1)
    return gce_items


def exposureMetricByItemGroup(ground_truth, predictions, k, checkin_counts):
    """
    Compute the average exposure rate of items according to group.
    """
    user_count = len(ground_truth)

    exposure = Counter([x for u in ground_truth.keys() for x in predictions[u][:k]])
    exposure = pd.DataFrame(exposure.items(), columns=['poi_id', 'exposure']).set_index('poi_id')
    exposure = exposure.join(checkin_counts[['short_head']], how='right').fillna(0)
    exposure['exposure'] = exposure['exposure'] / user_count

    results = {
        'short_head': exposure[exposure['short_head']]['exposure'].mean(),
        'long_tail': exposure[~exposure['short_head']]['exposure'].mean()
    }
    return results