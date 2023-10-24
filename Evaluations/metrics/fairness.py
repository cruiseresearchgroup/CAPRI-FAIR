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
    Z_u = rg_u['rg_u'].sum()
    p_m = rg_u[['rg_u', 'active']].groupby('active').sum() / Z_u
    p_m['rg_fair'] = [0.5, 0.5]
    print("[[ User precision ratios ]]")
    print(p_m)
    beta = 2
    p_m['product'] = (p_m['rg_u'] ** (1-beta)) * (p_m['rg_fair'] ** beta)

    gce_users = (1 / (beta * (1-beta))) * ((p_m['product'].sum()) - 1)
    return gce_users

def gceGlobalItemFairness(ground_truth, predictions, k, checkin_counts):
    """
    Computes the Generalized Cross-Entropy (GCE) between short-head and long-tail
    items. The distinction between these types of items is done elsewhere.

    Note that this is a global metric: it is computed across ALL recommendation
    lists at once. It is not a per-item metric.
    """

    total_coverage_realestate = k * len(ground_truth)
    rg_i = rg_i = Counter([x for u in ground_truth.keys() for x in predictions[u]])
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