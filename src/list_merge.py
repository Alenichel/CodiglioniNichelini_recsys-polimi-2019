#!/usr/bin/env python3

import numpy as np


def round_robin_list_merger(lists):
    list_len = min(map(lambda x: len(x), lists))
    final_list = list()
    for idx in range(list_len):
        for l in lists:
            if l[idx] not in final_list:
                final_list.append(l[idx])
    return final_list


def frequency_list_merger(lists):
    v = np.concatenate(lists)
    unique, count = np.unique(v, return_counts=True)
    count_descending = count.argsort()[::-1]
    return unique[count_descending]


def medrank(lists):
    n_lists = len(lists)
    list_len = min(map(lambda x: len(x), lists))
    rank = dict()
    for idx in range(list_len):
        for l_idx in range(n_lists):
            item = lists[l_idx][idx]
            if item in rank:
                rank[item][l_idx] = idx+1
            else:
                x = [np.inf for _ in range(n_lists)]
                x[l_idx] = idx+1
                rank[item] = x
    filtered_keys = list(filter(lambda k: rank[k].count(np.inf) < n_lists / 2, rank))
    rank = {k: rank[k] for k in filtered_keys}
    for item in rank:
        rank[item] = np.median(rank[item])
    final_rank = sorted(rank.keys(), key=lambda z: rank[z])
    if len(final_rank) < 10:
        for idx in range(list_len):
            for l in lists:
                if l[idx] not in final_rank:
                    final_rank.append(l[idx])
                    if len(final_rank) >= 10:
                        return final_rank
    return final_rank


if __name__ == '__main__':
    l1 = ['Ibis', 'Etap', 'Novotel', 'Mercure', 'Hilton', 'Sheraton', 'Crillon']
    l2 = ['Crillon', 'Novotel', 'Sheraton', 'Hilton', 'Ibis', 'Ritz', 'Lutetia']
    l3 = ['Le Roche', 'Lodge In', 'Ritz', 'Lutetia', 'Novotel', 'Sheraton', 'Mercure']
    print(medrank([l1, l2, l3]))
