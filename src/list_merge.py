#!/usr/bin/env python3

import numpy as np


def round_robin_list_merger(lists, at=10):
    n_lists = len(lists)
    for i in range(n_lists):
        assert len(lists[i]) >= at
    final_list = list()
    for idx in range(at):
        for l in lists:
            if l[idx] not in final_list:
                final_list.append(l[idx])
    return final_list


def frequency_list_merger(lists):
    v = np.concatenate(lists)
    unique, count = np.unique(v, return_counts=True)
    count_descending = count.argsort()[::-1]
    return unique[count_descending]


def medrank(lists, at=10):
    n_lists = len(lists)
    rank = dict()
    for l_idx in range(n_lists):
        for idx in range(at):
            item = lists[l_idx][idx]
            if item in rank:
                rank[item][l_idx] = idx+1
            else:
                x = [np.inf for _ in range(n_lists)]
                x[l_idx] = idx+1
                rank[item] = x
    for item in rank:
        rank[item] = np.median(rank[item])
    return sorted(rank.keys(), key=lambda z: rank[z])


if __name__ == '__main__':
    l1 = ['Ibis', 'Etap', 'Novotel', 'Mercure', 'Hilton', 'Sheraton', 'Crillon']
    l2 = ['Crillon', 'Novotel', 'Sheraton', 'Hilton', 'Ibis', 'Ritz', 'Lutetia']
    l3 = ['Le Roche', 'Lodge In', 'Ritz', 'Lutetia', 'Novotel', 'Sheraton', 'Mercure']
    print(medrank([l1, l2, l3]))
