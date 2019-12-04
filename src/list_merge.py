#!/usr/bin/env python3

import numpy as np


def round_robin_list_merger(lists):
    n_lists = len(lists)
    for i in range(n_lists - 1):
        assert len(lists[i]) == len(lists[i+1])
    list_len = len(lists[0])
    final_list = list()
    for idx in range(list_len):
        for l in lists:
            final_list.append(l[idx])
    return final_list


def frequency_list_merger(lists):
    v = np.concatenate(lists)
    unique, count = np.unique(v, return_counts=True)
    count_descending = count.argsort()[::-1]
    return unique[count_descending]
