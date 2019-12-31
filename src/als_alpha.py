#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


data = [

]


if __name__ == '__main__':
    print(len(data))
    bot = 0.04
    top = 0.043
    data = sorted(data, key=lambda d: d['params']['alpha'])
    x = [d['params']['alpha'] for d in data]
    y = [d['target'] for d in data]
    plt.plot(x, y)
    max_y = max(y)
    max_x = x[y.index(max_y)]
    print(max_x, max_y)
    x = [min(x), max(x)]
    y = [max_y, max_y]
    plt.plot(x, y, 'r--')
    x = [max_x, max_x]
    y = [bot, top]
    plt.plot(x, y, 'r--')
    plt.ylim(bot, top)
    plt.xlabel('alpha')
    plt.ylabel('MAP')
    plt.show()

