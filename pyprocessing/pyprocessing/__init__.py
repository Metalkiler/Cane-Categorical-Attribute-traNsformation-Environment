#     Luís Matos Copyright (c) 2020.
#
#     PCP transform: L.M. Matos, P. Cortez, R. Mendes, A. Moreau. Using Deep Learning for Mobile Marketing User
#     Conversion Prediction. In Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN 2019),
#     paper N-19327, Budapest, Hungary, July, 2019 (8 pages), IEEE, ISBN 978-1-7281-2009-6.
#     https://doi.org/10.1109/IJCNN.2019.8851888 http://hdl.handle.net/1822/62771
#
#     IDF transform: L.M. Matos, P. Cortez, R. Mendes and A. Moreau. A Comparison of Data-Driven Approaches for Mobile
#     Marketing User Conversion Predic- tion. In Proceedings of 9th IEEE International Conference on Intelligent Systems
#     (IS 2018), pp. 140-146, Funchal, Madeira, Portugal, September, 2018, IEEE, ISBN 978-1-5386-7097-2.
#     https://ieeexplore.ieee.org/document/8710472
#     http://hdl.handle.net/1822/61586

import math
from math import ceil

import numpy as np
import pandas as pd


def PCP(f, percentage=0.05):
    """

    The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent)
    into a single level.
    :type percentage: float

    """

    x = f.value_counts()
    N = len(x.index)
    CPercent = ceil(len(f) * percentage)
    tbc = f.value_counts()
    sums = 0
    a = []
    for i in range(0, len(tbc)):
        sums = sums + tbc[i]
        if sums >= CPercent:
            a.append(tbc.index[0:i])  # keep last level!
            break
    f2 = []
    for i in f:
        if i not in np.array(a):
            f2.append('Others')
        else:
            f2.append(i)
    fTreated = pd.Series(f2)
    return fTreated


def IDF(f, N):
    """

    The Inverse Document Frequency (IDF) uses f(x)= log(n/f_x),
    where n is the length of x and f_x is the frequency of x.


    """

    x = f.value_counts()

    res = np.zeros(N)
    idf = []
    for i in x:
        idf.append(math.log(N / i))
    L = np.unique(f)
    NL = len(L)

    for i in range(0, NL):
        I = np.where(f == L[i])
        res[I] = idf[i]

    return res
