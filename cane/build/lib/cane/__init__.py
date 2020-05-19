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


def PCP_Data(dataset=pd.DataFrame(), perc=0.05, mergeCategory="Others"):
    """
    The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a
    single level. It works by first sorting the feature levels according to their frequency in the training data.
    Then, the least frequent levels (summing up to a threshold percentage of P ) are merged into a single category
    denoted as "Others".
    Example:
        import pandas as pd
        import cane
        x=["a","a","a","b","b","b","b","b","c","c","c","c","c","c","c","d"]
        df=pd.DataFrame({"x":x,"x2":x})
        dataPCP, dicionary = cane.PCP_Data(df.copy()) #always send a copy of the dataframe
    :param mergeCategory: Category for merging the data (by default "Others")
    :param dataset: dataset to transform
    :param perc: threshold percentage of P
    :return: tuple containing the "Dataset" transformed and the dictionary for latter usage (Info)



    """
    PCP_Config = {}

    for column in dataset:
        dataset[column] = PCP_Single(f=dataset[column], perc=perc,
                                     mergeCategory=mergeCategory)  # já converte para número depois!
        name = dataset[column]
        dataset[column] = dataset[column].astype("category")

        PCP_Config[column] = dict(zip(np.unique(name), np.unique(dataset[column])))
    return dataset, PCP_Config


def IDF_Data(dataset):
    """
    The Inverse Document Frequency (IDF) uses f(x)= log(n/f_x),
    where n is the length of x and f_x is the frequency of x.
    Example:
        import pandas as pd
        import cane
        x=["a","a","a","b","b","b","b","b","c","c","c","c","c","c","c","d"]
        df=pd.DataFrame({"x":x,"x2":x})
        dataIDF = cane.IDF_Data(df.copy()) #always send a copy of the dataframe
    :param dataset: dataset to transform

    :return: Dataset with the IDF transformation
    """

    for column in dataset:
        dataset[column] = IDF_Single(f=dataset[column])  # já converte para número depois!

    return dataset


def PCP_Single(f, perc=0.05, mergeCategory="Others"):
    """

    The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a
    single level. It works by first sorting the feature levels according to their frequency in the training data.
    Then, the least frequent levels (summing up to a threshold percentage of P ) are merged into a single category
    denoted as "Others".

    :param mergeCategory: Category for merging the data (by default "Others")
    :param perc: the threshold percentage P


    """

    x = f.value_counts()
    N = len(f)
    CPercent = ceil(len(f) * perc)
    tbc = f.value_counts()
    sums = 0
    a = []
    for i in range(0, len(tbc)):
        sums = sums + tbc[i]
        if sums >= CPercent:
            a.append(tbc.index[0:i + 1])  # keep last level!
            break
    f2 = []
    for i in f:
        if i not in np.array(a):
            f2.append(mergeCategory)
        else:
            f2.append(i)
    fTreated = pd.Series(f2)
    return fTreated


def IDF_Single(f):
    """

    The Inverse Document Frequency (IDF) uses f(x)= log(n/f_x),
    where n is the length of x and f_x is the frequency of x.


    """

    x = f.value_counts(sort=False)
    N = len(f)
    res = f.copy()
    idf = {}
    for i in range(0, len(x)):
        idf[x.index[i]] = math.log(N / x.values[i])

    resTreated = res.replace(idf)
    return resTreated
