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
from pqdm.processes import pqdm
from functools import partial


def __pcp_single__(f, perc_inner=0.05, mergeCategoryinner="Others"):
    """
    The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a
    single level. It works by first sorting the feature levels according to their frequency in the training data.
    Then, the least frequent levels (summing up to a threshold percentage of P ) are merged into a single category
    denoted as "Others" for a Single Column.
    """
    x = f.value_counts()
    N = len(f)
    CPercent = ceil(len(f) * perc_inner)
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
            f2.append(mergeCategoryinner)
        else:
            f2.append(i)
    fTreated = pd.Series(f2)
    return fTreated


def pcp(dataset=pd.DataFrame(), perc=0.05, mergeCategory="Others", n_coresJob=1):
    """
    The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a
    single level. It works by first sorting the feature levels according to their frequency in the training data.
    Then, the least frequent levels (summing up to a threshold percentage of P ) are merged into a single category
    denoted as "Others", it uses all the dataset!

    :param mergeCategory: Category for merging the data (by default "Others")
    :param dataset: dataset to transform
    :param perc: threshold percentage of P
    :return: tuple containing the "Dataset" transformed and the dictionary for latter usage (Info)



    """

    PCP_Config = {}
    TransformedData = dataset.copy()
    dfFinal = pd.DataFrame()
    if not (isinstance(TransformedData, pd.DataFrame)):
        raise Exception("Dataset needs to be of type Pandas")
    if perc > 1:
        raise ValueError("Percentage goes from 0 to 1, which above 1 means above 100%")
    if isinstance(TransformedData, pd.DataFrame) and perc <= 1:
        columns_Processing = []
        columnsOld = []
        for column in TransformedData:
            columns_Processing.append(TransformedData[column])
            columnsOld.append(column)
        func = partial(__pcp_single__, perc_inner=perc, mergeCategoryinner=mergeCategory)

        d = pqdm(columns_Processing, func, n_jobs=n_coresJob)

        for i in d:
            dfFinal = pd.concat([dfFinal, i], axis=1)

        dfFinal.columns = columnsOld
        for column in dfFinal:
            name = dfFinal[column]
            dfFinal[column] = dfFinal[column].astype("category")

            PCP_Config[column] = dict(zip(np.unique(name), np.unique(dfFinal[column])))
        return dfFinal, PCP_Config


def __idf_single__(f):
    x = f.value_counts(sort=False)
    N = len(f)
    res = f.copy()
    idf = {}
    for i in range(0, len(x)):
        idf[x.index[i]] = math.log(N / x.values[i])

    resTreated = res.replace(idf)
    return resTreated


def idf(dataset, n_coresJob=1):
    """
    The Inverse Document Frequency (IDF) uses f(x)= log(n/f_x),
    where n is the length of x and f_x is the frequency of x.
    Example:
        import pandas as pd
        import cane
        x=["a","a","a","b","b","b","b","b","c","c","c","c","c","c","c","d"]
        df=pd.DataFrame({"x":x,"x2":x})
        dataIDF = cane.IDF_Data(df.copy()) #always send a copy of the dataframe
    :param n_coresJob: Number of cores to use
    :param dataset: dataset to transform

    :return: Dataset with the IDF transformation
    """

    TransformedData = dataset.copy()
    dfFinal = pd.DataFrame()
    columns_Processing = []
    columnsOld = []
    if not (isinstance(TransformedData, pd.DataFrame)):
        raise Exception("Dataset needs to be of type Pandas")
    else:

        for column in TransformedData:
            columns_Processing.append(TransformedData[column])
            columnsOld.append(column)

        d = pqdm(columns_Processing, __idf_single__, n_jobs=n_coresJob)

        for i in d:
            dfFinal = pd.concat([dfFinal, i], axis=1)

        dfFinal.columns = columnsOld
        return dfFinal


def __one_hot_single__(dataset, column_prefix=None):
    """ Application of the one-hot encoding preprocessing (e.g., [0,0,1
                                                                 0,1,0])
        Note: if you use the column_prefixer it is not possible to undo the one_hot encoding preprocessing
        If column_prefix is column then the column names will be used, else it will use the custom name provided
        :return: A new Dataset with the one-hot encoding transformation
    """
    if column_prefix is None:
        data = pd.get_dummies(dataset)
    else:
        if column_prefix.lower() == 'column':

            data = pd.get_dummies(dataset, prefix=dataset.name)
        else:
            data = pd.get_dummies(dataset, prefix=column_prefix)

    return data


def one_hot(dataset, column_prefix=None, n_coresJob = 1):
    """ Application of the one-hot encoding preprocessing (e.g., [0,0,1
                                                                 0,1,0])
        Note: if you use the column_prefixer it is not possible to undo the one_hot encoding preprocessing
        If column_prefix is column then the column names will be used, else it will use the custom name provided
        :return: A new Dataset with the one-hot encoding transformation
    """
    dfFinal = pd.DataFrame()
    columns_Processing = []
    columnsOld = []

    if not (isinstance(dataset, pd.DataFrame)):
        raise Exception("Dataset needs to be of type Pandas")
    else:

        for column in dataset:
            columns_Processing.append(dataset[column])
            columnsOld.append(column)

        func = partial(__one_hot_single__, column_prefix=column_prefix)
        d = pqdm(columns_Processing, func, n_jobs=n_coresJob)

        for i in d:
            dfFinal = pd.concat([dfFinal, i], axis=1)

    return dfFinal
