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

import itertools
import math
from functools import partial
from math import ceil
import numpy as np
import pandas as pd
from pqdm.processes import pqdm
from tqdm import tqdm


def __pcp_single__(f, perc_inner=0.05, mergeCategoryinner="Others"):
    """
        The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a
        single level. It works by first sorting the feature levels according to their frequency in the training data.
        Then, the least frequent levels (summing up to a threshold percentage of P ) are merged into a single category
        denoted as "Others" for a Single Column.
    """

    CPercent = ceil(len(f) * (1 - perc_inner))

    accumulated = itertools.accumulate(f.value_counts().items(), lambda a, b: (b[0], a[1] + b[1]))
    kept = {P[0] for P in itertools.takewhile(lambda a: a[1] <= CPercent, accumulated)}
    return pd.Series(X if X in kept else mergeCategoryinner for X in f)


def pcp(dataset=pd.DataFrame(), perc=0.05, mergeCategory="Others", n_coresJob=1, disableLoadBar=False,
        columns_use=None):
    """
    The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a
    single level. It works by first sorting the feature levels according to their frequency in the training data.
    Then, the least frequent levels (summing up to a threshold percentage of P ) are merged into a single category
    denoted as "Others", it uses all the dataset!

    :param columns_use: Specific columns to apply transformation(default None applies to every COLUMN).
    :param disableLoadBar: Chooses if you want load bar or not (default = True)
    :param n_coresJob: Number of cores to use for the preprocessing
    :param mergeCategory: Category for merging the data (by default "Others")
    :param dataset: dataset to transform
    :param perc: threshold percentage of P
    :return: the "Dataset" transformed



    """

    TransformedData = dataset.copy()

    assert isinstance(TransformedData, pd.DataFrame), "Dataset needs to be of type Pandas"
    assert 0 <= perc <= 1, "Percentage goes from 0 to 1, it may neither be negative nor above 1"
    if isinstance(TransformedData, pd.DataFrame) and perc <= 1:
        columns_Processing = []
        if columns_use is not None:
            assert all(flag in TransformedData.columns for flag in
                       columns_use), "Use columns specific to the dataset given the columns provided are not found " \
                                     + ' '.join([j for j in columns_use])
            if set(columns_use).issubset(TransformedData.columns):

                for column in columns_use:
                    columns_Processing.append(TransformedData[column])

        else:
            for column in TransformedData:
                columns_Processing.append(TransformedData[column])
        func = partial(__pcp_single__, perc_inner=perc, mergeCategoryinner=mergeCategory)

        d = pqdm(columns_Processing, func, n_jobs=n_coresJob, disable=disableLoadBar)

        if columns_use is not None:
            dfFinal = pd.concat([i for i in d], axis=1)
            dfFinal.columns = columns_use
            dfFinal = pd.concat([dfFinal, TransformedData[TransformedData.columns.difference(columns_use, sort=False)]],
                                axis=1,
                                sort=True)
        else:
            dfFinal = pd.concat([i for i in d], axis=1)
            dfFinal.columns = TransformedData.columns
        return dfFinal


def pcp_multicolumn(dataset=pd.DataFrame(), perc=0.05, mergeCategory="Others",
                    columns_use=None, disableLoadBar=True):
    """
    Similarly to the normal PCP this function uses X columns given merges and applies the pcp transformation to it.
    Next it will apply the transformation into the disaggregated columns sharing the transformation obtained previously

    :param disableLoadBar: Chooses if you want load bar or not (default = True)
    :param columns_use: Specific columns to apply transformation.
    :param mergeCategory: Category for merging the data (by default "Others")
    :param dataset: dataset to transform
    :param perc: threshold percentage of P
    :return: the "Dataset" transformed



    """

    TransformedData = dataset.copy()

    assert isinstance(TransformedData, pd.DataFrame), "Dataset needs to be of type Pandas"
    assert 0 <= perc <= 1, "Percentage goes from 0 to 1, it may neither be negative nor above 1"
    assert (columns_use is not None), "multicolumn PCP requires the usage of columns!"
    assert (len(columns_use) > 1), "multicolumn PCP requires the usage of more than 1 column!"
    if isinstance(TransformedData, pd.DataFrame) and perc <= 1 and columns_use is not None:

        assert all(flag in TransformedData.columns for flag in
                   columns_use), "Use columns specific to the dataset given the columns provided are not found " \
                                 + ' '.join([j for j in columns_use])
        if set(columns_use).issubset(TransformedData.columns):

            mergedColumn = []
            for column in columns_use:
                mergedColumn.append(TransformedData[column].values)

        dfTesting = pd.Series([y for x in mergedColumn for y in x], name="X")

        d = __pcp_single__(dfTesting, perc_inner=perc, mergeCategoryinner=mergeCategory)
        dic = {v: [i for i in np.unique(v)][0] for _, v in d.items()}
        for column in tqdm(columns_use, desc="Transformation", total=len(columns_use), disable=disableLoadBar):
            TransformedData[column] = TransformedData[column].map(dic)
            TransformedData[column] = TransformedData[column].fillna(mergeCategory)  # because of others

    return TransformedData


def idf_multicolumn(dataset, columns_use=None, disableLoadBar=False):
    """
    The Inverse Document Frequency (IDF) uses f(x)= log(n/f_x),
    where n is the length of x and f_x is the frequency of x.
    Next it will apply the transformation into the disaggregated columns sharing
    the transformation obtained previously

    :param disableLoadBar: Chooses if you want load bar or not (default = True)
    :param columns_use: List of columns to use
    :param dataset: dataset to transform

    :return: Dataset with the IDF transformation
    """

    TransformedData = dataset.copy()

    assert isinstance(TransformedData, pd.DataFrame), "Dataset needs to be of type Pandas"
    assert (columns_use is not None), "multicolumn idf requires the usage of columns!"
    assert (len(columns_use) > 1), "multicolumn idf requires the usage of more than 1 column!"
    if isinstance(TransformedData, pd.DataFrame) and columns_use is not None:

        assert all(flag in TransformedData.columns for flag in
                   columns_use), "Use columns specific to the dataset given the columns provided are not found " \
                                 + ' '.join([j for j in columns_use])
        if set(columns_use).issubset(TransformedData.columns):

            mergedColumn = []
            for column in columns_use:
                mergedColumn.append(TransformedData[column].values)

        dfTesting = pd.Series([y for x in mergedColumn for y in x], name="X")

        d = __idf_single_dic__(dfTesting)
        for column in tqdm(columns_use, desc="Transformation", total=len(columns_use), disable=disableLoadBar):
            TransformedData[column] = TransformedData[column].replace(d)
    return TransformedData


def PCPDictionary(dataset=pd.DataFrame(), columnsUse=None, targetColumn=None):
    """
    This function creates the dictionary to be used for the PCP transformation (on the test data).


    Parameters
    ----------
    dataset
    columnsUse

    Returns
    -------

    """
    DicColumnRenamer = {}
    if columnsUse is None:
        columns = dataset[dataset != targetColumn].tolist()
        columnsUse = columns

    for column in columnsUse:
        # print(column)
        name = dataset[column]
        dataset[column] = dataset[column].astype("str")
        DicColumnRenamer[column] = dict(zip(np.unique(dataset[[column]]), np.unique(name)))

    return DicColumnRenamer


def __idf_single__(f):
    x = f.value_counts(sort=False)
    N = len(f)
    res = f.copy()
    idf = {}
    for i in range(0, len(x)):
        idf[x.index[i]] = math.log(N / x.values[i])

    resTreated = res.replace(idf)
    return resTreated


def __idf_single_dic__(f):
    x = f.value_counts(sort=False)
    N = len(f)
    idf = {}
    for i in range(0, len(x)):
        idf[x.index[i]] = math.log(N / x.values[i])
    return idf


def idf(dataset, n_coresJob=1, disableLoadBar=False, columns_use=None):
    """
    The Inverse Document Frequency (IDF) uses f(x)= log(n/f_x),
    where n is the length of x and f_x is the frequency of x.

    :param columns_use: List of columns to use
    :param disableLoadBar: Chooses if you want load bar or not (default = True)
    :param n_coresJob: Number of cores to use
    :param dataset: dataset to transform

    :return: Dataset with the IDF transformation
    """

    TransformedData = dataset.copy()
    columns_Processing = []
    assert isinstance(TransformedData, pd.DataFrame), "Dataset needs to be of type Pandas"
    if isinstance(TransformedData, pd.DataFrame):
        #
        if columns_use is not None:
            assert all(flag in TransformedData.columns for flag in
                       columns_use), "Use columns specific to the dataset given the columns provided are not found " \
                                     + ' '.join([j for j in columns_use])
            if set(columns_use).issubset(TransformedData.columns):

                for column in columns_use:
                    columns_Processing.append(TransformedData[column])

        else:
            for column in TransformedData:
                columns_Processing.append(TransformedData[column])

        d = pqdm(columns_Processing, __idf_single__, n_jobs=n_coresJob, disable=disableLoadBar)
        if columns_use is not None:
            dfFinal = pd.concat([i for i in d], axis=1)
            dfFinal = pd.concat([dfFinal, TransformedData[TransformedData.columns.difference(columns_use, sort=False)]],
                                axis=1,
                                sort=True)
        else:
            dfFinal = pd.concat([i for i in d], axis=1)

        return dfFinal


def idfDictionary(Original=pd.DataFrame(), Transformed=pd.DataFrame, columns_use=None, targetColumn=None):
    """
    Creates the mapping for the IDF transformation in the test set using the training set

    Parameters
    ----------
    trOriginal Original Data
    trainIDFTransformed Data Transformed with idf
    cols Columns that used IDF

    Returns dictionary
    -------

    """
    dic = dict()
    if columns_use is None:
        columns = Original[Original != targetColumn].tolist()
        cols = columns

    for col in columns_use:
        df = pd.merge(Original[col], Transformed[col], left_index=True, right_index=True)
        df = df.set_index(df.columns[0])
        df.index.name = None
        df = df.rename(columns={df.columns[0]: col})
        df = df.to_dict('dict')

        # dic = dict(dic, **items)
        dic.update(df)

    return dic


def __one_hot_single__(dataset, column_prefix=None):
    """ Application of the one-hot encoding preprocessing (e.g., [0,0,1
                                                                 0,1,0])
        Note: if you use the column_prefixer it is not possible to undo the one_hot encoding preprocessing
        If column_prefix is column then the column names will be used, else it will use the custom name provided
        :param dataset: dataset to one-hot encode
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


def one_hot(dataset, column_prefix=None, n_coresJob=1, disableLoadBar=True, columns_use=None):
    """ Application of the one-hot encoding preprocessing (e.g., [0,0,1
                                                                 0,1,0])
        Note: if you use the column_prefixer it is not possible to undo the one_hot encoding preprocessing
        If column_prefix is column then the column names will be used, else it will use the custom name provided
        :param columns_use:
        :param column_prefix:
        :param n_coresJob: Number of cores you need for multiprocessing (e.g., 1 column per process)
        :param disableLoadBar: Chooses if you want load bar or not (default = True)
        :param dataset: dataset to one-hot encode

        :return: A new Dataset with the one-hot encoding transformation
    """
    dfFinal = pd.DataFrame()
    columns_Processing = []
    assert isinstance(dataset, pd.DataFrame), "Dataset needs to be of type Pandas"
    if isinstance(dataset, pd.DataFrame):
        if columns_use is not None:
            assert all(flag in dataset.columns for flag in
                       columns_use), "Use columns specific to the dataset given the columns provided are not found " \
                                     + ' '.join([j for j in columns_use])
            if set(columns_use).issubset(dataset.columns):

                for column in columns_use:
                    columns_Processing.append(dataset[column])

        else:
            for column in dataset:
                columns_Processing.append(dataset[column])

        func = partial(__one_hot_single__, column_prefix=column_prefix)
        d = pqdm(columns_Processing, func, n_jobs=n_coresJob, disable=disableLoadBar)

        if columns_use is not None:
            dfFinal = pd.concat([i for i in d], axis=1)
            dfFinal = pd.concat([dfFinal, dataset[dataset.columns.difference(columns_use, sort=False)]],
                                axis=1,
                                sort=True)
        else:
            dfFinal = pd.concat([i for i in d], axis=1)
    return dfFinal


def scale_data(df, column=[], n_cores=1, scaleFunc="", customfunc=None):
    assert isinstance(df, pd.DataFrame), "Dataset needs to be of type Pandas"
    assert (scaleFunc != "" or scaleFunc == "min_max" or scaleFunc == "std" or scaleFunc == "custom"), "Specify a " \
                                                                                                       "scaler (" \
                                                                                                       "e.g., " \
                                                                                                       "'min_max' or " \
                                                                                                       "'std') or " \
                                                                                                       "'custom' "

    if scaleFunc == 'custom':
        assert (callable(customfunc)), "Please provide a function for the custom function you want to use"

    if column is not None:
        assert all(flag in df.columns for flag in
                   column), "Use columns specific to the dataset given the columns provided are not found " \
                            + ' '.join([j for j in column])
    valArgs = []
    if len(column) == 0:
        columns = df.columns.values
        diff = columns
        for i in columns:
            valArgs.append(df[i])
    else:
        columns = df.columns.values
        for i in column:
            valArgs.append(df[i])
        diff = columns
    if scaleFunc == "min_max":
        func = partial(scale_single_min_max)
    elif scaleFunc == "std":
        func = partial(scale_single_std)
    else:
        func = partial(customfunc)
    d = pqdm(valArgs, func, n_jobs=n_cores)

    dfFinal = pd.concat([i for i in d], axis=1)

    Concated = pd.concat([df[diff], dfFinal[dfFinal.columns.values]], axis=1, sort=True)

    return Concated


def scale_single_min_max(val):
    minimum = min(val)
    maximum = max(val)
    return pd.DataFrame([round((i - minimum) / (maximum - minimum), 2) for i in val],
                        columns=[val.name + "_scalled_min_max"])


def scale_single_std(val):
    means = np.mean(val)
    stds = np.std(val)
    return pd.DataFrame([round((i - means) / stds, 2) for i in val],
                        columns=[val.name + "_scalled_std"])


def __version__():
    print("2.0.3")
