import pandas as pd
import cane
import timeit
import numpy as np

x = [k for s in ([k] * n for k, n in [('a', 30000), ('b', 50000), ('c', 70000), ('d', 10000), ('e', 1000)]) for k in s]
df = pd.DataFrame({f'x{i}': x for i in range(1, 130)})

print("PCP Examples")

print("PCP Baseline")
dataPCP = cane.pcp(df)  # uses the PCP method and only 1 core with perc == 0.05 for all columns
print(dataPCP)
print("PCP Baseline with multicore")
dataPCP = cane.pcp(df, n_coresJob=2)  # uses the PCP method and only 2 cores for all columns
print(dataPCP)
print("PCP Baseline with multicore showing progressbar")
dataPCP = cane.pcp(df, n_coresJob=2, disableLoadBar=False)  # With Progress Bar for all columns
print(dataPCP)
print("PCP Baseline with multicore for specific columns")
dataPCP = cane.pcp(df, n_coresJob=2, disableLoadBar=False,
                   columns_use=["x1", "x2"])  # With Progress Bar and specific columns

print(dataPCP)
# dicionary with the transformed data
print("PCP dicionary example")
dataPCP = cane.pcp(df)
dicionary = cane.PCPDictionary(dataset=dataPCP, columnsUse=dataPCP.columns,
                               targetColumn=None)  # no target feature to avoid going into dictionary
print("PCP dicionary")
print(dicionary)



print("IDF Examples")

print("IDF Baseline")
dataIDF = cane.idf(df)  # uses the IDF method and only 1 core for all columns 
print(dataIDF)
print("IDF Baseline with multicore")
dataIDF = cane.idf(df, n_coresJob=2)  # uses the IDF method and only 2 core for all columns
print(dataIDF)
print("IDF Baseline with multicore showing progressbar")
dataIDF = cane.idf(df, n_coresJob=2, disableLoadBar=False)  # With Progress Bar for all columns
print(dataIDF)
dataIDF = cane.idf(df, n_coresJob=2, disableLoadBar=False, columns_use=["x1", "x2"])  # specific columns
print("IDF Baseline with multicore for specific columns")
dataIDF = cane.idf_multicolumn(df, columns_use=["x1", "x2"])  # aplication of specific multicolumn setting IDF
print(dataIDF)

print("IDF dicionary example")
idfDicionary = cane.idfDictionary(Original=df, Transformed=dataIDF, columns_use=["x1", "x2"]
                                  , targetColumn=None)  # following the example above of the 2 columns
print("IDF dicionary")
print(idfDicionary)


print("1Hot Examples")

print("1Hot Baseline")
dataH = cane.one_hot(df)  # without a column prefixer
print(dataH)
print("1Hot Baseline original with column name prefix")
dataH2 = cane.one_hot(df, column_prefix='column')  # it will use the original column name prefix
# (useful for when dealing with id number columns)
print(dataH2)

print("1Hot with a custom prefix defined")
dataH3 = cane.one_hot(df, column_prefix='customColName')  # it will use a custom prefix defined by
# the value of the column_prefix
print(dataH3)

print("1Hot Baseline that will use the original column name prefix")
dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2)  # it will use the original column name prefix
# (useful for when dealing with id number columns)
# with 2 cores
print(dataH4)

print("1Hot With Progress Bar Active with 2 cores")

dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2
                      , disableLoadBar=False)  # With Progress Bar Active with 2 cores
print(dataH4)

print("1Hot With Progress Bar specific columns")

dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2
                      , disableLoadBar=False, columns_use=["x1", "x2"])  # With Progress Bar specific columns!
print(dataH4)

# specific example with multicolumn
x2 = [k for s in ([k] * n for k, n in [('a', 50),
                                       ('b', 10),
                                       ('c', 20),
                                       ('d', 15),
                                       ('e', 5)]) for k in s]

x3 = [k for s in ([k] * n for k, n in [('a', 40),
                                       ('b', 20),
                                       ('c', 1),
                                       ('d', 1),
                                       ('e', 38)]) for k in s]
df2 = pd.concat([pd.DataFrame({f'x{i}': x2 for i in range(1, 3)}), pd.DataFrame({f'y{i}': x3 for i in range(1, 3)})],
                axis=1)
dataPCP = cane.pcp(df2, n_coresJob=2, disableLoadBar=False)
print("normal PCP \n", dataPCP)
dataPCP2 = cane.pcp_multicolumn(df2, columns_use=["x1", "y1"])  # aplication of specific multicolumn setting PCP
print("multicolumn PCP \n", dataPCP2)

dataIDF = cane.idf(df2, n_coresJob=2, disableLoadBar=False, columns_use=["x1", "y1"])  # specific columns
print("normal idf \n", dataIDF)
dataIDF2 = cane.idf_multicolumn(df2, columns_use=["x1", "y1"])  # aplication of specific multicolumn setting IDF
print("multicolumn idf \n", dataIDF2)

# Time Measurement in 10 runs
print("Time Measurement in 10 runs (unicore)")
OT = timeit.timeit(lambda: cane.one_hot(df, column_prefix='column', n_coresJob=1), number=10)
IT = timeit.timeit(lambda: cane.idf(df), number=10)
PT = timeit.timeit(lambda: cane.pcp(df), number=10)
print("One-Hot Time:", OT)
print("IDF Time:", IT)
print("PCP Time:", PT)

# Time Measurement in 10 runs (multicore)
print("Time Measurement in 10 runs (multicore)")
OTM = timeit.timeit(lambda: cane.one_hot(df, column_prefix='column', n_coresJob=10), number=10)
ITM = timeit.timeit(lambda: cane.idf(df, n_coresJob=10), number=10)
PTM = timeit.timeit(lambda: cane.pcp(df, n_coresJob=10), number=10)
print("One-Hot Time Multicore:", OTM)
print("IDF Time Multicore:", ITM)
print("PCP Time Multicore:", PTM)

# IDF with pyspark configs
import cane
from pyspark.sql import SparkSession

print("Spark Example")
# Create PySpark SparkSession
spark = SparkSession.builder.getOrCreate()
# Create PySpark DataFrame from Pandas
sparkDF = spark.createDataFrame(df)
cols = sparkDF.columns
DFIDF, idf = spark_idf_multicolumn(sparkDF, cols) #with spark
print("Spark Example", DFIDF.show(20))
