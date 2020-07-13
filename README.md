# Cane - Categorical Attribute traNsformation Environment

CANE is a simpler but powerful preprocessing method for machine learning.

At the moment offers 3 preprocessing methods:

--> The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to "perc" percent) into a single level as presented in (<https://doi.org/10.1109/IJCNN.2019.8851888>), which, for example, can be "Others" category. It can be useful when dealing with several amounts of categorical information (e.g., city data).

An example of this can be viewed by the following figure:

![cities](cities.pdf "Example case with PCP")

Which the 1,000 highest frequency values (decreasing order) for the user city attribute for the TEST traffic data (which contains a total of 10,690 levels).
For this attribute and when $P=10\%$, PCP selects only the most frequent 688 levels (dashed vertical line) merging the other 10,002 infrequent levels into the ``Others'' label.
This method results in 689 binary inputs, which is much less than the 10690 binary inputs required by the standard one-hot transform (reduction of $\frac{10690-689}{10690}=94$\%)

--> The Inverse Document Frequency (IDF) codifies the categorical levels into frequency values, where the closer to 0 means, the more frequent it is (<https://ieeexplore.ieee.org/document/8710472>).

--> Finally it also has implemented a simpler standard One-Hot-Encoding method.

# Installation

To install this package please run the following command

``` cmd
pip install cane

```

# Suggestions and feedback

Any feedback will be appreciated.
For questions and other suggestions contact luis.matos@dsi.uminho.pt


# Example

``` python
import pandas as pd
import cane
import timeit
x = [k for s in ([k] * n for k, n in [('a', 30000), ('b', 50000), ('c', 70000), ('d', 10000), ('e', 1000)]) for k in s]
df = pd.DataFrame({f'x{i}' : x for i in range(1, 130)})

dataPCP = cane.pcp(df)  # uses the PCP method and only 1 core with perc == 0.05
dataPCP = cane.pcp(df, n_coresJob=2)  # uses the PCP method and only 2 cores
dataPCP = cane.pcp(df, n_coresJob=2,disableLoadBar = False)  # With Progress Bar

#dicionary with the transformed data

dicionary = cane.dic_pcp(dataPCP)
print(dicionary)

dataIDF = cane.idf(df)  # uses the IDF method and only 1 core
dataIDF = cane.idf(df, n_coresJob=2)  # uses the IDF method and only 2 core
dataIDF = cane.idf(df, n_coresJob=2,disableLoadBar = False)  # With Progress Bar

dataH = cane.one_hot(df)  # without a column prefixer
dataH2 = cane.one_hot(df, column_prefix='column')  # it will use the original column name prefix
# (useful for when dealing with id number columns)
dataH3 = cane.one_hot(df, column_prefix='customColName')  # it will use a custom prefix defined by
# the value of the column_prefix
dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2)  # it will use the original column name prefix
# (useful for when dealing with id number columns)
# with 2 cores

dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2
                      ,disableLoadBar = False)  # With Progress Bar Active!
# with 2 cores

#Time Measurement in 10 runs
print("Time Measurement in 10 runs (unicore)")
OT = timeit.timeit(lambda:cane.one_hot(df, column_prefix='column', n_coresJob=1),number = 10)
IT = timeit.timeit(lambda:cane.idf(df),number = 10)
PT = timeit.timeit(lambda:cane.pcp(df),number = 10)
print("One-Hot Time:",OT)
print("IDF Time:",IT)
print("PCP Time:",PT)

#Time Measurement in 10 runs (multicore)
print("Time Measurement in 10 runs (multicore)")
OTM = timeit.timeit(lambda:cane.one_hot(df, column_prefix='column', n_coresJob=10),number = 10)
ITM = timeit.timeit(lambda:cane.idf(df,n_coresJob=10),number = 10)
PTM = timeit.timeit(lambda:cane.pcp(df,n_coresJob=10),number = 10)
print("One-Hot Time Multicore:",OTM)
print("IDF Time Multicore:",ITM)
print("PCP Time Multicore:",PTM)


```


