# Cane - Categorical Attribute traNsformation Environment 
CANE is a simpler but powerful preprocessing method for machine learning. 


At the moment offers 2 preprocessing methods:

The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a single level as presented in (https://doi.org/10.1109/IJCNN.2019.8851888).

The Inverse Document Frequency (IDF) codifies the levels into frequency values, where the closer to 0 means, the more frequent it is (https://ieeexplore.ieee.org/document/8710472). 

Finally it also has implemented a simpler standard One-Hot-Encoding method.

For questions and other suggestions contact luis.matos@dsi.uminho.pt



# Example
``` python
import pandas as pd
import cane
x = ["a", "a", "a", "b", "b", "b", "b", "b", "c", "c", "c", "c", "c", "c", "c", "d"] * 10000  # replicates the list
df = pd.DataFrame({"x": x, "x2": x, "x3": x, "x4": x, "x5": x, "x6": x, "x7": x, "x8": x, "x9": x, "x10": x, "x11": x,
                   "x12": x})  # df with 12 columns

dataPCP, dicionary = cane.pcp(df)  # uses the PCP method and only 1 core
dataPCP, dicionary = cane.pcp(df, n_coresJob=2)  # uses the PCP method and only 2 cores
dataIDF = cane.idf(df)  # uses the IDF method and only 1 core
dataIDF = cane.idf(df, n_coresJob=2)  # uses the IDF method and only 2 core

dataH = cane.one_hot(df)  # without a column prefixer
dataH2 = cane.one_hot(df, column_prefix='column')  # it will use the original column name prefix
# (useful for when dealing with id number columns)
dataH3 = cane.one_hot(df, column_prefix='customColName')  # it will use a custom prefix defined by
# the value of the column_prefix
dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2)  # it will use the original column name prefix
# (useful for when dealing with id number columns)
# with 2 cores
```