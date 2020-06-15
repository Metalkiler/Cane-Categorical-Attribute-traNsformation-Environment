# Cane - Categorical Attribute traNsformation Environment 
CANE is a simpler but powerful preprocessing method for machine learning. 


At the moment offers 3 preprocessing methods:

--> The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to percentage value) into a single level as presented in (https://doi.org/10.1109/IJCNN.2019.8851888), which, for example, can be "Others" category. It can be useful when dealing with several amounts of categorical information (e.g., city data).

--> The Inverse Document Frequency (IDF) codifies the categorical levels into frequency values, where the closer to 0 means, the more frequent it is (https://ieeexplore.ieee.org/document/8710472). 

--> Finally it also has implemented a simpler standard One-Hot-Encoding method.


 

# Instalation

To install this package please run the following command:

``` cmd
pip install cane 

```
Any feedback would be appreciated.


For questions and other suggestions contact luis.matos@dsi.uminho.pt


# Example
``` python
import pandas as pd
import cane

x = [k for s in ([k] * n for k, n in [('a', 30000), ('b', 50000), ('c', 70000), ('d', 10000), ('e', 1000)]) for k in s]

df = pd.DataFrame({f'x{i}' : x for i in range(1, 13)})

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
