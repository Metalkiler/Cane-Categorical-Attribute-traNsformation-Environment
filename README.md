# Cane - Categorical Attribute traNsformation Environment
[![Downloads](https://pepy.tech/badge/cane)](https://pepy.tech/project/cane) [![Downloads](https://pepy.tech/badge/cane/month)](https://pepy.tech/project/cane) [![Downloads](https://pepy.tech/badge/cane/week)](https://pepy.tech/project/cane)

CANE is a simpler but powerful preprocessing method for machine learning.

At the moment offers some preprocessing methods:

--> The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to "perc" percent) into a single level as presented in (<https://doi.org/10.1109/IJCNN.2019.8851888>), which, for example, can be "Others" category. It can be useful when dealing with several amounts of categorical information (e.g., city data).

An example of this can be viewed by the following pdf:
 
<p><a href="https://github.com/Metalkiler/Cane-Categorical-Attribute-traNsformation-Environment/blob/master/cities.pdf">View PDF</a>.</p>
  

Which the 1,000 highest frequency values (decreasing order) for the user city attribute for the TEST traffic data (which contains a total of 10,690 levels).
For this attribute and when <img src="https://render.githubusercontent.com/render/math?math=P=10">, PCP selects only the most frequent 688 levels (dashed vertical line) merging the other 10,002 infrequent levels into the "Others" label.

This method results in 689 binary inputs, which is much less than the 10690 binary inputs required by the standard one-hot transform (reduction of <img src="https://render.githubusercontent.com/render/math?math=\frac{10690-689}{10690}=94"> percentage points).

--> The Inverse Document Frequency (IDF) codifies the categorical levels into frequency values, where the closer to 0 means, the more frequent it is (<https://ieeexplore.ieee.org/document/8710472>).

--> Implementation of a simpler One-Hot-Encoding method.

--> Minmax and Standard scaler (based on sklearn functions) with column selection and multicore support. Also, it is possible to apply these transformations to specific columns only instead of the full dataset (follow the example). However it only works with numerical data (e.g., MSE, decision scores)

--> You can also provide a custom scaler version of your own! (check example)




Future Function ideas:
-- 
MultiColumn scale (based on the implementation of IDF and PCP)
Scaling of IDF values (normalized IDF) 




# Installation

To install this package please run the following command

``` cmd
pip install cane
```

# New
Version 2.0.4:

[x] - Dicionary for IDF transformation and general improvements



# Suggestions and feedback

Any feedback will be appreciated.
For questions and other suggestions contact luis.matos@dsi.uminho.pt
Found any bugs? Post Them on the github page of the project! (https://github.com/Metalkiler/Cane-Categorical-Attribute-traNsformation-Environment)

Thanks for the support!

# Example

``` python
import pandas as pd
import cane
import timeit
import numpy as np
x = [k for s in ([k] * n for k, n in [('a', 30000), ('b', 50000), ('c', 70000), ('d', 10000), ('e', 1000)]) for k in s]
df = pd.DataFrame({f'x{i}' : x for i in range(1, 130)})

dataPCP = cane.pcp(df)  # uses the PCP method and only 1 core with perc == 0.05 for all columns
dataPCP = cane.pcp(df, n_coresJob=2)  # uses the PCP method and only 2 cores for all columns
dataPCP = cane.pcp(df, n_coresJob=2,disableLoadBar = False)  # With Progress Bar for all columns
dataPCP = cane.pcp(df, n_coresJob=2,disableLoadBar = False, columns_use = ["x1","x2"])  # With Progress Bar and specific columns



#dicionary with the transformed data
dataPCP = cane.pcp(df) 
dicionary = cane.PCPDictionary(dataset = dataPCP, columnsUse = dataPCP.columns,
                              targetColumn = None) #no target feature to avoid going into dictionary
print(dicionary)

dataIDF = cane.idf(df)  # uses the IDF method and only 1 core for all columns 
dataIDF = cane.idf(df, n_coresJob=2)  # uses the IDF method and only 2 core for all columns
dataIDF = cane.idf(df, n_coresJob=2,disableLoadBar = False)  # With Progress Bar for all columns
dataIDF = cane.idf(df, n_coresJob=2,disableLoadBar = False, columns_use = ["x1","x2"]) # specific columns
dataIDF = cane.idf_multicolumn(df, columns_use = ["x1","x2"])  # aplication of specific multicolumn setting IDF

idfDicionary = cane.idfDictionary(Original = df, Transformed = dataIDF, columns_use = ["x1","x2"]
                                , targetColumn=None) #following the example above of the 2 columns
                                
                                
dataH = cane.one_hot(df)  # without a column prefixer
dataH2 = cane.one_hot(df, column_prefix='column')  # it will use the original column name prefix
# (useful for when dealing with id number columns)
dataH3 = cane.one_hot(df, column_prefix='customColName')  # it will use a custom prefix defined by
# the value of the column_prefix
dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2)  # it will use the original column name prefix
# (useful for when dealing with id number columns)
# with 2 cores

dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2
                      ,disableLoadBar = False)  # With Progress Bar Active with 2 cores

dataH4 = cane.one_hot(df, column_prefix='column', n_coresJob=2
                      ,disableLoadBar = False,columns_use = ["x1","x2"])  # With Progress Bar specific columns!



#specific example with multicolumn
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
df2 = pd.concat([pd.DataFrame({f'x{i}' : x2 for i in range(1, 3)}),pd.DataFrame({f'y{i}' : x3 for i in range(1, 3)})], axis=1)
dataPCP = cane.pcp(df2, n_coresJob=2,disableLoadBar = False)
print("normal PCP \n",dataPCP)
dataPCP2 = cane.pcp_multicolumn(df2, columns_use = ["x1","y1"])  # aplication of specific multicolumn setting PCP
print("multicolumn PCP \n",dataPCP2)

dataIDF = cane.idf(df2, n_coresJob=2,disableLoadBar = False, columns_use = ["x1","y1"]) # specific columns
print("normal idf \n",dataIDF)
dataIDF2 = cane.idf_multicolumn(df2, columns_use = ["x1","y1"])  # aplication of specific multicolumn setting IDF
print("multicolumn idf \n",dataIDF2)



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

# Scaler Example with cane

These examples present the usage of cane with the standard methods (standard scaler e min max scaler).
Also, it is presented how to implement a custom scaler function of your own with cane!
``` python
#New Scaler Function 



dfNumbers = pd.DataFrame(np.random.randint(0,100000,size=(100000, 12)), columns=list('ABCDEFGHIJKL'))
cane.scale_data(dfNumbers, n_cores = 3, scaleFunc="min_max") # all columns using 3 cores
cane.scale_data(dfNumbers, column=["A","B"], n_cores = 3, scaleFunc="min_max") # scale specific columns
cane.scale_data(dfNumbers, column=["A","B"], n_cores = 3, scaleFunc="std") #standard Scaler



#####################Custom Function Example#######################

#This will be an example file you of your custom function (e.g., "functions.py")
import pandas as pd
import numpy as np
import cane 

def customFunc(val):
       return pd.DataFrame([round((i - 1) / 3, 2) for i in val],columns=[val.name + "_custom_scalled_function])



### This is will be your main script

from functions import *
# with a custom function to apply to data:
if __name__ == "__main__":
    dfNumbers = pd.DataFrame(np.random.randint(0,100000,size=(100000, 12)), columns=list('ABCDEFGHIJKL'))
    cane.scale_data(dfNumbers, column=["A","B"], n_cores = 3, scaleFunc="custom", customfunc = customFunc)
    

```


