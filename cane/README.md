# Categorical Arrangement of Nominal variables Environment (CANE) 
CANE is a simpler but powerful preprocessing method for machine learning. 


At the moment offers 2 preprocessing methods:

The Percentage Categorical Pruned (PCP) merges all least frequent levels (summing up to perc percent) into a single level as presented in (https://doi.org/10.1109/IJCNN.2019.8851888).

The Inverse Document Frequency (IDF) codifies the levels into frequency values, where the closer to 0 means, the more frequent it is (https://ieeexplore.ieee.org/document/8710472). 

For questions and other suggestions contact luis.matos@dsi.uminho.pt



# example
``` python
import pandas as pd
import cane
x=["a","a","a","b","b","b","b","b","c","c","c","c","c","c","c","d"]
df=pd.DataFrame({"x":x,"x2":x})
dataPCP, dicionary = cane.PCP_Data(df.copy()) #always send a copy of the dataframe
dataIDF = cane.IDF_Data(df.copy()) #always send a copy of the dataframe
```