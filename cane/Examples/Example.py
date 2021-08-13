import pandas as pd
import cane
import timeit
import numpy as np
if __name__ == "__main__":
    x = [k for s in ([k] * n for k, n in [('a', 30000), ('b', 20000), ('c', 10000)]) for k in
         s]
    x2 = [k for s in ([k] * n for k, n in [('z', 15000), ('b', 10000), ('a', 20000)]) for k in
         s]
    df = pd.DataFrame({f'x{i}': x for i in range(1, 130)})
    df2 = pd.DataFrame({f'x{i}': x2 for i in range(1, 130)})

    ##idf
    datatransformed = cane.idf(df)
    dicionaryIDF = cane.idfDicionary(df, datatransformed, datatransformed.columns) #geração de dicionário

    datatransformed = cane.pcp(df)
    dicionaryPCP = cane.PCPDicionary(df, datatransformed, datatransformed.columns)  # geração de dicionário