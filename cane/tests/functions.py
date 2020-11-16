# This will be an example file you of your custom function (e.g., "functions.py")
import pandas as pd
import numpy as np
import cane


def customFunc(val):
    return pd.DataFrame([round((i - 1) / 3, 2) for i in val], columns=[val.name + "_custom_scalled_min_max"])
