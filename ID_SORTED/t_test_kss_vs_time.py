from t_test import remove_outlier
import pandas as pd
import numpy as np
import math
import scipy

if "__name__" == "__main__":
    df_main = pd.read_excel("kss_vs_time.xlsx")