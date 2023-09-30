import pandas as pd
from mrmr import mrmr_regression
from sklearn.feature_selection import SelectKBest, f_regression

if __name__ =="__main__":
    # Loading Dataset as pandas
    dataset = pd.read_csv("features_and_labels_30s.csv")
    X = dataset.iloc[:,0:-2]
    y = dataset.iloc[:,-1]

    # MRMR selection
    selected_features_MRMR = mrmr_regression(X=X,y=y, K=15)
    print(selected_features_MRMR)

    # SelecKBest selection
    selector = SelectKBest(f_regression, k=15)
    selector.fit(X, y)
    cols_idxs = selector.get_support(indices=True)
    features_df_new = X.iloc[:,cols_idxs]
    print(features_df_new.columns.to_list())
