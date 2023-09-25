import pandas as pd
import numpy as np
import math
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outlier(arr):
    original_len = np.array(arr).shape[0]
    new_arr = []
    for i in range(original_len):
        if not math.isnan(arr[i]):
            new_arr.append(arr[i])
    new_arr = np.array(new_arr)
    # print(new_arr)

    p95, p5 = np.percentile(new_arr, [95, 5])
    new_arr = new_arr[new_arr < p95]
    new_arr = new_arr[new_arr > p5]

    while new_arr.shape[0] < original_len:
        new_arr = np.append(new_arr, np.nan)
    return new_arr

## CODE TO RUN

if __name__ == "__main__":
    df_main = pd.read_csv("features_vs_time_of_day.csv")
    df_out = pd.DataFrame()
    row_headers = ["n_sameple afternoon", "n_sameple night", "mean_afternoon", "std_afternoon", "mean_night", "std_night","t-val", "p-val","alternative", "usable?"]
    row_col = pd.DataFrame({"feature_used":row_headers})
    df_out = pd.concat([df_out, row_col], axis = 1)
    counter = 0
    for i in range(0,len(df_main.columns),3):
        print(i)
        if i==381:
            continue
        # Obtaining the data
        afternoon_header = df_main.columns[i+1]
        night_header = df_main.columns[i+2]

        cleaned_afternoon_data = remove_outlier(df_main[afternoon_header].to_numpy()).tolist()
        cleaned_night_data = remove_outlier(df_main[night_header].to_numpy()).tolist()

        afternoon_col = pd.DataFrame({afternoon_header:cleaned_afternoon_data})
        night_col = pd.DataFrame({night_header:cleaned_night_data})
        df = pd.concat([afternoon_col, night_col], axis=1)
        # print(df)

        # Reshape the dataframe suitable for statsmodels package 
        # df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=[afternoon_header, night_header])
        # df_melt.columns = ['index', 'Drowsiness Level', afternoon_header[0:-2]]

        # Showing the box plots
        # ax = sns.boxplot(x="Drowsiness Level", y=afternoon_header[0:-2], data=df_melt, color='#99c2a2')
        # # ax = sns.swarmplot(x="Drowsiness Level", y=afternoon_header[0:-6], data=df_melt, color='#7d0013')
        # title_string = afternoon_header[0:-2] + " vs Drowsiness Level"
        # ax.set_title(title_string)    

        print('done')
        cleaned_afternoon_data_nan = np.array(cleaned_afternoon_data)
        cleaned_afternoon_data = []
        for i in range(cleaned_afternoon_data_nan.shape[0]):
            if not np.isnan(cleaned_afternoon_data_nan[i]):
                cleaned_afternoon_data.append(cleaned_afternoon_data_nan[i])

        cleaned_night_data_nan = np.array(cleaned_night_data)
        cleaned_night_data = []
        for i in range(cleaned_night_data_nan.shape[0]):
            if not np.isnan(cleaned_night_data_nan[i]):
                cleaned_night_data.append(cleaned_night_data_nan[i])

        # Conducting t-tests
        t_val_less, p_val_less = scipy.stats.ttest_ind(np.array(cleaned_afternoon_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', alternative='less', permutations=None, random_state=None, trim=0)
        t_val_more, p_val_more = scipy.stats.ttest_ind(np.array(cleaned_afternoon_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', alternative='greater', permutations=None, random_state=None, trim=0)
        t_val_two, p_val_two = scipy.stats.ttest_ind(np.array(cleaned_afternoon_data), np.array(cleaned_night_data), axis=0, equal_var=True, nan_policy='omit', permutations=None, random_state=None, trim=0)

        p_val = np.min([p_val_less, p_val_more, p_val_two])
        id_p = np.argmin([p_val_less, p_val_more, p_val_two])
        t_vec = [t_val_less, t_val_more, t_val_two]
        t_val = t_vec[id_p]

        if id_p==0:
            print_alternative = "less"
        elif id_p==1:
            print_alternative = "greater"
        else:
            print_alternative = "two sided"

        flagger = False
        if p_val<0.05:
            flagger = True
            counter +=1

        # Printing out values for visualising
        mean_afternoon = np.mean(cleaned_afternoon_data)
        std_afternoon = np.std(cleaned_afternoon_data)
        mean_night = np.mean(cleaned_night_data)
        std_night = np.std(cleaned_night_data)
        # print("For " + afternoon_header[0:-6] + " data,")
        # print("kss6 mean:" + str(mean_afternoon), "std: " + str(std_afternoon))
        # print("kss7 mean:" + str(mean_night), "std: " + str(std_night))
        # print("t-value: " + str(t_val), "p-value" + str(p_val))

        # plt.show()

        # Saving the values
        array_to_save = [len(cleaned_afternoon_data), len(cleaned_night_data), mean_afternoon, std_afternoon, mean_night, std_night, t_val, p_val, print_alternative, flagger]
        new_col = pd.DataFrame({afternoon_header[0:-2]: array_to_save})
        df_out = pd.concat([df_out, new_col], axis=1)

        df = 0

    print(counter)
    df_out.to_csv("results_15s/t_test_an.csv")
