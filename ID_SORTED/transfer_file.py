import os
import pandas as pd

if __name__=='__main__':
    source_folder = "unused/KSS_sort"
    dest_folder = "KSS_sort_45/"
    extract_folder = "results_45s/"
    for pre_post in os.listdir(source_folder):
        os.mkdir(dest_folder+pre_post)
        for KSS_val in os.listdir(source_folder+"/"+pre_post):
            os.mkdir(dest_folder+pre_post+"/"+KSS_val)
            if len(os.listdir(source_folder+"/"+pre_post+"/"+KSS_val+"/"+"cap"+"/"))==0:
                    continue
            else:
                os.mkdir(dest_folder+pre_post+"/"+KSS_val+"/"+"cap")
                os.mkdir(dest_folder+pre_post+"/"+KSS_val+"/"+"emg")
                os.mkdir(dest_folder+pre_post+"/"+KSS_val+"/"+"ecg")
                os.mkdir(dest_folder+pre_post+"/"+KSS_val+"/"+"eeg")
                for sample in os.listdir(source_folder+"/"+pre_post+"/"+KSS_val+"/"+"ecg"):
                    try:
                        if sample[2] == "_":
                            print("hi")
                            id = sample[0:3]
                            ecg_string = id + "ecg_features.csv"
                            emg_string = id + "emg_features.csv"
                            eeg_string = id + "eeg_features.csv"
                            cap_string = id + "cap_features.csv"
                            
                            ecg_file = pd.read_csv(extract_folder + "ecg_45_csv" + "/" + ecg_string)
                            emg_file = pd.read_csv(extract_folder + "emg_45_csv" + "/" + emg_string)
                            eeg_file = pd.read_csv(extract_folder + "eeg_45_csv" + "/" + eeg_string)
                            cap_file = pd.read_csv(extract_folder + "cap_45_csv" + "/" + cap_string)

                            ecg_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"ecg"+ "/"+ecg_string)
                            emg_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"emg"+ "/"+emg_string)
                            eeg_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"eeg"+ "/"+eeg_string)
                            cap_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"cap"+ "/"+cap_string)
                        elif sample[3] == "_":
                            print("hello")
                            id = sample[0:4]
                            ecg_string = id + "ecg_features.csv"
                            emg_string = id + "emg_features.csv"
                            eeg_string = id + "eeg_features.csv"
                            cap_string = id + "cap_features.csv"
                            
                            ecg_file = pd.read_csv(extract_folder + "ecg_45_csv" + "/" + ecg_string)
                            emg_file = pd.read_csv(extract_folder + "emg_45_csv" + "/" + emg_string)
                            eeg_file = pd.read_csv(extract_folder + "eeg_45_csv" + "/" + eeg_string)
                            cap_file = pd.read_csv(extract_folder + "cap_45_csv" + "/" + cap_string)

                            ecg_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"ecg"+ "/"+ecg_string)
                            emg_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"emg"+ "/"+emg_string)
                            eeg_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"eeg"+ "/"+eeg_string)
                            cap_file.to_csv(dest_folder+pre_post+"/"+KSS_val+"/"+"cap"+ "/"+cap_string)
                    except FileNotFoundError:
                        continue