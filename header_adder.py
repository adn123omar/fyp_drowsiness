import os
import pandas as pd
import csv

if __name__ == "__main__":
    for file in os.listdir('recordings'):
        data = pd.read_csv("recordings/" + file + "/bci_clipped.csv", header=None)
        column_list = ["Sample Index", "EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3", "EXG Channel 4", "EXG Channel 5", "EXG Channel 6", "EXG Channel 7", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2", "Other", "Other", "Other", "Other", "Other", "Other", "Other", "Analog Channel 0", "Analog Channel 1", "Analog Channel 2", "Timestamp", "Other", "Timestamp (Formatted)"]
        
        with open("recordings/" + file + "/bci1_clipped.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(column_list)
            for i in range(len(data)):
                line = data.values[i,:][0].strip('" ')
                line = line.replace(" ","")
                line = line.split(",")
                # print(line)
                # print(new_data.shape)
                writer.writerow(line)
            
