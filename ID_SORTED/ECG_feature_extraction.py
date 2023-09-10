from __future__ import division
import numpy as np
#from pyhrv.hrv import hrv
from ecgdetectors import Detectors
from ecg_functions import *
import pandas as pd
import hrvanalysis
import tsfel
import os
import math
from hrv.filters import quotient
import neurokit2 as nk
from scipy import stats
from scipy.signal import butter,filtfilt, find_peaks
import matplotlib.pyplot as plt



def featureExtraction(rri, ECG, fs):
    time_domain = hrvanalysis.get_time_domain_features(rri)
    frequency_domain = hrvanalysis.get_frequency_domain_features(rri)
    other = hrvanalysis.get_csi_cvi_features(rri)

    HRV = [time_domain['mean_nni'], time_domain['cvnni']]
    Time_Features = [time_domain['cvsd'], time_domain['max_hr'], time_domain['mean_hr'], time_domain['median_nni'], time_domain['min_hr'], time_domain['nni_20'], time_domain['nni_50'], time_domain['pnni_20'],
                                                    time_domain['pnni_50'], time_domain['range_nni'], time_domain['rmssd'], time_domain['sdnn'], time_domain['sdsd'], time_domain['std_hr']]

    Frequency_Features = [frequency_domain['lf'], frequency_domain['hf'], frequency_domain['lf_hf_ratio'], frequency_domain['lfnu'], frequency_domain['hfnu'], frequency_domain['total_power'], frequency_domain['vlf']]
    Other_Features = [other['csi'], other['cvi']]

    return np.float64(np.concatenate((np.array(HRV), np.array(Time_Features), np.array(Frequency_Features), np.array(Other_Features))))

def WE(y, level=4, wavelet='coif2'):
    from math import log
    n = len(y)

    sig = y

    ap = {}

    for lev in range(0, level):
        (y, cD) = pywt.dwt(y, wavelet)
        ap[lev] = y

    # Energy

    Enr = np.zeros(level)
    for lev in range(0, level):
        Enr[lev] = np.sum(np.power(ap[lev], 2)) / n

    Et = np.sum(Enr)

    Pi = np.zeros(level)
    for lev in range(0, level):
        Pi[lev] = Enr[lev] / Et

    we = - np.sum(np.dot(Pi, np.log(Pi)))

    return we


def ECG_features(a, win, fs, ECG, detectors1):
    c = ECG[range(a * win, a * win + win)].T
    r_peaks1 = np.array(detectors1.pan_tompkins_detector(c))
    # Calculate RRI
    rri1 = (np.diff(r_peaks1) / fs) * 10 ** 3
    # Filter RRI to remove big peaks
    filt_rri = np.array(quotient(rri1))
    d = np.std(filt_rri)
    b0 = np.median(filt_rri)
    filt_rri[np.where(filt_rri > 1200)] = b0
    filt_rri[np.where(filt_rri < 600)] = b0

    print("r_peaks1")
    print(len(r_peaks1))
    ecg_rate = nk.ecg_rate(r_peaks1, sampling_rate=fs, desired_length=len(c))
    print("ecg_rate")
    print(ecg_rate)

    # Breathing
    edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs)
    # Clean signal
    cleaned = nk.rsp_clean(edr, sampling_rate=fs)

    print("cleaned")
    print(cleaned)
    # Extract peaks

    try:
        df, peaks_dict = nk.rsp_peaks(cleaned)
    except IndexError:
        peaks_dict = float('nan')

    info = nk.rsp_fixpeaks(peaks_dict)
    formatted = pd.DataFrame({"RSP_Raw": edr, "RSP_Clean": cleaned})
    # Extract rate
    try:

        rsp_rate = nk.rsp_rate(formatted, sampling_rate=fs)
    except IndexError:
        rsp_rate = [float('nan')]
    if math.isnan(rsp_rate[0]) == False:
        #print(rsp_rate)
        rsp = np.nanmean(rsp_rate)
        rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=fs, show=False)
        RRV_median = rrv['RRV_MedianBB'][0]
        RRV_mean = rrv['RRV_MeanBB'][0]
        RRV_ApEn = rrv['RRV_ApEn'][0]
    else:
        rsp = float('nan')
        RRV_median = float('nan')
        RRV_mean = float('nan')
        RRV_ApEn = float('nan')


    # HR and Entropy
    HR = np.nanmedian(ecg_rate)
    HR_std = np.std(ecg_rate)
    Sh_Ent = nk.entropy_shannon(filt_rri)[0]
    Approx = nk.entropy_approximate(filt_rri)[0]
    #sample = nk.entropy_sample(filt_rri) DOES NOT WORK
    fuzzy = nk.entropy_fuzzy(filt_rri)[0]
    #Multiscale = nk.entropy_multiscale(filt_rri) DOES NOT WORK
    wave_ent = WE(filt_rri)
    HRV_mean = np.nanmedian(filt_rri)
    HRV_std = np.nanstd(filt_rri)
    HRV_kurt = stats.kurtosis(filt_rri, nan_policy='omit')
    HRV_var = np.nanvar(filt_rri)
    HRV_skew = stats.skew(filt_rri, nan_policy='omit')

    #time_domain = hrvanalysis.get_time_domain_features(filt_rri)
    frequency_domain = hrvanalysis.get_frequency_domain_features(filt_rri)
    # other = hrvanalysis.get_csi_cvi_features(filt_rri)
    #HR = time_domain['mean_hr']
    if HR > 120:
        HR = np.NaN
    elif HR < 40:
        HR = np.NaN
    VLF = frequency_domain['vlf']
    LF = frequency_domain['lf']
    HF = frequency_domain['hf']
    LF_HF = frequency_domain['lf_hf_ratio']
    Power = frequency_domain['total_power']

    ECG_features_out = np.hstack((HR, VLF, LF, HF, LF_HF, Power, rsp, RRV_median, RRV_mean, RRV_ApEn, HR_std, Sh_Ent,
                                  Approx, fuzzy, wave_ent, HRV_mean, HRV_std, HRV_kurt, HRV_var, HRV_skew))
    return ECG_features_out

## CODE TO DETECT FEATURES 
fs = 125
wind_size = 15
window_overlap = 14
wind_n_elements = fs * wind_size
# ecg_features_list = ["HR", "VLF", "LF", "HF", "LF_HF", "Power", "rsp", "RRV_median", "RRV_mean", "RRV_ApEn", "HR_std", "Sh_Ent",
#                                 "Approx", "fuzzy", "wave_ent", "HRV_mean", "HRV_std", "HRV_kurt", "HRV_var", "HRV_skew"]
    
# detector = Detectors(fs)
ecg_features_list = ["mean_nni", "cvnni", "cvsd", "max_hr", "mean_hr", "median_nni", "min_hr", "nni_20", "nni_50", "pnni_20", "pnni_50", "range_nni",
                              "rmssd", "fuzzy", "sdnn", "sdsd", "std_hr", "lf", "hf", "lf_hf_ratio","lfnu", "total_power", "vlf","csi","cvi"]

# folder path
dir_path = "bci_filtered"
    
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        df = pd.DataFrame(columns=ecg_features_list)

        file_dir = "bci_filtered\\"+path
        data = np.array(pd.read_csv(file_dir, header=None))

        # data = np.array(pd.read_csv(file_dir))
        ecg_data = data[:,0]

        n_seconds_valid = np.floor(len(ecg_data)/fs)
        print(n_seconds_valid)

        # number_a = n_seconds_valid - (n_seconds_valid % 5)
        # # print(number_a)
        # window_start_indices = np.arange(0,int(number_a)+1,5)
        # window_start_indices = np.append(window_start_indices, n_seconds_valid)
        # print(window_start_indices)

        ecg_data_clipped = ecg_data[0:int(n_seconds_valid)*fs]

        # print(ecg_data_clipped)
        # print(len(ecg_data_clipped))
        # n_windows_check = len(ecg_data_clipped)/wind_n_elements
        # # print(n_windows_check)
        # n_windows =  math.floor(len(ecg_data_clipped)/wind_n_elements)
        # print(n_windows)

        features_array = np.zeros((math.ceil((len(ecg_data_clipped)-wind_n_elements+1)/(5*fs)), 25))
        print(np.shape(features_array))
        
        # flag = not (n_windows == n_windows_check)
        
        dest_file_dir = "features_filtered\\"+path
        counter = 0
        for i in range(0, len(ecg_data_clipped)-wind_n_elements+1, 5*fs):
            # print(i)
            # ecg_sample = ecg_data_clipped[i*wind_n_elements:i*wind_n_elements+wind_n_elements]
            ecg_sample = ecg_data_clipped[i: i+wind_n_elements]
            # print(len(ecg_sample))
            detector = Detectors(fs)
            rr_peaks = np.array(detector.pan_tompkins_detector(ecg_sample))
            # print(rr_peaks)

            # if len(r_peaks1)<30:
            #     pass
            #     print("window size of this is "+str(len(ecg_sample)) + " and i is" + str(i))
            # else:
            # times = np.linspace(0,wind_n_elements/125,len(ecg_sample))/60
            # rr_peaks, _ = find_peaks(-ecg_sample, prominence=0.12, distance=100)
            # plotter_lmao = np.zeros(np.shape(ecg_sample))
            # for i in range(len(rr_peaks)):
            #     plotter_lmao[rr_peaks[i]] = ecg_sample[rr_peaks[i]]
            # x_data = range(1,len(ecg_sample))
            # plt.plot(times, ecg_sample)
            # plt.plot(np.array(times)[rr_peaks], ecg_sample[rr_peaks], "x", label='Blink')
            # # plt.scatter(times, plotter_lmao,marker='o')
            # plt.show()
            # break
            # rr_peaks = np.array(detector.pan_tompkins_detector(ecg_sample))


            rri = (np.diff(rr_peaks) / fs) * 10 ** 3
            filt_rri = np.array(quotient(rri))
            d = np.std(filt_rri)
            b0 = np.median(filt_rri)
            filt_rri[np.where(filt_rri > 1200)] = b0
            filt_rri[np.where(filt_rri < 600)] = b0

            try:
                features = featureExtraction(rri= filt_rri, ECG= ecg_sample, fs= fs)
                features_array[counter,:] += features
                counter += 1
            except ZeroDivisionError:
                counter += 1
            # features = ECG_features(i, wind_n_elements, fs, ecg_data_clipped, detector)
            # print((features))
            # features_array[i,:] = features
        # break
        # if flag:
        #     ''' Run for the remainder seconds if the number of seconds of data recording is not multiple of 10'''
        #     # ecg_sample = ecg_data_clipped[i*wind_n_elements:int(n_seconds_valid)*fs]

        #     detector = Detectors(fs)
        #     # rr_peaks = np.array(detector.pan_tompkins_detector(ecg_sample))
        #     # rri = (np.diff(rr_peaks) / fs) * 10 ** 3
        #     # filt_rri = np.array(quotient(rri))
        #     # d = np.std(filt_rri)
        #     # b0 = np.median(filt_rri)
        #     # filt_rri[np.where(filt_rri > 1200)] = b0
        #     # filt_rri[np.where(filt_rri < 600)] = b0

        #     # features = featureExtraction(rri= filt_rri, ECG= ecg_sample, fs= fs)
        #     # features_array[-1,:] += features
        #     features = ECG_features(i, wind_n_elements, fs, ecg_data, detector)
        #     features_array[-1,:] += features

        # print(np.shape(features_array))
        for j in range(np.shape(features_array)[0]):
            df.loc[len(df)] = features_array[j,:]

        df.to_csv(dest_file_dir[0:-13]+"_ecg_features_filtered.csv")    



