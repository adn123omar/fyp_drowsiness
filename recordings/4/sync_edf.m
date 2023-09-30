clear all; close all; clc;

%% edf time

tt = edfread("20-00-51.edf");
tt = tt(1:600,:);

edf_hr = 20;
edf_min = 0;
edf_sec = 51;

time_edf = edf_hr*60*60+edf_min*60+edf_sec+7*60+46;

%% bci time
bci_data = importdata("bci1.txt");

bci_data = bci_data(7:end);

time_vec_bci = zeros(length(bci_data),1);

for i = 1:length(bci_data)
    lol = bci_data{i};
    counter = 0;
    final_num = "";
    hr = "";
    min = "";
    sec = "";
    id = length(lol);

    while counter<1
        if lol(id)== ','
            counter = counter + 1;
        end
        if counter==1
            id = id+11;
            hr = strcat(hr,lol(id+2));
            hr = strcat(hr,lol(id+3));
            min = strcat(min,lol(id+5));
            min = strcat(min,lol(id+6));
            sec = strcat(sec,lol(id+8:id+13));
            hr = str2double(hr);
            min = str2double(min);
            sec = str2double(sec);
        end
        id = id - 1;
    end
    time_bci = hr*60*60+min*60+sec;

    time_vec_bci(i) = time_bci;
end

%% acc time

acc_data = importdata("acc.txt");
time_vec_acc = zeros(length(acc_data),1);

for j = 1:length(acc_data)
    acc1 = acc_data{j};
    acc_hr = str2double(acc1(12:13));
    acc_min = str2double(acc1(15:16));
    acc_sec = str2double(acc1(18:19));
    time_acc = acc_hr*60*60+acc_min*60+acc_sec;
    time_vec_acc(j) = time_acc;
end

%% cropping data

bci_data = bci_data(time_vec_bci>=time_edf);
bci_data = bci_data(1:600*250);
acc_data = acc_data(time_vec_acc>=time_edf);
acc_data_val = time_vec_acc(1)+601;
acc_data = acc_data(time_vec_acc<acc_data_val);

writetimetable(tt,"edf1_clipped.csv");
writecell(acc_data,"acc_clipped.csv");
writecell(bci_data,"bci_clipped.csv");















