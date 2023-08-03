import numpy as np
from os import listdir
from os.path import join, splitext
import wfdb
import scipy.signal as sg
import pickle
from tqdm import tqdm
import torch

def read_signals_annos(path, file_set):
    
    files = {splitext(item)[0] for item in listdir(path)}
    files = list(files)
    files.sort()

    signals = []
    annos = []
    
    conversion = {'N':0, 
                'L':1, 
                'R':2,
                'e':3, 
                'j':4, 
                'A':5, 
                'a':6, 
                'J':7, 
                'S':8, 
                'V':9, 
                'E':10,
                'F':11,
                'Q':12,
                }
    
    record_id = 0

    # butterworth filter
    b, a = sg.butter(5, [0.5, 75], 'bandpass', fs=360) 
    
    for item in tqdm(files):
        if len(item) > 3:  # skip non-records
            continue
        if file_set is not None and item not in file_set:
            continue

        # read signals
        signal, field = wfdb.rdsamp(join(path, item))
        # read annos
        anno = wfdb.rdann(join(path, item),'atr')
        annot = {}

        # remove symbols and samples that are not in conversion
        beat_symbols = list(conversion.keys())
        anno.symbol = np.array(anno.symbol)
        anno.sample = np.array(anno.sample)
        selected_symbols = anno.symbol[np.where(np.isin(anno.symbol, beat_symbols))[0]]
        selected_samples = anno.sample[np.where(np.isin(anno.symbol, beat_symbols))[0]]
        # convert symbols
        selected_symbols = [conversion[symbol] for symbol in selected_symbols]

        if item == '114':
            signal1 = signal[:, 1]    # Note: 114 channels are flipped
        else:
            signal1 = signal[:, 0]
        
        # noise and baseline wandering removal - Butterworth filter
        signal1 = sg.filtfilt(b, a, signal[:, 0])

        # peak alignment
        search = 20
        for i, s in enumerate(selected_samples):  
            search_range = signal1[s-search:s+search]
            if signal1[s] > 0 :
                s = s - search + np.argmax(search_range) 
            else:
                s = s - search + np.argmin(search_range)
            selected_samples[i] = s

        # peak magnitude standardization
        positive = signal1[selected_samples] > 0
        signal1 = signal1 / np.mean(signal1[selected_samples[positive]])

        signals.append(torch.from_numpy(signal1).to(torch.float32))
        
        annot['labels'] = torch.tensor(selected_symbols).to(torch.long)
        annot['samples'] = torch.from_numpy(selected_samples).to(torch.long)
        annot['record_name'] = torch.tensor(int(anno.record_name))
        annot['record_id'] = torch.tensor(int(record_id))
        record_id += 1

        annos.append(annot)

    # zip signals and annos
    output = [[signals[i], annos[i]] for i in range(len(signals))]
        
    return output

DS1 = {'101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230'}
ds1_data = read_signals_annos('data', file_set=DS1)

with open("data\ds1.pkl", "wb") as f:
    pickle.dump(ds1_data, f)

DS2 = {'100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','222','228','231','232','233','234'}
ds2_data = read_signals_annos('data', file_set=DS2)

with open("data\ds2.pkl", "wb") as f:
    pickle.dump(ds2_data, f)