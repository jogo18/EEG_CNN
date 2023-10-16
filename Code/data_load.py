import scipy.io as sio
import numpy as np
def data_loader(ID, dB, onoff, index):
    path = "C:/Users/jakob/Documents/AAU/7_semester/Projekt/DATA"
    file = f"{path}/{onoff}_{dB}/ID{ID}_preprocesseddata_{dB}dB{onoff}"
    data = sio.loadmat(file)
    #Load trial data and fix it for Python
    temp_trial_data = data['ic_clean']['trial']
    conc_trial_data = np.concatenate(np.concatenate(np.concatenate(np.concatenate(temp_trial_data))))
    trial_data = []
    for i in range(len(conc_trial_data[index])):
        trial_data.append(conc_trial_data[index:64,i])
    #Load time data and fix it for Python
    temptimes = data['ic_clean']['time'][0]
    times = np.concatenate(np.concatenate(np.concatenate(np.concatenate(temptimes))))
    #Load trial info and fix it for Python
    temptrialinfo = data['ic_clean']['trialinfo']
    conc_trial_info = np.concatenate(np.concatenate(temptrialinfo))
    trial_info = []
    for i in conc_trial_info[index,:]:
        trial_info.append(chr(i))
    trial_info = "".join(trial_info)
    #Load unmixing matrix and fix it for Python
    temp_unmixing = data['ic_clean']['icaunmixingmatrix']
    unmixing_matrix = np.concatenate(np.concatenate(temp_unmixing))
    return times, trial_data, trial_info, unmixing_matrix
