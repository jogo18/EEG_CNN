import scipy.io as sio
import numpy as np
def data_loader(ID, dB, onoff, index):
    path = "C:/Users/jakob/Documents/AAU/7_semester/Projekt/DATA"
    file = f"{path}/{onoff}_{dB}/ID{ID}_preprocesseddata_{dB}dB{onoff}"
    data = sio.loadmat(file)
    temp_trial_data = data['ic_clean']['trial']
    temptimes = data['ic_clean']['time'][0]
    times = np.concatenate(np.concatenate(np.concatenate(np.concatenate(temptimes))))
    conc_trial_data = np.concatenate(np.concatenate(np.concatenate(np.concatenate(temp_trial_data))))
    temptrialinfo = data['ic_clean']['trialinfo']
    conc_trial_info = np.concatenate(np.concatenate(temptrialinfo))
    trial_data = []
    for i in range(len(conc_trial_data[index])):
        trial_data.append(conc_trial_data[index:64,i])
    trial_info = []
    for i in conc_trial_info[index,:]:
        trial_info.append(chr(i))
    trial_info = "".join(trial_info)
    return times, trial_data, trial_info
