import scipy.io as sio
import numpy as np

# ID = '0001'
# dB = '3'
# onoff = 'ON'

def data_loader(id, dB, onoff):
    path = "C:/Users/jakob/Documents/AAU/7_semester/Projekt/DATA"
    file = f"{path}/{onoff}_{dB}/ID{id}_preprocesseddata_{dB}dB{onoff}"
    data = sio.loadmat(file)
    temp_trial_data = data['ic_clean']['trial']
    temp_labels = data['ic_clean']['label']
    labels = np.concatenate(np.concatenate(np.concatenate(np.concatenate(temp_labels))))
    temptimes = data['ic_clean']['time'][0]
    times = np.concatenate(np.concatenate(np.concatenate(np.concatenate(temptimes))))
    trial_data = np.concatenate(np.concatenate(np.concatenate(np.concatenate(temp_trial_data))))
    return labels, times, trial_data
    

# asd = data_loader(ID, dB, onoff)

# print(asd)