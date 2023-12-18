import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchsummary import summary
import pickle
import pandas as pd
import numpy as np
import glob
import scipy.io as sio
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm


# CNN Model
class EEGNet(nn.Module):
        def __init__(self):
            super(EEGNet, self).__init__()

            self.F1 = 8
            self.F2 = 8
            self.D = 2
            
            # Conv2d(in,out,kernel,stride,padding,bias)
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, 32), padding='same', bias=False),
                nn.BatchNorm2d(self.F1)
            )
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.F1, self.D*self.F1, (8, 1), groups=self.F1, bias=False),
                nn.BatchNorm2d(self.D*self.F1),
                nn.PReLU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(0.25)
            )
            
            self.Conv3 = nn.Sequential(
                nn.Conv2d(self.D*self.F1, self.D*self.F1, (1, 8), padding='same', groups=self.D*self.F1, bias=False),
                nn.Conv2d(self.D*self.F1, self.F2, (1, 1), bias=False),
                nn.BatchNorm2d(self.F2),
                nn.PReLU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(0.25)
            )
            
            self.classifier = nn.Linear(456, 1, bias=True)
            
        def forward(self, x):
            
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.Conv3(x)
            
            x = torch.flatten(x)
            x = self.classifier(x)
            x = nn.functional.sigmoid(x)
            return x
# Dataset class for processing and 
class DTUEEG(Dataset):
    def __init__(self, path):
        self.dir_path = path
        files = glob.glob(self.dir_path + '\*.PKL')
        self.raw_data = []
        for nrfile in files:
            with open(nrfile, 'rb') as f:
                    self.raw_data.append(pickle.load(f))
        self.data = []
        self.labels = []
        for x in range(len(self.raw_data)):
            self.data.append(self.raw_data[x][0])
            self.labels.append(self.raw_data[x][1])
        self.normalized_data = []
        for row in self.data:
            xmin = np.min(row)
            xmax = np.max(row)
            normalized_row = np.array([2*(x-xmin)/(xmax - xmin) - 1 for x in row])
            self.normalized_data.append(normalized_row)
        self.stacked_data = np.hstack(self.normalized_data)
        self.labels = np.repeat(self.labels, len(self.raw_data[0][0][0])//64)


    def __getitem__(self, idx):
            self.templabel = self.labels[idx]
            if self.templabel == 1:
                self.label = 0
            else:
                self.label = 1
            self.batch = self.stacked_data[:, 64*idx:64*idx+64]
            return self.batch, self.label
        
    def __len__(self):
        return len(self.stacked_data[0])//64
def main():
    #  Define Device, Model, Error Function and Optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGNet().to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)
    # Load Saved Model State
    loadpath = r"C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\ModelDicts\P_8_8_2_Zenodo_K_32_8_8_slightmorebatchsizePRELU.tar"
    checkpoint = torch.load(loadpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    #Testing Loop
    tst_loss_values = []
    tst_acc_values = []
    tst_correct = 0
    tst_acc = 0
    tst_running_loss = 0
    total = 0
    for id in range(18):
        id_path = id+1
        path3 = f'C:/Users/Jakob/Desktop/AAU/7Semester/Projekt/P7_projekt/DATA/DTU/dtupkl/tst/{id_path}'
        testing_data = DTUEEG(path3)
        tst_dataloader = DataLoader(testing_data, batch_size=21*30, shuffle=True)
        ##  Testing
        tst_data, tst_labels = next(iter(tst_dataloader))
        model.eval()
        for i, x_tst in enumerate(tst_data):
            with torch.no_grad():
                x_tst = torch.unsqueeze(x_tst, 0)
                x_tst = torch.unsqueeze(x_tst, 0)
                x_tst = x_tst.float().to(device)
                y_tst = torch.unsqueeze(tst_labels[i], 0)
                y_tst = y_tst.float().to(device)
                y_pred = model(x_tst)
                loss = loss_fn(y_pred, y_tst)
                tst_running_loss += loss.item()
                total += 1
                if torch.round(y_pred) == tst_labels[i]:
                    tst_correct += 1
        
    print(f'total amount of correct preds is for current model is : {tst_correct/total}')
if __name__ == '__main__':
    main()