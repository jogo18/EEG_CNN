import matplotlib as plt
import torch
import torch.nn as nn
# import torcheeg.transforms as transformspip l
from torch.utils.data import DataLoader
# from torcheeg.datasets import BaseDataset
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import csv
path = r'C:/Users/Jakob/Desktop/AAU/7Semester/Projekt/Python/0001_3db_ON_trial2.csv'
path2 = 'C:/Users/jakob/Documents/AAU/7_semester/Projekt/DATA/ON_3/ID0001_3_dBON/trial2'

    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
        nn.Conv1d(64, 64, 5, 2),
        nn.MaxPool1d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 5, 2),
            nn.MaxPool1d(2,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 5, 2),
            nn.MaxPool1d(2,2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(20928 , 5232),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(5232, 1308),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1308, 2),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



class TestDataSet(Dataset):
    def __init__(self):
        self.df = pd.read_csv(path)
        self.df = self.df.drop(self.df.index[[-1]])
        self.df = self.df.drop(self.df.index[[-1]])
        self.dataset = torch.tensor(self.df.to_numpy()).float()
        self.dataset = torch.unsqueeze(self.dataset, 0)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def main():
    #LABELS DEFINED AS FOLLOWS: IF TARGET IS MaleT1, LABEL IS SET TO 0, IF TARGET IS SET TO FemaleT1 LABEL IS SET TO 1 
    # labels = np.concatenate((np.zeros(10),np.ones(10)))
    labels = torch.zeros(2)
    test = TestDataSet()
    test_dataload = DataLoader(test, batch_size=1, shuffle=False)
    def train_one_epoch():
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(test_dataload):
            # Every data instance is an input + label pair
            inputs = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            # if i % 1000 == 999:
            #     last_loss = running_loss / 1000 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(training_loader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.
        return running_loss/len(test_dataload)

    conv1 = nn.Sequential(
        nn.Conv1d(64, 64, 5, 2),
        nn.MaxPool1d(2, 2)
        )
    conv2 = nn.Sequential(
    nn.Conv1d(64, 64, 5, 2),
    nn.MaxPool1d(2,2)
    )
    conv3 = nn.Sequential(
    nn.Conv1d(64, 64, 5, 2),
    nn.MaxPool1d(2,2)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    batch_size = 1
    EPOCHS = 2
    epoch_number = 0
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        model.train(True)
        avg_loss = train_one_epoch()
        running_vloss = 0.0
        # model.eval()
        epoch_number += 1
    
    print(f'LOSS train {avg_loss}')
        

# #Define training and testing process

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CNN().to(device)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
# batch_size = 64


# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch_idx, batch in enumerate(dataloader):
#         X = batch[0].to(device)
#         y = batch[1].to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch_idx % 100 == 0:
#             loss, current = loss.item(), batch_idx * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# asd = train(test_dataload, model, loss_fn, optimizer)

# print(asd)
# def valid(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     val_loss, correct = 0, 0
#     with torch.no_grad():
#         for batch in dataloader:
#             X = batch[0].to(device)
#             y = batch[1].to(device)

#             pred = model(X)
#             val_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     val_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
















if __name__ == '__main__':
    main() 