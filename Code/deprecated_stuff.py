class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 256), 1, padding='same'),
            nn.BatchNorm2d(64),
            nn.PReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 64, (64, 1), 1, padding='valid'),
                nn.PReLU(),
                nn.AvgPool2d(1,4),
                nn.Dropout(0.5)
            )
            self.conv4 = nn.Sequential(
                # nn.BatchNorm1d(64),
                nn.Conv2d(64, 64, (1, 32), 1, padding='same'),
                nn.Conv2d(64, 64, (1, 1), 1, padding='same'),
                nn.AvgPool2d(1,8),
                nn.PReLU()
            )
            self.fc1 = nn.Sequential(
                nn.Linear(1024, 2),
                nn.PReLU(),
            )
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = torch.flatten(x)
            x = self.fc1(x)
            # x = F.softmax(x)
            return x
        
    # valpath = r'C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\OFF_8\ID0002_preprocesseddata_8dBOFF.mat'
    # correct = 0
    # total = 0
    # validation_data = SplitTrainingDataSet(valpath)
    # # for i in range(0,20):
    # val_dataloader = DataLoader(validation_data, batch_size = 220, shuffle = False)
    # iteration = iter(val_dataloader)
    # data, labels = next(iteration)
    # model.eval()
    # with torch.no_grad():
    #     for i, x in enumerate(data):
    # #         print(x.shape)
    # #         # print(y.shape)
    #         x = torch.unsqueeze(x, 0)
    #         x = torch.unsqueeze(x, 0)
    #         test = x.float().to(device)
    #         y = labels[i].float().to(device)
    #         y_pred = model(test)
    #         pred = torch.argmax(y_pred)
        
    #         if pred == torch.argmax(y):
    #             correct +=1
    #         print(f'PREDICTION is {pred}, LABEL IS {torch.argmax(y)}')
    #         total += 1

    # print(f"Testing is {correct/total*100}% Accurate, with {total} total validation samples")
        ####Validation Loop
    # val_path = r'C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\Validation\*.mat'
    # path = r'C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\ON_8\ID0001_preprocesseddata_3dBON.mat'
    # # files = glob.glob(val_path)
    # correct = 0
    # total = 0
    # # for path in files:
    # for j in range(16, 20):
    #     validation_data = SplitValData(path, j)
    #     # for i in range(0,20):
    #     val_dataloader = DataLoader(validation_data, batch_size = 33, shuffle = False)
    #     iteration = iter(val_dataloader)
    #     data, labels = next(iteration)
    #     model.eval()     # Optional when not using Model Specific layer
    #     with torch.no_grad():
    #         for i, x in enumerate(data):
    #             x = torch.unsqueeze(x, 0)
    #             x = torch.unsqueeze(x, 0)
    #             test = x.float().to(device)
    #             y = labels[i].float().to(device)
    #             y_pred = model(test)
    #             pred = torch.argmax(y_pred)
            
    #             if pred == torch.argmax(labels):
    #                 correct +=1
    #             print(f'PREDICTION is {pred}, LABEL IS {torch.argmax(labels)}')
    #             total += 1

      
                    # loss = loss_fn(target,labels)
                    # valid_loss = loss.item() * data.size(0)
                    # print(len(val_dataloader))

    # print(f"Testing is {correct/total*100}% Accurate, with {total} total validation samples")
    
#     # #     # # print(f'Epoch {i+1} \t\t  \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
    #### Training Loop

    # running_loss = 0
    # acc = 0
    # EPOCHS = 100
    # for i in range(0, EPOCHS):
    #     for path in paths:
    #         files = glob.glob(path)
    #         for file in files:
    #             if file == r'C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\ON_8\*.mat':
    #                 maxfileidx = 19
    #             else:
    #                 maxfileidx = 20
    #             for e in range(0, maxfileidx):    
    #                     model.train()
    #                     training_data = SplitTrainingDataSet(file, e)
    #                     train_dataloader = DataLoader(training_data, batch_size=33, shuffle=True)
    #                     iteration = iter(train_dataloader)
    #                     data, labels = next(iteration)
    #                     # # print(data.shape)
    #                     # with tqdm.tqdm(train_dataloader, unit = 'batch') as tepoch:
    #                     for i, x in enumerate(data):
    #                         # tepoch.set_description(f"Epoch {i}")
    #                         x = torch.unsqueeze(x, 0)
    #                         test = x.float().to(device)
    #                         y = labels[i].float().to(device)
    #                         y_pred = model(test)

    #                         loss = loss_fn(y_pred, y)
    #                         # writer.add_scalar("Loss/train", loss, e)
    #                         optimizer.zero_grad()
    #                         loss.backward()
    #                         optimizer.step()
    #                         running_loss += loss.item()
    #                         # acc += (torch.argmax(y_pred) == torch.argmax(y)).float().sum()
    #                         # acc = acc/33
    #                             # tepoch.set_postfix(loss=loss.item(), accuracy= acc)
    #                         print(f"\nLABEL: {y} PRED: {y_pred}:") 
    #     #             # writer.flush()


    ######################TESTING GROUND####################################
    # data_path = r"C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\ON_8\ID0002_preprocesseddata_8dBON"
    # data = sio.loadmat(data_path)
    # labels = []
    # for i in range(len(data['ic_clean']['trialinfo'][0][0])):
    #     label = data['ic_clean']['trialinfo'][0][0][i]
    #     temp = []
    #     for i in label[18:]:
    #         temp.append(chr(i))
    #     temp = "".join(temp)
    #     if temp.find('Male') < temp.find('Female'):
    #         labels.append(1)
    #     else:
    #         labels.append(0)
    # labels = np.repeat(labels, 11)

    # print(labels)
    # dataset = []
    # for j in data['ic_clean']['trial'][0][0][0]:
    #     dataset.append(j[:-2, 4096:])
    # test = np.hstack(dataset)


    # # asd = test[0:64, 0:1536]
    # # asd2 = test[65*1, 1536*1+1536]

    # print(len(test[0]))
    # conv1 = nn.Sequential(
    #         nn.Conv2d(1, 64, (1, 256), 1, padding = 'same'),
    #         nn.BatchNorm2d(64),
    #         nn.PReLU()
    #         )
    # conv2 = nn.Sequential(
    #     nn.Conv2d(64, 64, (64, 1), 1, padding='valid'),
    #     nn.AvgPool2d(1,4),
    #     nn.ELU()
    # )
    # conv3 = nn.Sequential(
    #     nn.Conv2d(64, 64, (1, 16), 1, padding = 'same'),
    #     # nn.MaxPool2d(4,4),
    #     nn.Conv2d(64, 64, (1, 1), 1, padding='same'),
    #     nn.AvgPool2d(1,8),
    #     nn.ELU()
    # )
    # fc1 = nn.Sequential(
    #     nn.Linear(20928 , 5232),
    #     nn.ReLU()
    # )
                # Conv2d(in,out,kernel,stride,padding,bias)
    # D = 2
    # F1 = 16
    # F2 = 8
    # conv1 = nn.Sequential(
    #     nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
    #     nn.BatchNorm2d(F1)
    # )
    
    # conv2 = nn.Sequential(
    #     nn.Conv2d(F1, D*F1, (22, 1), groups=F1, bias=False),
    #     nn.BatchNorm2d(D*F1),
    #     nn.ELU(),
    #     nn.AvgPool2d((1, 4)),
    #     nn.Dropout(0.5)
    # )
    
    # conv3 = nn.Sequential(
    #     nn.Conv2d(D*F1, D*F1, (1, 16), padding=(0, 8), groups=D*F1, bias=False),
    #     nn.Conv2d(D*F1, F2, (1, 1), bias=False),
    #     nn.BatchNorm2d(F2),
    #     nn.ELU(),
    #     nn.AvgPool2d((1, 8)),
    #     nn.Dropout(0.5)
    # )

    # path = r"C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\ON_3\ID0001_preprocesseddata_3dBON.mat"
    # training_data = SplitTrainingDataSet(path, 0)

    # train_dataloader = DataLoader(training_data, batch_size=2, shuffle=False)

    # testasd, labels = next(iter(train_dataloader))
    # print(testasd.shape)

    # asd = torch.unsqueeze(testasd, 1)
    # print(asd.shape)
    # test1 = conv1(asd.float())
    # print(test1.shape)
    # test2 = conv2(test1) 
    # print(test2.shape)
    # test3 = conv3(test2)
    # print(test3.shape)
    # test4 = torch.flatten(test3)
    # print(test4.shape)
    # # test3 = conv3(test2)
    # # test4 = torch.flatten(test3)
    # print(test2.shape)
    # # print(test3.shape)
    # # print(asd.shape)
    # print(test4.shape)
    # print(asd[0, :, 0, 0])


    # training_data = SplitTrainingDataSet(path, 0)
    # train_dataloader = DataLoader(training_data, batch_size=10, shuffle=False)
    # test_dataloader = DataLoader(validation_data, batch_size = 1, shuffle = False)
##############################################################################################################################

    # valpath = r'C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\OFF_8\ID0002_preprocesseddata_8dBOFF.mat'
    # correct = 0
    # total = 0
    # validation_data = SplitTrainingDataSet(valpath)
    # # for i in range(0,20):
    # val_dataloader = DataLoader(validation_data, batch_size = 220, shuffle = False)
    # iteration = iter(val_dataloader)
    # data, labels = next(iteration)
    # model.eval()
    # with torch.no_grad():
    #     for i, x in enumerate(data):
    # #         print(x.shape)
    # #         # print(y.shape)
    #         x = torch.unsqueeze(x, 0)
    #         x = torch.unsqueeze(x, 0)
    #         test = x.float().to(device)
    #         y = labels[i].float().to(device)
    #         y_pred = model(test)
    #         pred = torch.argmax(y_pred)
        
    #         if pred == torch.argmax(y):
    #             correct +=1
    #         print(f'PREDICTION is {pred}, LABEL IS {torch.argmax(y)}')
    #         total += 1

    # print(f"Testing is {correct/total*100}% Accurate, with {total} total validation samples")

    ####Validation Loop
    # val_path = r'C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\Validation\*.mat'
    # path = r'C:\Users\Jakob\Desktop\AAU\7Semester\Projekt\P7_projekt\DATA\ON_8\ID0001_preprocesseddata_3dBON.mat'
    # # files = glob.glob(val_path)
    # correct = 0
    # total = 0
    # # for path in files:
    # for j in range(16, 20):
    #     validation_data = SplitValData(path, j)
    #     # for i in range(0,20):
    #     val_dataloader = DataLoader(validation_data, batch_size = 33, shuffle = False)
    #     iteration = iter(val_dataloader)
    #     data, labels = next(iteration)
    #     model.eval()     # Optional when not using Model Specific layer
    #     with torch.no_grad():
    #         for i, x in enumerate(data):
    #             x = torch.unsqueeze(x, 0)
    #             x = torch.unsqueeze(x, 0)
    #             test = x.float().to(device)
    #             y = labels[i].float().to(device)
    #             y_pred = model(test)
    #             pred = torch.argmax(y_pred)
            
    #             if pred == torch.argmax(labels):
    #                 correct +=1
    #             print(f'PREDICTION is {pred}, LABEL IS {torch.argmax(labels)}')
    #             total += 1

                    # loss = loss_fn(target,labels)
                    # valid_loss = loss.item() * data.size(0)
                    # print(len(val_dataloader))

    # print(f"Testing is {correct/total*100}% Accurate, with {total} total validation samples")
    
#     # #     # # print(f'Epoch {i+1} \t\t  \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
