import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda")

data = pd.read_csv('data_5drivers/2016_07_01.csv')

#generate trajectories from CSV file 
def aggr(data):
    traj_raw = data.values[:,1:]
    traj = np.array(sorted(traj_raw,key = lambda d:d[2]))
    label = data.iloc[0][0]
    return [traj,label]
processed_data = data.groupby('plate').apply(aggr)

#generate some dummy features 
training = []
labels = []
for traj in processed_data:
    feature = [len(traj[0]),sum(traj[0][:,-1])]
    label = traj[1]
    training.append(feature)
    labels.append(label)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

#Create Dataset for pytorch training 
trainset = CustomTensorDataset(tensors=(torch.tensor(training,dtype=torch.float32), torch.tensor(labels,dtype=torch.long)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True)

#define your model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#train your model

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (inp,lab) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inp.to(device), lab.to(device)
        print(inputs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))


#save model
torch.save(net.state_dict(),"testmodel.pth")