#!/usr/bin/env python3

# %%

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim
import random
import utils


def load_data():
    data_train = np.load('../simu_20000_0.1_90_140_train.npy')
    data_test = np.load('../simu_10000_0.1_141_178_test.npy')

    test_example = 10000

    data = np.concatenate((data_train, data_test), 0)

    cols = []
    for i in range(1, 11):
        for j in range(1, 101):
            cols.append(f'sec{i}Hz{j}')
    cols.extend(['ID', 'Time', 'H', 'R', 'S', 'D'])

    df = pd.DataFrame(data, columns=cols)
    df = df[df.S > df.D]
    train_example = df.shape[0] - test_example

    df_allfeas = df

    train_feas = list(df_allfeas.columns[:1000])
    train_feas.extend(['S', 'D'])
    len(train_feas)

    """送入模型验证与模型最后训练的数据"""
    df_model = df_allfeas[train_feas]

    train = df_model.iloc[:train_example].reset_index(drop=True)
    test = df_model.iloc[-test_example:].reset_index(drop=True)

    """模型验证"""
    X_train = train.drop(['S', 'D'], 1)
    Y_train_S = np.array(train['S'])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    """最后模型训练"""
    X_test = test.drop(['S', 'D'], 1)
    Y_test_S = np.array(test['S'])

    scaler_test = StandardScaler()
    X_test = scaler_test.fit_transform(X_test)

    # 要求 所有的模型的input都是np
    utils.model_data_info('Val Model', X_train, Y_train_S)
    utils.model_data_info('Train', X_train, Y_train_S)
    utils.model_data_info('Test', X_test, Y_test_S)

    return X_train, Y_train_S, X_test, Y_test_S

def data_loader(train_x, train_y, test_x, test_y, batch_size):
    # data_train , data_test (N, fea_len)
    ratio = 0.8
    train_len = int(train_x.shape[0] * ratio)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x[:train_len]), torch.FloatTensor(train_y[:train_len]))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x[:train_len]), torch.FloatTensor(train_y[:train_len]))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y))

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=8,
                            prefetch_factor=4,pin_memory=True,drop_last=False)    

    val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=True,num_workers=8,
                            prefetch_factor=4,pin_memory=True,drop_last=False)

    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=8,
                           prefetch_factor=4,pin_memory=True,drop_last=False)
    
    return train_loader, val_loader, test_loader

class CNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        # inputs(b, 1, 1000)
        self.conv1 = torch.nn.Conv1d(1, 4, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(4, 2, kernel_size=20, stride=2)
        self.conv3 = torch.nn.Conv1d(2, 2, kernel_size=20, stride=2)
        self.conv4 = torch.nn.Conv1d(2, 1, kernel_size=20, stride=2)
        self.fc1 = torch.nn.Linear(109, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        self.fc3 = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, inputs):  
        inputs = inputs.unsqueeze(1)
        # print(inputs.shape)      
        inputs = self.conv1(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv2(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv3(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv4(inputs)
        # print(inputs.shape)

        inputs = self.relu(self.fc1(inputs))
        inputs = self.relu(self.fc2(inputs))
        inputs = self.fc3(inputs)

        return inputs.squeeze(-1)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def train():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        loss_train = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

def val():
    mae = []
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss_test = loss_fn(outputs, labels)
        mae.append(loss_test.item())
    print(f'val epoch: {epoch}, mae:{np.mean(mae)}')
    return mae

def test():
    mae = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss_test = loss_fn(outputs, labels)
        mae.append(loss_test.item())
    print(f'mae:{np.mean(mae)}')

if __name__ == '__main__':
    setup_seed(42)
    batch_size = 32
    epoches = 50
    lr = 0.0001
    weight_decay = 5e-4
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_x, train_y, test_x, test_y = load_data()

    np.save('X_test.npy', test_x)
    np.save('Y_test_S.npy', test_y)

    train_loader, val_loader, test_loader = data_loader(train_x, train_y, test_x, test_y, batch_size)
    torch.save(test_loader, "S_test_loader.pth")
    model = CNN(1000, 1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_fn = torch.nn.L1Loss()
    print(model)

    min_mae = 10000
    for epoch in range(1, epoches+1):
        train()
        val_mae = val()
        if val_mae < min_mae:
            torch.save(model, 'model_nn.pth')
