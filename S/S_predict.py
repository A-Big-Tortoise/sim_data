#!/usr/bin/env python3

# %%

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import numpy as np
import torch


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

def plot_2vectors(label, pred, name):
    def calc_mae(gt, pred):
        return np.mean(abs(np.array(gt) - np.array(pred)))

    list1 = label
    list2 = np.array(pred)
    if len(list2.shape) == 2:
        mae = calc_mae(list1, list2[:, 0])
    else:
        mae = calc_mae(list1, list2)

        sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])

    plt.clf()
    plt.text(0, np.min(list2), f'MAE={mae}')

    # plt.plot(range(num_rows), list2, label=name + ' prediction')
    plt.scatter(np.arange(list2.shape[0]), list2[sorted_id], s=1, alpha=0.5, label=f'{name} prediction', color='blue')
    plt.scatter(np.arange(list1.shape[0]), list1[sorted_id], s=1, alpha=0.5, label=f'{name} label', color='red')

    plt.legend()
    plt.savefig(f'{name}.png')
    print(f'Saved plot to {name}.png')
    plt.show()

def test():
    mae = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss_test = loss_fn(outputs, labels)
        mae.append(loss_test.item())
    print(f'mae:{np.mean(mae)}')

if __name__ == '__main__':

    model = torch.load('model_nn.pth', map_location=torch.device('cpu'))
    test_loader = torch.load('./S_test_loader.pth', map_location=torch.device('cpu'))
    loss_fn = torch.nn.L1Loss()
    print(model)

    X_test = np.load('./X_test.npy')
    Y_test_S = np.load('./Y_test_S.npy')
    preds = model(torch.Tensor(X_test))
    # print(Y_test_S.shape)
    # print(preds.detach().numpy().shape
    #       )
    plot_2vectors(Y_test_S, preds.squeeze(-1).detach().numpy(), 'S')
