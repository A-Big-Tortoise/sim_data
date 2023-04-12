#!/usr/bin/env python3

# %%


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_weight():
    path = './Weight.npy'
    # np.save('Weight.npy', np.array([lgb_mae, cab_mae]))
    weight = np.load(path)
    return weight[0], weight[1]

def model_predict(path, X):
    model = joblib.load(path)
    return model.predict(X)

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


if __name__ == '__main__':
    lgb_mae, cab_mae = load_weight()
    # cab_mae = 7.21293717222982
    # lgb_mae =7.112225244477853

    X_test = np.load('./X_test.npy')
    Y_test_D = np.load('./Y_test_D.npy')

    predictions_lgb = model_predict('./model_lgb.m', X_test)
    predictions_cb = model_predict('./model_catb.m', X_test)
    test_Weighted = (1 - lgb_mae / (lgb_mae + cab_mae)) * predictions_lgb + (
                1 - cab_mae / (lgb_mae + cab_mae)) * predictions_cb

    plot_2vectors(Y_test_D, test_Weighted, 'D')
