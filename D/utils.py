import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


def heatmap(data_df):
    corr_data = abs(data_df.corr())
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_data, linewidths=0.1, cmap=sns.cm.rocket_r)


def model_data_info(datatype, X, Y):
    print(f'{datatype} info:')
    print(f'X.shape:{X.shape}, Y.shape:{Y.shape}')


def MemoryReduce(data):
    start_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage: %.2f MB' % start_mem)

    for col in data.columns:
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
        else:
            data[col] = data[col].astype('category')

    end_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage after optimization: %.2f MB' % end_mem)
    print('Decreased by %.1f%%' % (100 * (start_mem - end_mem) / start_mem))
    return data


def draw_result(labels, predict):

    df_pred = pd.DataFrame({'labels': labels, 'pred': predict})

    df_pred_down = df_pred.sort_values(by='labels')

    lens = df_pred.shape[0]
    plt.figure(figsize=(10, 10))
    plt.scatter(np.arange(lens), df_pred_down.labels, s=3)
    plt.scatter(np.arange(lens), df_pred_down.pred, s=3)

    mean_absolute_error(predict, labels)
