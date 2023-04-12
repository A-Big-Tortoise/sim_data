#!/usr/bin/env python3

# %%

import scipy
import scipy.signal as signal
import utils
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data():
    data_train = np.load('../simu_20000_0.1_90_140_train.npy')
    data_test = np.load('../simu_10000_0.1_141_178_test.npy')

    data = np.concatenate((data_train, data_test), 0)
    fs = 100  # 采样率
    ecg = data[:, :-6]

    # 设计滤波器
    fc = 1  # 截止频率为1Hz
    order = 5  # 滤波器阶数为2
    b, a = signal.butter(order, fc / (fs / 2), 'highpass')

    filtered = signal.filtfilt(b, a, ecg)
    data[:, :-6] = filtered

    cols = []
    for i in range(1, 11):
        for j in range(1, 101):
            cols.append(f'sec{i}Hz{j}')
    cols.extend(['ID', 'Time', 'H', 'R', 'S', 'D'])

    df = pd.DataFrame(data, columns=cols)

    df = df[df.S > df.D]  # 删除错误数据

    df.drop(['ID', 'Time'], axis=1, inplace=True)

    test_exemple = 10000
    trian_example = df.shape[0] - test_exemple

    return df, trian_example, test_exemple


def sta_features(fea, df):
    # data (N, 10 * 100)
    df_sta = df.copy()
    data = df_sta.values.copy()[:, :-4]
    # print(f'data.shape:{data.shape}')

    data_median = np.median(data, 1, keepdims=True)
    data_mean = np.mean(data, 1, keepdims=True)
    data_var = np.var(data, 1, keepdims=True)
    data_sc = np.mean((data - data_mean) ** 3, 1, keepdims=True)

    df_sta[fea] = np.concatenate(
        (data_median, data_mean, data_var, data_sc), 1)
    return df_sta


def calculate_total_energy(signal):   # 计算信号的总能量
    return np.sum(np.abs(signal)**2)


def calculate_bandwidth(spectrum):
    max_fft_mag = np.max(spectrum[1:])
    half_max = max_fft_mag / 2    # 找到幅度谱中大于等于峰值一半的频率分量
    high_freqs = np.where(spectrum >= half_max)[0]
    bandwidth = high_freqs[-1] - high_freqs[0]    # 计算频带宽度
    return bandwidth / 10, high_freqs[-1], high_freqs[0]


def calculate_ECI(signal):
    _, Pxx = scipy.signal.welch(signal, fs=100)
    # 计算能量集中度指数
    eci = np.sum(Pxx ** 2) / (np.sum(Pxx) ** 2)
    return eci


def calculate_THD(spectrum):
    # 计算信号总功率
    P_total = np.sum(np.abs(spectrum) ** 2) / len(spectrum)
    # 计算前 10 个谐波分量的功率
    P_harmonics = np.abs(spectrum[1:11]) ** 2 / len(spectrum)
    # 计算谐波含量
    thd = np.sqrt(np.sum(P_harmonics)) / np.sqrt(P_total)
    return thd


def extractfeatures_freq(signal):
    spectrum = np.abs(np.fft.fft(signal))
    bandwidth, freq_min, freq_max = calculate_bandwidth(spectrum)

    # 找到频谱中的最大值 (峰值) 及其位置
    max_amp = np.max(spectrum)
    max_freq = np.argmax(spectrum)

    eci = calculate_ECI(signal)
    thd = calculate_THD(spectrum)

    mean = np.mean(spectrum)
    std = np.std(spectrum)
    kurt = np.mean((spectrum - mean) ** 4) / pow(std, 4)

    # 计算频谱变化率特征
    diff_spectrum = np.diff(spectrum)
    change_rate = np.mean(diff_spectrum / spectrum[:-1])
    # 计算频谱带宽特征
    bw = np.sum(spectrum >= np.max(spectrum) / 2)

    return [bandwidth, max_amp, max_freq,  freq_min, eci, thd, mean, std,
            kurt, change_rate, bw]

def PrepareData(df_allfeas, train_feas, train_example, test_example):
    """送入模型验证与模型最后训练的数据"""
    df_model = df_allfeas[train_feas].iloc[1:,:]

    train = df_model.iloc[:-test_example].reset_index(drop=True)
    test = df_model.iloc[-test_example:].reset_index(drop=True)

    """模型验证"""
    X_train = train.drop(['S','D'], 1)
    Y_train_D = train['D']
    Y_train_S = train['S']

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Y_train_D_mean = Y_train_D.mean()
    # Y_train_D_std = Y_train_D.std()
    # Y_train_D = (Y_train_D - Y_train_D_mean) / Y_train_D_std

    """最后模型训练"""
    X_test = test.drop(['S', 'D'], 1)
    Y_test_D = test['D']
    Y_test_S = test['S']

    scaler_test = MinMaxScaler()
    X_test = scaler_test.fit_transform(X_test)
    # Y_test_D = (Y_test_D - Y_train_D_mean) / Y_train_D_std

    # 要求 所有的模型的input都是np
    utils.model_data_info('Val Model', X_train, Y_train_D)
    utils.model_data_info('Train', X_train, Y_train_D)
    utils.model_data_info('Test', X_test, Y_test_D)
    return X_train, Y_train_D, Y_train_S, X_test, Y_test_D, Y_test_S

def freq_features(fea, df):
    data_all = df.values.copy()[:, :-4]

    freq_feas = []
    # print(data_all.shape)
    for row in data_all:
        freq_feas.append(extractfeatures_freq(row))

    df_freq = df_sta.copy()

    df_freq[fea] = np.array(freq_feas)
    return df_freq

def cab_validation(X_train, Y_train):
    cab_mae, fold_ = 0, 0
    sub_cab = 0

    kfolder = KFold(n_splits=2, shuffle=True, random_state=2023)

    for train_index, val_index in kfolder.split(X_train, Y_train):
        fold_ = fold_ + 1
        print("fold n°{}".format(fold_))

        k_x_train, k_y_train = X_train[train_index], Y_train[train_index]
        k_x_vali, k_y_vali = X_train[val_index], Y_train[val_index]

        cb_params = {
            'n_estimators': 10000,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'learning_rate': 0.02,
            'depth': 6,
            'use_best_model': True,
            'subsample': 0.6,
            'bootstrap_type': 'Bernoulli',
            'reg_lambda': 3,
            'one_hot_max_size': 2,
        }

        model_cb = CatBoostRegressor(**cb_params)

        # train the model
        model_cb.fit(
            k_x_train, k_y_train,
            eval_set=[(k_x_vali, k_y_vali)],
            verbose=300,
            early_stopping_rounds=300,
        )

        val_cab = model_cb.predict(k_x_vali)
        sub_cab += val_cab / kfolder.n_splits

        print('val mae: ', mean_absolute_error(val_cab, k_y_vali))
        cab_mae += mean_absolute_error(val_cab, k_y_vali) / kfolder.n_splits
    print('MAE of cab:', cab_mae)
    return cab_mae

def lgb_validation(X_train, Y_train):
    # 模型验证
    kfolder = KFold(n_splits=2, shuffle=True, random_state=2023)

    lgb_mae, fold_ = 0, 0
    sub_lgb = 0

    for train_index, vali_index in kfolder.split(X_train, Y_train):
        fold_ = fold_ + 1
        print("fold n°{}".format(fold_))
        k_x_train, k_y_train = X_train[train_index], Y_train[train_index]
        k_x_vali, k_y_vali = X_train[vali_index], Y_train[vali_index]

        clf = LGBMRegressor(
            n_estimators=20000,
            learning_rate=0.02,
            boosting_type='gbdt',
            objective='regression_l1',
            max_depth=-1,
            num_leaves=31,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_freq=1,
            bagging_fraction=0.8,
            lambda_l2=2,
            random_state=42,
            metric='mae',
        )

        clf.fit(
            k_x_train, k_y_train,
            eval_set=[(k_x_vali, k_y_vali)],
            eval_metric='mae',
            early_stopping_rounds=300,
            verbose=300
        )
        val_lgb = clf.predict(k_x_vali, ntree_end=clf.best_iteration_)
        sub_lgb += val_lgb / kfolder.n_splits
        print('val mae: ', mean_absolute_error(val_lgb, k_y_vali))
        lgb_mae += mean_absolute_error(val_lgb, k_y_vali) / kfolder.n_splits
    print('MAE of lgb:', lgb_mae)
    return lgb_mae

def train(X_train, Y_train, X_test, Y_test):
    clf = LGBMRegressor(
        n_estimators=30000,
        learning_rate=0.02,
        boosting_type='gbdt',
        objective='regression_l1',
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_freq=1,
        bagging_fraction=0.8,
        lambda_l2=2,
        random_state=42,
        metric='mae',
    )

    clf.fit(
        X_train, Y_train,
        eval_set=[(X_test, Y_test)],
        eval_metric='mae',
        early_stopping_rounds=200,
        verbose=300
    )

    cb_params = {
        'n_estimators': 10000,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'learning_rate': 0.02,
        'depth': 6,
        'use_best_model': True,
        'subsample': 0.6,
        'bootstrap_type': 'Bernoulli',
        'reg_lambda': 3,
        'one_hot_max_size': 2,
    }

    model_cb = CatBoostRegressor(**cb_params)

    # train the model
    model_cb.fit(
        X_train, Y_train,
        eval_set=[(X_test, Y_test)],
        verbose=300,
        early_stopping_rounds=300,
    )

    joblib.dump(model_cb, "model_catb.m")
    joblib.dump(clf, "model_lgb.m")

if __name__ == '__main__':

    df, train_example, test_example = load_data()

    sta_feas = ['Median', 'Mean', 'Var',  'Sc']
    freq_feas = ['bandwidth', 'max_amp', 'max_freq', 'freq_min',
                 'eci', 'thd', 'mean', 'std', 'kurt', 'change_rate', 'bw']

    df_sta = sta_features(sta_feas, df)

    df_sta_freq = freq_features(freq_feas, df)
    print(df_sta_freq.shape)

    # 选定训练所需要的标签
    df_allfeas = df_sta_freq

    train_feas = list(df_allfeas.columns[1000:])

    X_train, Y_train_D, Y_train_S, X_test, Y_test_D, Y_test_S = PrepareData(df_allfeas, train_feas, train_example, test_example)

    np.save('X_test.npy', X_test)
    np.save('Y_test_D.npy', Y_test_D)
    np.save('Y_test_S.npy', Y_test_S)

    cab_mae = cab_validation(X_train, Y_train_D)
    lgb_mae = lgb_validation(X_train, Y_train_D)
    train(X_train, Y_train_D, X_test, Y_test_D)

    np.save('Weight.npy', np.array([lgb_mae, cab_mae]))
