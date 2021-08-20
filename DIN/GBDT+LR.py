# -- coding: utf-8 --
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
warnings.filterwarnings('ignore')

# debug模式： 从训练集中划出一部分数据来调试代码
def get_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click



def gbdt_lr_model(train_data,test_data,num_label,category_feature=None, continuous_feature=None):
    # train
    #测试数据放后面
    
    (x_train, y_train), (x_val, y_val) =  train_data
    test_X, wm_order_id=test_data
    # # 离散特征one-hot编码
    # for col in category_feature:
    #     onehot_feats = pd.get_dummies(data[col], prefix=col)
    #     data.drop([col], axis=1, inplace=True)
    #     data = pd.concat([data, onehot_feats], axis=1)

    # train = data[data['Label'] != -1]
    # target = train.pop('Label')
    # test = data[data['Label'] == -1]
    # test.drop(['Label'], axis=1, inplace=True)

    # # 划分数据集
    # x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)
    x_train,x_val=np.concatenate(x_train,dim=0),np.concatenate(x_val,dim=0)
     gbm = lgb.LGBMClassifier(objective='binary',
                             subsample=0.8,
                             min_child_weight=0.5,
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=1000,
                             )

    gbm.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100,
            )

    model = gbm.booster_

    gbdt_feats_train = model.predict(x_train, pred_leaf=True)
    gbdt_feats_val = model.predict(x_val, pred_leaf=True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_val_gbdt_feats = pd.DataFrame(gbdt_feats_val, columns=gbdt_feats_name)

    x_train = pd.concat([x_train, df_train_gbdt_feats], axis=1)
    x_val = pd.concat([x_val, df_val_gbdt_feats], axis=1)
    # train_len = x_train.shape[0]
    data = pd.concat([x_train, x_val])
    target=pd.concat([y_train,y_val])
    del x_train
    del x_val
    gc.collect()

    # # # 连续特征归一化
    # scaler = MinMaxScaler()
    # for col in continuous_feature:
    #     data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    #
    # for col in gbdt_feats_name:
    #     onehot_feats = pd.get_dummies(data[col], prefix=col)
    #     data.drop([col], axis=1, inplace=True)
    #     data = pd.concat([data, onehot_feats], axis=1)
    #
    # train = data[: train_len]
    # test = data[train_len:]
    # del data
    # gc.collect()
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)

    # for test
    fname = '../results/DIN_base_seq.txt'
    outf = open(fname, 'w')
    for dense_input, sparse_input, seq_input, order_id in tqdm(list(zip(*test_X, wm_order_id))):
        data = [
            np.repeat([dense_input], num_label, axis=0),
            np.repeat([sparse_input], num_label, axis=0),
            np.repeat([seq_input], num_label, axis=0),
            np.arange(num_label)[..., np.newaxis]#delete
        ]
        print([d.shape for d in data])
        gbdt_feats_test = model.predict(x_test, pred_leaf=True)
        gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_test.shape[1])]
        df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)
        x_test = pd.concat([x_test, df_test_gbdt_feats], axis=1)
        y_pred =lr.predict_proba(x_test)#注意要取前5的输出
        topK = y_pred
        #result store
        outf.write("{}\t{}\n".format(order_id, "\t".join([str(i) for i in topK])))




if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # file = '../data/原数据-20210301-20210328/'
    file = "../data/SMP新数据-20210607-20210702/"
    
    feature_columns, behavior_list, train, val, test= pickle.load(open('../data/data-DIN.pkl', 'rb'))
    feature_columns, behavior_list, test_X, wm_order_id = pickle.load(open('../data/data-DIN-test.pkl', 'rb'))
    train_data=(train,val,)
    test_data=(test_X, wm_order_id,)
    gbdt_lr_model(train_data,test_data, num_label= 29924)