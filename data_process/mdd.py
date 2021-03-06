import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# npois = 29071
# nspus = 179778

npois = 29924
nusers = 200000
# nspus = 195244

sparse_feature_count = {
    # 'wm_poi_id': npois,
    'user_id': nusers,
    'aor_id': 11,
    'ord_period_name': 5,
    'aoi_id': 3700,     # 3625，有不包含在train中的地址
    'takedlvr_aoi_type_name': 13
}

def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    return {'feat': feat}


def create_amazon_electronic_dataset(file, embed_dim=8, maxlen=40):
    print('==========Data Preprocess Start============')
    with open('raw_data/remap.pkl', 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i-1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:
                test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # test_data.append([hist_i, [pos_list[i]], 1])
                # test_data.append([hist_i, [neg_list[i]], 0])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # val_data.append([hist_i, [pos_list[i]], 1])
                # val_data.append([hist_i, [neg_list[i]], 0])
            else:
                train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                # train_data.append([hist_i, [pos_list[i]], 1])
                # train_data.append([hist_i, [neg_list[i]], 0])

    # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    behavior_list = ['item_id']  # , 'cate_id'

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
             pad_sequences(val['hist'], maxlen=maxlen),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)


def create_mdd_dataset(datapath, embed_dim=8, maxlen=40):
    print('==========Data Preprocess Start============')
    d_session = pd.read_csv(datapath+"orders_poi_session.txt", sep='\t')
    d_train = pd.read_csv(datapath+"orders_train.txt", encoding="utf-8", sep='\t')
    # 暂不考虑 SPU
    # d_train_spu = pd.read_csv(datapath+"orders_spu_train.txt", encoding="utf-8", sep='\t')
    # d_train_spu_agg = d_train_spu.groupby('wm_order_id')["wm_food_spu_id"].apply(list).reset_index(name='spu_ids')
    """
    # user_id	wm_order_id	wm_poi_id	aor_id	order_price_interval	order_timestamp	ord_period_name	order_scene_name	aoi_id	takedlvr_aoi_type_name	dt
    # wm_order_id	clicks	dt
    # wm_order_id	wm_food_spu_id	dt
    """
    data = pd.merge(d_train, d_session, on="wm_order_id")
    # data = pd.merge(data, d_train_spu_agg, on="wm_order_id")

    # nusers = data['user_id'].nunique()
    # npois = data['wm_poi_id'].nunique()
    data['clicks'] = data['clicks'].apply(lambda x: [int(i) for i in x.split('#')] if not pd.isnull(x) else [])
    # spus = set()
    # for l in data['clicks'].values:
    #     spus |= set(l)
    # nspus = len(spus)

    # ============ feature columns
    # user_id	wm_order_id	wm_poi_id	aor_id	order_price_interval	order_timestamp	ord_period_name	order_scene_name	aoi_id	takedlvr_aoi_type_name	dt
    # === 不加入其他特征
    le = LabelEncoder()
    # data['order_price_interval'] = le.fit_transform(data['order_price_interval'])
    # # data['order_scene_name'] = data['order_scene_name'].replace("未知", -1).astype(int)
    data['order_scene_name'] = le.fit_transform(data['order_scene_name'])
    # data['aoi_id'] =le.fit_transform(data['aoi_id'])
    data['aoi_id'].fillna(3699, inplace=True)
    data['takedlvr_aoi_type_name'] = le.fit_transform(data['takedlvr_aoi_type_name'])
    feature_columns = [[ ],[
        sparseFeature('wm_poi_id', npois + 1, embed_dim),   # behavior 1
        # other features
        # sparseFeature('user_id', nusers + 1, embed_dim),
        # sparseFeature('aor_id', 11, embed_dim),                     # data['aor_id'].nunique() = 11
        # # sparseFeature('order_price_interval', 5, embed_dim),        # data['order_price_interval'].nunique() = 5    ['<29', '>=65', '[29,36)', '[36,49)', '[49,65)']
        # sparseFeature('ord_period_name', 5, embed_dim),             # data['ord_period_name'].nunique() =  5
        # # sparseFeature('order_scene_name', 10, embed_dim),           # data['order_scene_name'].nunique() = 10       [未知 0-9]
        # sparseFeature('aoi_id', 3551, embed_dim),                   # data['aoi_id'].nunique() = 3625               [nan 0-3623]
        # sparseFeature('takedlvr_aoi_type_name', 13, embed_dim),     # data['takedlvr_aoi_type_name'].nunique() = 13 [未知 0-11]
    ] + [sparseFeature(f, c, embed_dim) for f, c in sparse_feature_count.items()]]
    # behavior
    behavior_list = ['wm_poi_id']


    data_feat = data[[
        'user_id', 'aor_id', 'order_price_interval', 'ord_period_name', 'order_scene_name', 'aoi_id', 'takedlvr_aoi_type_name',
        "wm_poi_id", 'wm_order_id', 'clicks'
    ]]
    data_seq = data[['wm_order_id', "clicks", "wm_poi_id"]]
    data_gen = []
    for wm_order_id, clicks, wm_poi_id in tqdm(data_seq.values):
        def gen_neg():
            pos_list = clicks + [wm_poi_id]
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, npois - 1)
            return neg
        neg_list = [gen_neg() for i in range(10)]
        # 1:10 生成负例
        data_gen.append([wm_order_id, [wm_poi_id], 1])
        for neg in neg_list:
            data_gen.append([wm_order_id, [neg], 0])

    data_gen = pd.DataFrame(data_gen, columns=['wm_order_id', 'item_id', 'label'])
    data = pd.merge(data_gen, data_feat, on="wm_order_id", how='left')

    print('==================Padding===================')
    data_X = [
        np.array([0.0] * len(data)),
        # np.concatenate([data[f].values.reshape(-1,1) for f in ['user_id', 'aor_id', 'order_price_interval', 'ord_period_name', 'order_scene_name', 'aoi_id', 'takedlvr_aoi_type_name']], axis=-1),
        np.concatenate([data[f].values.reshape(-1, 1) for f in
                        sparse_feature_count.keys()], axis=-1),
        pad_sequences(data['clicks'], maxlen=maxlen)[..., np.newaxis],
        np.array(data['item_id'].tolist())
    ]
    data_y = data['label'].values

    n_samples = len(data_y)
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    data_X = [a[idx] for a in data_X]
    data_y = data_y[idx]
    sp1, sp2 = int(n_samples*.1), int(n_samples*.2)
    test_X, val_X, train_X = [d[:sp1] for d in data_X], [d[sp1:sp2] for d in data_X], [d[sp2:] for d in data_X]
    test_y, val_y, train_y = data_y[:sp1], data_y[sp1:sp2], data_y[sp2:]
    # data_X, test_X, data_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=2020)
    # train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=0.2, random_state=2021)

    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)

def create_test_dataset(datapath, embed_dim=8, maxlen=40):
    print('==========Data Preprocess Start============')
    d_session = pd.read_csv(datapath+"orders_poi_session.txt", sep='\t')
    d_test = pd.read_csv(datapath + "orders_test_poi.txt", encoding="utf-8", sep='\t')
    """
    wm_order_id	clicks	dt
    user_id	wm_order_id	aor_id	order_timestamp	ord_period_name	aoi_id	takedlvr_aoi_type_name	dt
    """
    data = pd.merge(d_test, d_session, on="wm_order_id")
    data['clicks'] = data['clicks'].apply(lambda x: [int(i) for i in x.split('#')] if not pd.isnull(x) else [])

    le = LabelEncoder()
    # data['order_price_interval'] = le.fit_transform(data['order_price_interval'])
    # data['order_scene_name'] = le.fit_transform(data['order_scene_name'])
    # data['aoi_id'] = le.fit_transform(data['aoi_id'])
    data['aoi_id'].fillna(3699, inplace=True)
    data['takedlvr_aoi_type_name'] = le.fit_transform(data['takedlvr_aoi_type_name'])

    # 返回特征
    feature_columns = [[ ],[
        sparseFeature('wm_poi_id', npois + 1, embed_dim),   # behavior 1
    ] + [sparseFeature(f, c, embed_dim) for f, c in sparse_feature_count.items()]]
    # behavior
    behavior_list = ['wm_poi_id']


    print('==================Padding===================')
    seq_input = pad_sequences(data['clicks'], maxlen=maxlen)[..., np.newaxis]
    # item_input = np.arange(npois)[..., np.newaxis]
    data_X = [
        np.array([0.] * len(data)),
        # np.array([0.] * len(data)),
        np.concatenate([data[f].values.reshape(-1, 1) for f in
                        sparse_feature_count.keys()], axis=-1),
        seq_input
    ]

    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, data_X, data['wm_order_id']

if __name__ == '__main__':
    fpath = "../data/SMP新数据-20210607-20210702/"
    feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y) = \
        create_mdd_dataset(fpath, 8, 20)
    pickle.dump((feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)),
                open('../data/data-DIN.pkl', 'wb'))
    feature_columns, behavior_list, test_X, wm_order_id = \
        create_test_dataset(fpath, 8, 20)
    pickle.dump((feature_columns, behavior_list, test_X, wm_order_id),
                open('../data/data-DIN-test.pkl', 'wb'))