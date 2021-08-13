from model import DIN
from utils import create_mdd_dataset, sparseFeature, denseFeature, \
    create_test_dataset, npois


import pandas as pd
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import binary_crossentropy

maxlen = 20

embed_dim = 8
att_hidden_units = [80, 40]
ffn_hidden_units = [256, 128, 64]
dnn_dropout = 0.5
att_activation = 'sigmoid'
ffn_activation = 'prelu'

learning_rate = 0.001
batch_size = 4096

# test_path = '../data/原数据-20210301-20210328/'
test_path = "../data/SMP新数据-20210607-20210702/"
feature_columns, behavior_list, test_X, wm_order_id = create_test_dataset(test_path, embed_dim, maxlen)


# ========================== Create dataset =======================
# feature_columns, behavior_list, train, val, test = create_mdd_dataset(file, embed_dim, maxlen)
# train_X, train_y = train
# val_X, val_y = val
# test_X, test_y = test

model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
            ffn_activation, maxlen, dnn_dropout)
model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
              metrics=[AUC()])

"""
# seq
din_weights.epoch_0037.val_loss_0.0806.ckpt
# 
save-DIN/din_weights.epoch_0009.val_loss_0.0655.ckpt
"""
check_path = 'save-DIN/din_weights.epoch_0009.val_loss_0.0655.ckpt'
# check_path = 'save'
model.load_weights(check_path)

# model.built = True

# 重新评估模型
# loss,acc = model.evaluate(test_X, test_y, batch_size=batch_size, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

nsamples = 9999

test_num = len(test_X[0])
# pred = []
fname = '../output/DIN_base_seq.txt'
outf = open(fname, 'w')
for dense_input, sparse_input, seq_input, order_id in tqdm(list(zip(*test_X, wm_order_id))):
    data = [
        np.repeat(dense_input, npois),
        np.repeat(sparse_input, npois),
        np.repeat([seq_input], npois, axis=0),
        np.arange(npois)[..., np.newaxis]
    ]
    out = model(data).numpy().squeeze()
    topK = np.argsort(out)[::-1][:5]
    # pred.append(topK)
    outf.write("{}\t{}\n".format(order_id, "\t".join([str(i) for i in topK])))

# output = np.hstack([wm_order_id[..., np.newaxis][:nsamples], np.array(pred)])
