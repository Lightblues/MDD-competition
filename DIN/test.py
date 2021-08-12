from model import DIN
from utils import create_mdd_dataset

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import binary_crossentropy

file = '../data/原数据-20210301-20210328/'
maxlen = 20

embed_dim = 8
att_hidden_units = [80, 40]
ffn_hidden_units = [256, 128, 64]
dnn_dropout = 0.5
att_activation = 'sigmoid'
ffn_activation = 'prelu'

learning_rate = 0.001
batch_size = 4096
epochs = 5

# ========================== Create dataset =======================
feature_columns, behavior_list, train, val, test = create_mdd_dataset(file, embed_dim, maxlen)
train_X, train_y = train
val_X, val_y = val
test_X, test_y = test


model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
            ffn_activation, maxlen, dnn_dropout)
model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
              metrics=[AUC()])

check_path = 'save/din_weights.epoch_0005.val_loss_0.1283.ckpt'
# check_path = 'save'
model.load_weights(check_path)
# model.built = True

# 重新评估模型
# loss,acc = model.evaluate(test_X, test_y, batch_size=batch_size, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
out = model(test_X)