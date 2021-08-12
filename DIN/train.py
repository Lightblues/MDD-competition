import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from matplotlib import pyplot as plt

from model import DIN
from utils import *

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # file = '../data/原数据-20210301-20210328/'
    file = "../data/SMP新数据-20210607-20210702/"
    maxlen = 20
    
    embed_dim = 8
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'
    ffn_activation = 'prelu'

    learning_rate = 0.001
    batch_size = 4096
    epochs = 200
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = create_mdd_dataset(file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test
    # ============================Build Model==========================
    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation, 
        ffn_activation, maxlen, dnn_dropout)
    model.summary()
    # ============================model checkpoint======================
    check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # check_path = 'save/din_weights.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                    verbose=1, save_freq='epoch',
                                                    save_best_only=True
                                                    )
    # model.load_weights(check_path)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # 早停
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.01, verbose=1),  # 调整学习率
        checkpoint,
    ]

    history = model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        callbacks=callbacks,
        validation_data=(val_X, val_y),
        batch_size=batch_size,
    )
    ## """可视化下看看训练情况"""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./DIN-base-seq-loss.png')
    # plt.show()
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])

"""
Epoch 1/5
418/418 [==============================] - 50s 113ms/step - loss: 0.1937 - auc: 0.9722 - val_loss: 0.3197 - val_auc: 0.9789
Epoch 2/5
418/418 [==============================] - 49s 118ms/step - loss: 0.1428 - auc: 0.9790 - val_loss: 0.1416 - val_auc: 0.9798
Epoch 3/5
418/418 [==============================] - 49s 116ms/step - loss: 0.1405 - auc: 0.9799 - val_loss: 0.1395 - val_auc: 0.9807
Epoch 4/5
418/418 [==============================] - 48s 115ms/step - loss: 0.1375 - auc: 0.9813 - val_loss: 0.1371 - val_auc: 0.9820
Epoch 5/5
418/418 [==============================] - 47s 113ms/step - loss: 0.1315 - auc: 0.9836 - val_loss: 0.1305 - val_auc: 0.9843
53/53 [==============================] - 3s 53ms/step - loss: 0.1294 - auc: 0.9844
test AUC: 0.984377
"""

