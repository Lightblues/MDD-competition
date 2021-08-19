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

    learning_rate = 0.01
    batch_size = 4096
    epochs = 200
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = create_mdd_dataset(file, embed_dim, maxlen)
    # feature_columns, behavior_list, train, val, test = pickle.load(open('../data/data-DIN.pkl', 'rb'))
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test
    # ============================Build Model==========================
    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation, 
        ffn_activation, maxlen, dnn_dropout)
    model.summary()
    # ============================model checkpoint======================
    check_path = 'save-DIN/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
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
    # train_X_ = [a[:20] for a in train_X]
    # train_y_ = model(train_X_)
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
    plt.savefig('./DIN-base-loss.png')
    # plt.show()
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])

