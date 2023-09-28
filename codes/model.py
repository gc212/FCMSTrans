import os
import numpy as np
import sys
# from Multi_scale import MultiScaleConv
from Mutil_scale_prot import MultiScaleConvA
from Mutil_scale_pspp import MultiScaleConvB
import tensorflow as tf
from Encoder import Encoder


def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (
                tf.ones_like(y_true) - y_pred) + tf.keras.backend.epsilon()
        focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    return binary_focal_loss_fixed


def get_model():
    # inputESM=tf.keras.layers.Input(shape=(201,1280))#输入的ESM所提取的特征（201*1280）
    inputESM = tf.keras.layers.Input(shape=(121, 1280))
    # inputFeature=tf.keras.layers.Input(shape=(48,))#输入的关于蛋白质结构、功能的特征（48维）
    inputProt = tf.keras.layers.Input(shape=(121, 1024)) # 输入prottrans特征
    inputPSPP = tf.keras.layers.Input(shape=(9, 27))  # 输入pssm_ss_pdo_psa特征
    sequence = tf.keras.layers.Dense(512)(inputESM)
    sequence = tf.keras.layers.Dense(256)(sequence)
    sequence = Encoder(2, 256, 4, 1024, rate=0.3)(sequence)
    # sequence=sequence[:,100,:]
    sequence = sequence[:, 60, :]
    sequence_prot = tf.keras.layers.Dense(512)(inputProt)
    sequence_prot = tf.keras.layers.Dense(256)(sequence_prot)
    Prot = MultiScaleConvA()(sequence_prot)
    PSPP = MultiScaleConvB()(inputPSPP)
    sequenceconcat = tf.keras.layers.Concatenate()([sequence, Prot, PSPP])
#     sequenceconcat = tf.keras.layers.Concatenate()([sequence, Prot])
#     feature = tf.keras.layers.Dense(2048, activation='relu')(sequenceconcat)
#     feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(1024, activation='relu')(sequenceconcat)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(512, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(256, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    feature = tf.keras.layers.Dense(128, activation='relu')(feature)
    feature = tf.keras.layers.Dropout(0.4)(feature)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(feature)
    qa_model = tf.keras.models.Model(inputs=[inputESM, inputProt, inputPSPP], outputs=y)
#     qa_model = tf.keras.models.Model(inputs=inputESM, outputs=y)
    adam = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
    # 改
    qa_model.compile(loss=[binary_focal_loss(alpha=.6, gamma=2)], optimizer=adam, metrics=['accuracy'])
    # qa_model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)],optimizer=adam,metrics=['accuracy'])
    qa_model.summary()
    return qa_model
