import tensorflow as tf
import numpy as np
from model import get_model
import os
from my_test import test


def data_generator(train_esm, train_prot, train_pspp, batch_size):

    L = train_pspp.shape[0]
    train_y = train_pspp[:, 0]
    train_pspp = train_pspp[:, 1:]

    while True:
        for i in range(0, L, batch_size):
            batch_esm = train_esm[i:i+batch_size]
            batch_prot = train_prot[i:i+batch_size]
            batch_pspp = train_pspp[i:i+batch_size]

            print(batch_esm.shape)
            
            batch_pspp = np.array(batch_pspp).reshape(len(batch_pspp), 9, 27)
            print(batch_pspp.shape)

            yield ([batch_esm, batch_prot, batch_pspp], train_y[i:i+batch_size])


def train():
    train_esm = np.lib.format.open_memmap('/hy-tmp/features_npy/esm/PMD/train_0.9.npy')
    train_prot = np.lib.format.open_memmap('/hy-tmp/features_npy/prottrans/PMD/train_0.9.npy')
    train_pspp = np.lib.format.open_memmap('/hy-tmp/features_npy/pssm_ss_psa_pdo/PMD/train_0.9.npy')


    L = train_pspp.shape[0]
    val_split = 0.1  # 10% of the training data for validation
    val_size = int(L * val_split)
    train_size = L - val_size

    batch_size = 32
    train_steps = train_size // batch_size
    val_steps = val_size // batch_size


    qa_model = get_model()
    valiBestModel = './save_model/model_residue_train_0.9/model_regular.h5'
    checkpoiner = tf.keras.callbacks.ModelCheckpoint(filepath=valiBestModel, monitor='val_loss', save_weights_only=True,
                                                     verbose=1, save_best_only=True)
    earlyStopPatience = 10
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=earlyStopPatience, verbose=0,
                                                     mode='auto')

    log_dir = "logs/fit/model/sequenceModel"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)

    generator = data_generator(train_esm[:train_size], train_prot[:train_size], train_pspp[:train_size], batch_size)
    val_generator = data_generator(train_esm[train_size:], train_prot[train_size:], train_pspp[train_size:], batch_size)

    history_callback = qa_model.fit_generator(
        generator,
        steps_per_epoch=train_steps,
        epochs=10000,
        verbose=1,
        callbacks=[checkpoiner, earlystopping],
        validation_data=val_generator,
        validation_steps=val_steps,
        workers=1
    )
    
    generator.close()
    val_generator.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    train()
    test('/home/model/TransPPMP/save_model/model_residue_train_0.9/model_regular.h5')
