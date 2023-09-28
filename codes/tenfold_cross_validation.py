import tensorflow as tf
import numpy as np
from model import get_model
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef, f1_score, roc_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve
import openpyxl as op


filename = '/home/result/PRE/PRE_tenFold_crossValidation.xlsx'


def op_toexcel(data, file): # openpyxl库储存 数据到excel
    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]
        ws.append(data) # 每次写入一行
        wb.save(filename)
    else:
        wb = op.Workbook()  # 创建工作簿对象
        ws = wb['Sheet']  # 创建子表
        ws.append(['MCC', 'ACC', 'AUC', 'SN', 'SP', 'F1', 'TP', 'TN', 'FP', 'FN'])  # 添加表头
        ws.append(data) # 每次写入一行
        wb.save(filename)


def data_generator(train_esm, train_prot, train_pspp, train_y, batch_size):
    L = train_esm.shape[0]

    while True:
        for i in range(0, L, batch_size):
            batch_esm = train_esm[i:i + batch_size].copy()
            batch_prot = train_prot[i:i + batch_size].copy()
            batch_pspp = train_pspp[i:i + batch_size].copy()
            batch_y = train_y[i:i + batch_size].copy()

            batch_pspp = np.array(batch_pspp).reshape(len(batch_pspp), 9, 27)

            yield ([batch_esm, batch_prot, batch_pspp], batch_y)


def cross_validation(train_esm, train_prot, train_pspp, train_y, valid_esm, valid_prot, valid_pspp, valid_y,
                     test_esm, test_prot, test_pspp, test_y, k):

    # 训练、验证each epoch的步长
    train_size = train_pspp.shape[0]
    val_size = valid_pspp.shape[0]
    batch_size = 32
    train_steps = train_size // batch_size
    val_steps = val_size // batch_size

    test_pspp = np.array(test_pspp).reshape(len(test_pspp), 9, 27)

    print(train_pspp.shape)
    print(valid_pspp.shape)
    print(test_pspp.shape)

    print(f"Fold {k} - Training samples: {train_esm.shape[0]}, Test samples: {test_esm.shape[0]}")

    qa_model = get_model()
    valiBestModel = f'./save_model/ten_fold_model/model_regular_fold.h5'
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=valiBestModel,
        monitor='val_loss',
        save_weights_only=True,
        verbose=1,
        save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        verbose=0,
        mode='auto'
    )


    train_generator = data_generator(train_esm, train_prot, train_pspp, train_y, batch_size)
    val_generator = data_generator(valid_esm, valid_prot, valid_pspp, valid_y, batch_size)

    history_callback = qa_model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=10000,
        verbose=1,
        callbacks=[checkpointer, early_stopping],
        validation_data=val_generator,
        validation_steps=val_steps,
        shuffle=True,
        workers=1
    )
    train_generator.close()
    val_generator.close()

    print(f"\nFold {k} - Validation Loss: {history_callback.history['val_loss'][-1]:.4f}, " +
          f"Validation Accuracy: {history_callback.history['val_accuracy'][-1]:.4f}")

    print(f"Fold {k} - Testing:")
#         cross_validation_test(valiBestModel, test_esm, test_prot, test_pspp, test_y)

    y_pred = qa_model.predict([test_esm, test_prot, test_pspp]).reshape(-1,)

    y_pred_new = []
    for value in y_pred:
        if value < 0.5:
            y_pred_new.append(0)
        else:
            y_pred_new.append(1)
    y_pred_new = np.array(y_pred_new)
    tn, fp, fn, tp = confusion_matrix(test_y, y_pred_new).ravel()

    fpr, tpr, thresholds = roc_curve(test_y, y_pred_new, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    print("Matthews相关系数: " + str(matthews_corrcoef(test_y, y_pred_new)))
    print("ACC: ", (tp+tn) / (tp+tn+fp+fn))
    print("AUC: ", roc_auc)

    mcc = float(format((matthews_corrcoef(test_y, y_pred_new)), '.4f'))
    acc = float(format((tp+tn) / (tp+tn+fp+fn), '.4f'))
    auc = float(format(roc_auc, '.4f'))
    sn = float(format(tp / (tp + fn), '.4f'))
    sp = float(format(tn / (tn + fp), '.4f'))
    f1 = float(format(f1_score(test_y, y_pred_new), '.4f'))


    # 保存每一次跑的结果到excel表格
    result = mcc, acc, auc, sn, sp, f1, tp, tn, fp, fn
    op_toexcel(result, filename)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    all_esm = np.lib.format.open_memmap('/hy-tmp/features_npy/esm/PRE/PRE_esm.npy')
    all_prot = np.lib.format.open_memmap('/hy-tmp/features_npy/prottrans/PRE/PRE_prot.npy')
    all_pspp = np.lib.format.open_memmap('/hy-tmp/features_npy/pssm_ss_psa_pdo/PRE/PRE_pspp.npy')

    all_y = all_pspp[:, 0]
    all_PSPP = all_pspp[:, 1:]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    k = 1
    for train_index, test_index in cv.split(all_PSPP, all_y):
        # 训练集
        train_Esm = all_esm[train_index]
        train_Prot = all_prot[train_index]
        train_Pspp = all_PSPP[train_index]
        train_Y = all_y[train_index]

        # 打乱训练集顺序并划分出验证集
        # （1）分层打乱
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_inx, valid_inx in split.split(train_Pspp, train_Y):
            # 验证集
            valid_Esm = train_Esm[valid_inx]
            valid_Prot = train_Prot[valid_inx]
            valid_Pspp = train_Pspp[valid_inx]
            valid_Y = train_Y[valid_inx]
            # 训练集
            train_Esm = train_Esm[train_inx]
            train_Prot = train_Prot[train_inx]
            train_Pspp = train_Pspp[train_inx]
            train_Y = train_Y[train_inx]

        # 测试集
        test_Esm = all_esm[test_index]
        test_Prot = all_prot[test_index]
        test_Pspp = all_PSPP[test_index]
        test_Y = all_y[test_index]

        cross_validation(train_Esm, train_Prot, train_Pspp, train_Y, valid_Esm, valid_Prot, valid_Pspp, valid_Y,
                         test_Esm, test_Prot, test_Pspp, test_Y, k)

        k += 1











