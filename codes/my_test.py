from sklearn.metrics import matthews_corrcoef, f1_score, roc_curve
from sklearn import metrics
import os
import numpy as np
from model import get_model
from sklearn.metrics import confusion_matrix
import openpyxl as op


filename = '/home/result/PMD/PMD_esm_prot_pspp.xlsx'

def op_toexcel(data,filename): # openpyxl库储存 数据到excel
    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]
        ws.append(data) # 每次写入一行
        wb.save(filename)
    else:
        wb = op.Workbook()  # 创建工作簿对象
        ws = wb['Sheet']  # 创建子表
        ws.append(['MCC', 'ACC', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1', 'FPR', 'FNR',
                  'TN', 'FP', 'FN', 'TP'])  # 添加表头
        ws.append(data) # 每次写入一行
        wb.save(filename)

def test(modelFile):
    test_esm = np.load('/hy-tmp/features_npy/esm/PMD/test_0.1.npy')
    test_prot = np.load('/hy-tmp/features_npy/prottrans/PMD/test_0.1.npy')
    test_pspp = np.load('/hy-tmp/features_npy/pssm_ss_psa_pdo/PMD/test_0.1.npy')

    y_true = test_pspp[:, 0]
    test_pspp = test_pspp[:, 1:]
    test_pspp = np.array(test_pspp).reshape(len(test_pspp), 9, 27)


    # load model
    train_model = get_model()
    train_model.load_weights(modelFile)
    y_pred = train_model.predict([test_esm, test_prot, test_pspp]).reshape(-1,)
    y_pred_new = []
    for value in y_pred:
        if value < 0.5:
            y_pred_new.append(0)
        else:
            y_pred_new.append(1)
    y_pred_new = np.array(y_pred_new)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_new, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    print("Matthews相关系数: " + str(matthews_corrcoef(y_true, y_pred_new)))
    print("ACC: ", (tp+tn) / (tp+tn+fp+fn))
    print("AUC: ", roc_auc)
    print('sensitivity/recall:', tp / (tp + fn))
    print('specificity:', tn / (tn + fp))
    print('precision:', tp / (tp + fp))
    print('negative predictive value:', tn / (tn + fn))
    print("F1值: " + str(f1_score(y_true, y_pred_new)))
    print('error rate:', fp / (tp + tn + fp + fn))
    print('false positive rate:', fp / (tn + fp))
    print('false negative rate:', fn / (tp + fn))
    print('TN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)

    mcc = float(format((matthews_corrcoef(y_true, y_pred_new)), '.4f'))
    acc = float(format((tp+tn) / (tp+tn+fp+fn), '.4f'))
    auc = float(format(roc_auc, '.4f'))
    sen = float(format(tp / (tp + fn), '.4f'))
    spe = float(format(tn / (tn + fp), '.4f'))
    pre = float(format(tp / (tp + fp), '.4f'))
    npv = float(format(tn / (tn + fn), '.4f'))
    f1 = float(format(f1_score(y_true, y_pred_new), '.4f'))
    fpr = float(format(fp / (tn + fp), '.4f'))
    fnr = float(format(fn / (tp + fn), '.4f'))

    # 保存每一次跑的结果到excel表格
    result = mcc, acc, auc, sen, spe, pre, npv, f1, fpr, fnr, tn, fp, fn, tp
    op_toexcel(result, filename)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test('/home/model/TransPPMP/save_model/model_residue_train_0.9/model_PMD_best_0.h5')
