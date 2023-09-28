import pandas as pd
import numpy as np
import os
import torch
import esm
import sys
import h5py
from sklearn.model_selection import train_test_split

"""
将esm模型提取的特征划分为训练集和测试集（划分随机种子设置为42），并且将训练集和测试集保存为.npy文件
"""
project_path = "D:\gc\project\TransPPMP"
rootpath = os.path.abspath(project_path)
sys.path.append(rootpath)
FASTA_PATH = rootpath + "/data/PRE_cut/PRE_121.fasta"
EMB_PATH = rootpath + "/MMP_PRE_PMD/PRE_esm_121"
EMB_LAYER = 33

ys = []
Xs = []
esm_header = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('_')[1]
    esm_header.append(header.split('_')[0] + '_' + header.split('_')[1] + '_' + header.split('_')[2])
    ys.append(float(scaled_effect))
    header = header.replace('|', '_').replace('\\', '_').replace('/', '_').replace('*', '_').replace(':', '_').replace(
        '?', '_')
    # fn = f'{EMB_PATH}/{header[1:]}.pt'
    fn = f'{EMB_PATH}/{header[0:]}.pt'
    embs = torch.load(fn)
    # Xs.append(embs['mean_representations'][EMB_LAYER])
    Xs.append(embs['representations'][EMB_LAYER])

Xs = torch.stack(Xs, dim=0).numpy()
ys = np.array(ys)
esm_header = np.array(esm_header)


train_size = 0.8
# Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, train_size=train_size, random_state=42)
Xs_train, Xs_test, esm_header_train, esm_header_test = train_test_split(Xs, esm_header, train_size=train_size, random_state=42)

print(Xs_train.shape, Xs_test.shape)
print(esm_header_train.shape, esm_header_test.shape)
# np.save(rootpath + '/features_npy/esm/MMP/train_9.npy', Xs_train)
# np.save(rootpath + '/features_npy/esm/MMP/test_9.npy', Xs_test)
# print("esm特征文件保存完成！")

"""
将prottrans提取的特征根据esm_header划分为训练集和测试集并保存为.npy文件
"""
#
# with h5py.File(rootpath + "/features/prottrans/protein_embeddings_PRE_residue.h5", 'r') as f:
#     protein = []
#     embeddings = []
#     for key in f.keys():
#         # print(type(key))
#         embeddings.append(f[key][:])
#         key = key.split('_')
#         key = key[0] + '_' + key[1] + '_' + key[2]
#         protein.append(key)
#
#
#     protein = np.array(protein)
#     embeddings = np.array(embeddings)
#     print(protein.shape, protein[0:5])
#     print(embeddings.shape, embeddings[0:5])

def gen_train_test(esm_header, header, features):
    train = []
    tou = []
    for _header in esm_header:
        for _index, head in enumerate(header):
            if _header == head:
                tou.append(head)
                train.append(features[_index])
            else:
                continue

    train = np.array(train)
    tou = np.array(tou)

    return train, tou

# train_prottrans, train_tou = gen_train_test(esm_header_train, protein, embeddings)
# test_prottrans, test_tou = gen_train_test(esm_header_test, protein, embeddings)
#
# print(train_prottrans.shape,train_tou[0:5])
# print(esm_header_train.shape, esm_header_train[0:5])
# print(test_prottrans.shape, test_tou[0:5])
# print(esm_header_test.shape, esm_header_test[0:5])
#
# np.save(rootpath + '/features_npy/prottrans/PRE/train.npy', train_prottrans)
# np.save(rootpath + '/features_npy/prottrans/PRE/test.npy', test_prottrans)
#
# print("prottrans特征文件保存完成！")


"""
将pssm、pdo、ss、psa根据esm_header划分为训练集和测试集并保存为.npy文件
"""
#
pppp_file = pd.read_excel(rootpath + '/features/pssm_ss_psa_pdo/PRE/PRE_PSPP_9.xlsx')
pppp_header = pppp_file.loc[16:, 'Name']
pppp_features = pppp_file.loc[16:, 'Label':'psa26']

pppp_header = np.array(pppp_header)
pppp_features = np.array(pppp_features)

train_pppp, train_tou = gen_train_test(esm_header_train, pppp_header, pppp_features)
test_pppp, test_tou = gen_train_test(esm_header_test, pppp_header, pppp_features)

print(train_pppp.shape, train_tou[0:5])
print(esm_header_train.shape, esm_header_train[0:5])
print(test_pppp.shape, test_tou[0:5])
print(esm_header_test.shape, esm_header_test[0:5])
#
# np.save(rootpath + '/features_npy/pssm_ss_psa_pdo/PRE/train.npy', train_pppp)
# np.save(rootpath + '/features_npy/pssm_ss_psa_pdo/PRE/test.npy', test_pppp)
# print("PPPP特征文件保存完成！")


