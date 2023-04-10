import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split

np.random.seed(123)

# 设置基础路径
base_dir = './HAM10000'
# 合并图像路径到字典中
file_paths = glob(os.path.join(base_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in file_paths}
# 用于显示更友好的标签的字典
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# 读取皮肤数据元数据文件
metadata = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))

# 利用上面创建的imageid_path_dict合并图像路径到元数据框中
metadata['path'] = metadata['image_id'].map(imageid_path_dict.get)
metadata['cell_type'] = metadata['dx'].map(lesion_type_dict.get) 
metadata['cell_type_idx'] = pd.Categorical(metadata['cell_type']).codes
metadata['age'].fillna((metadata['age'].mean()), inplace=True)

# 移除目标变量从特征中
features=metadata.drop(columns=['cell_type_idx'],axis=1)
target=metadata['cell_type_idx']

# 将数据集划分为训练集和测试集
train_features, test_features, train_targets, test_targets = train_test_split(features, target, test_size=0.005, random_state=6324)

# 仅保存一张图像作为训练数据
train_features_1 = train_features.iloc[[0,1,2,3,4]]
train_targets_1 = train_targets.iloc[[0,1,2,3,4]]

print('without one-hot',train_targets_1)
# 获取测试图像的路径
test_image_paths = test_features['path']

# 定义独热编码函数
def one_hot_encode(labels):
    n_labels = np.max(labels) + 1

    # 
    n_labels = 7
    one_hot = np.eye(n_labels)[labels]
    return one_hot

# 对训练集和测试集进行独热编码
train_targets_1, train_targets, test_targets = one_hot_encode(np.array(train_targets_1)), one_hot_encode(np.array(train_targets)), one_hot_encode(np.array(test_targets))

print('with one-hot',train_targets_1)
# 将数据转换为字典格式
train_data = {
    'img_path': np.array(train_features['path']),
    'label': np.array(train_targets)
}
test_data = {
    'img_path': np.array(test_image_paths),
    'label': np.array(test_targets)
}

# 仅保存包含一张图像的训练数据
train_data_1 = {
    'img_path': np.array(train_features_1['path']),
    'label': np.array(train_targets_1)
}

# 将元数据保存到npy文件中
np.save(os.path.join(base_dir, 'train_meta_1.npy'), train_data_1)
np.save(os.path.join(base_dir, 'train_meta.npy'), train_data)
np.save(os.path.join(base_dir, 'test_meta.npy'), test_data)

# data = np.load(os.path.join(base_dir, 'train_meta_1.npy'),allow_pickle=True).item()
# print('shape :', data['label'].shape)
# print('data :')
# print(data['label'][0])

# data = np.load(os.path.join(base_dir, 'train_meta.npy'),allow_pickle=True).item()
# print('shape :', data['label'].shape)
# print('data :')
# print(data['label'][0])