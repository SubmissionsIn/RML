import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from utils import noisify


def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min()
    max_val = input_matrix.max()
    input_range = max_val - min_val
    scaled_matrix = (input_matrix - min_val) / input_range * (max_value - min_value) + min_value
    return scaled_matrix


class MyDataset(Dataset):
    def __init__(self, path, trainset_rate=0.7, type='train', seed=0, dataname='xxx', mode='RML', noise_rate=0.5):
        scaler = MinMaxScaler()
        if dataname == 'WebKB':
            self.V1 = scipy.io.loadmat(path + 'WebKB')['X'][0][0].astype(np.float32)
            self.V2 = scipy.io.loadmat(path + 'WebKB')['X'][0][1].astype(np.float32)
            self.num = self.V1.shape[0]
            self.view_dims = [self.V1.shape[1], self.V2.shape[1]]
            self.Data = [self.V1, self.V2]
            self.Y = scipy.io.loadmat(path + 'WebKB')['gnd'].astype(np.int32).reshape(self.num,) - 1
            # print(self.Y)

        if dataname == 'DHA':
            self.V1 = (scipy.io.loadmat(path + 'DHA.mat')['X1'].astype(np.float32))
            self.V2 = (scipy.io.loadmat(path + 'DHA.mat')['X2'].astype(np.float32))
            self.num = self.V1.shape[0]
            self.view_dims = [self.V1.shape[1], self.V2.shape[1]]
            self.Data = [self.V1, self.V2]
            self.Y = scipy.io.loadmat(path + 'DHA.mat')['Y'].astype(np.int32).reshape(self.num, )

            # self.Data = [self.V1, self.V2, np.tile(self.Y.astype(np.float32).reshape(self.num, 1), (1, 1000))]
            # self.view_dims = [self.V1.shape[1], self.V2.shape[1], 1000]

        if dataname == 'Caltech':
            mat = scipy.io.loadmat("./data/Caltech.mat")
            # print(mat)
            scaler = MinMaxScaler()
            v1 = (mat['X1'].astype('float32'))
            v2 = (mat['X2'].astype('float32'))
            v3 = scale_normalize_matrix(mat['X3'].astype('float32'))   # 1
            v4 = (mat['X4'].astype('float32'))
            v5 = (mat['X5'].astype('float32'))
            v6 = (mat['X6'].astype('float32'))
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1], v3.shape[1], v4.shape[1], v5.shape[1], v6.shape[1]]
            self.Data = [v1, v2, v3, v4, v5, v6]
            # print(self.Data)
            y = np.squeeze(mat['Y']).astype('int')
            # print(y[0:100])       # 7 class labels are [6  2  3  3  4  1  4  95  6  2  4  2  4  1  4  6  0 ...]
            for i in range(len(y)):
                if y[i] == 95:
                    y[i] = 5  # cleaning labels to [0, 1, 2 ... K-1] for visualization
            self.Y = y

            # self.Data.append(np.tile(self.Y.astype(np.float32).reshape(self.num, 1), (1, 1000)))
            # self.view_dims.append(1000)

        if dataname == 'Caltech101_20':
            # mat = scipy.io.loadmat("./data/Caltech.mat")
            mat = scipy.io.loadmat("./data/Caltech101_20.mat")
            # print(mat)
            scaler = MinMaxScaler()
            v1 = (mat['X'][0][0].astype('float32'))
            v2 = (mat['X'][0][1].astype('float32'))
            v3 = scale_normalize_matrix(mat['X'][0][2].astype('float32'))   # 1
            v4 = (mat['X'][0][3].astype('float32'))
            v5 = (mat['X'][0][4].astype('float32'))
            v6 = (mat['X'][0][5].astype('float32'))
            self.num = v1.shape[0]
            self.view_dims = [v4.shape[1], v5.shape[1]]
            self.Data = [v4, v5]
            print(self.Data)
            y = np.squeeze(mat['Y']).astype('int') - 1
            self.Y = y

            # self.Data.append(np.tile(self.Y.astype(np.float32).reshape(self.num, 1), (1, 1000)))
            # self.view_dims.append(1000)

        if dataname == 'BDGP':
            data1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
            data2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
            labels = scipy.io.loadmat(path + 'BDGP.mat')['Y'][0]
            v1 = scale_normalize_matrix(data1)
            v2 = scale_normalize_matrix(data2)
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1]]
            self.Data = [v1, v2]
            self.Y = labels

            # self.Data.append(np.tile(self.Y.astype(np.float32).reshape(self.num, 1), (1, 1000)))
            # self.view_dims.append(1000)

        if dataname == 'NGs':
            self.Y = scipy.io.loadmat(path + 'NGs')['truelabel'][0][0].astype(np.int32).reshape(500, )
            self.V1 = scipy.io.loadmat(path + 'NGs')['data'][0][0].astype(np.float32)
            self.V2 = scipy.io.loadmat(path + 'NGs')['data'][0][1].astype(np.float32)
            self.V3 = scipy.io.loadmat(path + 'NGs')['data'][0][2].astype(np.float32)
            self.v1 = np.transpose(self.V1)
            self.v2 = np.transpose(self.V2)
            self.v3 = np.transpose(self.V3)
            # self.v1 = scale_normalize_matrix(self.V1)
            # self.v2 = scale_normalize_matrix(self.V2)
            # self.v3 = scale_normalize_matrix(self.V3)
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'VOC':
            self.Y = scipy.io.loadmat(path + 'VOC')['Y'].astype(np.int32).reshape(5649, )
            self.V1 = scipy.io.loadmat(path + 'VOC')['X1'].astype(np.float32)
            self.V2 = scipy.io.loadmat(path + 'VOC')['X2'].astype(np.float32)
            self.num = self.V1.shape[0]
            self.view_dims = [self.V1.shape[1], self.V2.shape[1]]
            self.Data = [self.V1, self.V2]

        if dataname == 'CCV':
            self.v1 = np.load(path + 'STIP.npy').astype(np.float32)
            scaler = MinMaxScaler()
            self.v1 = scaler.fit_transform(self.v1)
            self.v2 = np.load(path + 'SIFT.npy').astype(np.float32)
            self.v3 = np.load(path + 'MFCC.npy').astype(np.float32)
            self.Y = np.load(path + 'label.npy')
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'Fc_COIL_20':
            self.Y = scipy.io.loadmat(path + 'Fc_COIL_20')['Y'].astype(np.int32).reshape(1440, )
            self.v1 = scipy.io.loadmat(path + 'Fc_COIL_20')['X1'].astype(np.float32)
            self.v2 = scipy.io.loadmat(path + 'Fc_COIL_20')['X2'].astype(np.float32)
            self.v3 = scipy.io.loadmat(path + 'Fc_COIL_20')['X3'].astype(np.float32)
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'Fashion':
            self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000, )
            self.v1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32).reshape(10000, 784)
            self.v2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32).reshape(10000, 784)
            self.v3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32).reshape(10000, 784)
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'MNIST-USPS':
            self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
            self.v1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32).reshape(5000, 784)
            self.v2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32).reshape(5000, 784)
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1]]
            self.Data = [self.v1, self.v2]

        if dataname == 'RGB-D':
            mat = scipy.io.loadmat("./data/RGB-D.mat")
            self.v1 = (mat['X1'].astype('float32'))
            self.v2 = (mat['X2'].astype('float32'))
            self.Y = np.squeeze(mat['Y']).astype('int')
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1]]
            self.Data = [self.v1, self.v2]

        if dataname == 'Cora':
            mat = scipy.io.loadmat("./data/Cora.mat")
            # print(mat)
            self.v1 = (mat['coracites'].astype('float32'))
            self.v2 = (mat['coracontent'].astype('float32'))
            # self.v3 = (mat['corainbound'].astype('float32'))
            # self.v4 = (mat['coraoutbound'].astype('float32'))
            self.Y = np.squeeze(mat['y']).astype('int') - 1
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1]]
            self.Data = [self.v1, self.v2]

            # print(self.v1.shape, self.v2.shape, self.v3.shape, self.v4.shape)
            # self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1], self.v4.T.shape[1]]
            # self.Data = [self.v1, self.v2, self.v3, self.v4.T]

        if dataname == 'YoutubeVideo':
            mat = scipy.io.loadmat("./data/Video-3V.mat")
            self.v1 = mat['X1'].astype('float32')
            self.v2 = mat['X2'].astype('float32')
            self.v3 = mat['X3'].astype('float32')
            self.Y = np.squeeze(mat['Y']).astype('int') - 1  # cleaning labels to [0, 1, 2 ... K-1] for visualization
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'Prokaryotic':
            mat = scipy.io.loadmat("./data/Prokaryotic.mat")
            # print(mat)
            # print(mat['X'][0][0])
            # print(mat['X'][1][0])
            # print(mat['X'][2][0])
            # X_list.append(mat['X'][0][0].astype('float32'))
            # X_list.append(mat['X'][1][0].astype('float32'))
            # X_list.append(mat['X'][2][0].astype('float32'))
            # scaler.fit_transform  scale_normalize_matrix
            self.v1 = (mat['gene_repert'].astype('float32'))
            self.v2 = scaler.fit_transform(mat['proteome_comp'].astype('float32'))
            self.v3 = scaler.fit_transform(mat['text'].astype('float32'))
            self.Y = np.squeeze(mat['Y']).astype('int') - 1
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'Synthetic3d':
            mat = scipy.io.loadmat("./data/Synthetic3d.mat")
            # print(mat)
            self.v1 = mat['X'][0][0].astype('float32')
            self.v2 = mat['X'][1][0].astype('float32')
            self.v3 = mat['X'][2][0].astype('float32')
            self.Y = np.squeeze(mat['Y']).astype('int')
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'LandUse_21':
            mat = scipy.io.loadmat("./data/LandUse-21.mat")
            print(mat)
            self.v1 = scaler.fit_transform(mat['X'][0][0].astype('float32'))
            self.v2 = scaler.fit_transform(mat['X'][0][1].astype('float32'))
            self.v3 = scaler.fit_transform(mat['X'][0][2].astype('float32'))
            self.Y = np.squeeze(mat['Y']).astype('int')
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'Scene_15':
            mat = scipy.io.loadmat("./data/Scene_15.mat")
            print(mat)
            self.v1 = scaler.fit_transform(mat['X'][0][0].astype('float32'))
            self.v2 = scaler.fit_transform(mat['X'][0][1].astype('float32'))
            self.v3 = scaler.fit_transform(mat['X'][0][2].astype('float32'))
            self.Y = np.squeeze(mat['Y']).astype('int')
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'Cifar100':
            mat = scipy.io.loadmat("./data/cifar100.mat")
            # print(mat)
            # print(mat['data'][0][0].T.shape)
            # print(mat['data'][1][0].T.shape)
            # print(mat['data'][2][0].T.shape)
            # print(mat['truelabel'][0][0].T[0].shape)
            self.v1 = mat['data'][0][0].T.astype('float32')
            self.v2 = mat['data'][1][0].T.astype('float32')
            self.v3 = mat['data'][2][0].T.astype('float32')
            self.Y = np.squeeze(mat['truelabel'][0][0].T[0]).astype('int')
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]

        if dataname == 'pascal07_six_view':
            mat = scipy.io.loadmat("./data/pascal07_six_view.mat")
            print(mat)
            v1 = StandardScaler().fit_transform(mat['X'][0][0].astype('float32'))
            v2 = StandardScaler().fit_transform(mat['X'][0][1].astype('float32'))
            v3 = StandardScaler().fit_transform(mat['X'][0][2].astype('float32'))
            v4 = StandardScaler().fit_transform(mat['X'][0][3].astype('float32'))
            v5 = StandardScaler().fit_transform(mat['X'][0][4].astype('float32'))
            v6 = StandardScaler().fit_transform(mat['X'][0][5].astype('float32'))
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1], v3.shape[1], v4.shape[1], v5.shape[1], v6.shape[1]]
            self.Data = [v1, v2, v3, v4, v5, v6]
            y = np.squeeze(mat['label']).astype('int').argmax(1)
            print(y[0:10])
            self.Y = y

        if dataname == 'mirflickr_six_view':
            mat = scipy.io.loadmat("./data/mirflickr_six_view.mat")
            print(mat)
            v1 = StandardScaler().fit_transform(mat['X'][0][0].astype('float32'))
            v2 = StandardScaler().fit_transform(mat['X'][0][1].astype('float32'))
            v3 = StandardScaler().fit_transform(mat['X'][0][2].astype('float32'))
            v4 = StandardScaler().fit_transform(mat['X'][0][3].astype('float32'))
            v5 = StandardScaler().fit_transform(mat['X'][0][4].astype('float32'))
            v6 = StandardScaler().fit_transform(mat['X'][0][5].astype('float32'))
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1], v3.shape[1], v4.shape[1], v5.shape[1], v6.shape[1]]
            self.Data = [v1, v2, v3, v4, v5, v6]
            y = np.squeeze(mat['label']).astype('int').argmax(1)
            print(y[0:10])
            self.Y = y

        if dataname == 'NUS_WIDE':
            root = '../UCCH-main/data/NUS-WIDE-TC10/'
            import scipy.io as sio
            data_img = sio.loadmat(root + 'nus-wide-tc10-xall-vgg.mat')['XAll']
            data_txt = sio.loadmat(root + 'nus-wide-tc10-yall.mat')['YAll']
            labels = sio.loadmat(root + 'nus-wide-tc10-lall.mat')['LAll']
            print(data_img.shape, data_txt.shape, labels.shape)
            v1 = data_img.astype('float32')
            v2 = data_txt.astype('float32')
            y = labels.argmax(1)
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1]]
            self.Data = [v1, v2]
            self.Y = y

        if dataname == 'MSCOCO':
            import scipy.io as sio
            import h5py
            root = '../UCCH-main/data/MSCOCO/'
            path = root + 'MSCOCO_deep_doc2vec_data_rand.h5py'
            data = h5py.File(path)
            data_img = data['XAll'][()]
            data_txt = data['YAll'][()]
            labels = data['LAll'][()]
            print(data_img.shape, data_txt.shape, labels.shape)
            v1 = data_img.astype('float32')
            v2 = data_txt.astype('float32')
            y = labels.argmax(1)
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1]]
            self.Data = [v1, v2]
            self.Y = y

        if dataname == 'IAPR':
            import scipy.io as sio
            import os
            root = '../UCCH-main/data/IAPR-TC12/'
            file_path = os.path.join(root, 'iapr-tc12-rand.mat')
            data = sio.loadmat(file_path)

            valid_img = data['VDatabase'].astype('float32')
            valid_txt = data['YDatabase'].astype('float32')
            valid_labels = data['databaseL']

            test_img = data['VTest'].astype('float32')
            test_txt = data['YTest'].astype('float32')
            test_labels = data['testL']

            data_img, data_txt, labels = np.concatenate([valid_img, test_img]), np.concatenate(
                [valid_txt, test_txt]), np.concatenate([valid_labels, test_labels])

            print(data_img.shape, data_txt.shape, labels.shape)
            print(data_img)
            print(data_txt)
            v1 = scaler.fit_transform(data_img.astype('float32'))
            v2 = data_txt.astype('float32')
            y = labels.argmax(1)
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1]]
            self.Data = [v1, v2]
            self.Y = y

        if dataname == 'MIRFLICKR25K':
            import scipy.io as sio
            import os
            root = '../UCCH-main/data/MIRFLICKR25K/'
            data_img = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
            data_txt = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
            labels = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']
            print(data_img.shape, data_txt.shape, labels.shape)
            v1 = (data_img.astype('float32'))
            v2 = (data_txt.astype('float32'))
            y = labels.argmax(1)
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1]]
            self.Data = [v1, v2]
            self.Y = y

        self.view_num = len(self.Data)

        print("\n")
        # if mode == 'RML_LCE':
        index = list(range(0, self.num, 1))
        # print(index[0:10])
        random.seed(seed)
        random.shuffle(index)
        # print(index[0:10])
        for v in range(self.view_num):
            # print(self.Data[v].shape)
            self.Data[v] = self.Data[v][index]
        # print(self.Y.shape)
        self.Y = self.Y[index]

        self.classnum = np.unique(self.Y)

        if type == 'train':
            print("Training set...")
            self.num = int(trainset_rate*self.num)
            for v in range(self.view_num):
                self.Data[v] = self.Data[v][:self.num, :]
                print(self.Data[v].shape)
            self.Y = self.Y[:self.num]
            print(self.Y.shape)
            print(self.Y[0:10])
            if mode == 'RML_LCE':
                Ys = np.asarray([[self.Y[i]] for i in range(self.num)])
                if noise_rate > 0:
                    self.train_noisy_labels, _ = noisify(
                                                        nb_classes=len(self.classnum),
                                                        train_labels= Ys,
                                                        noise_type='symmetric',
                                                        noise_rate=noise_rate,
                                                        random_state=0
                                                    )
                else:
                    self.train_noisy_labels = Ys
                print("Noise label rate: " + str(noise_rate))
                self.Y = np.array(self.train_noisy_labels.T[0])
                print(self.train_noisy_labels.T[0][0:10])

        if type == 'test':
            print("Test set...")
            print(self.Y[0:10])
            self.num = int(trainset_rate*self.num)
            for v in range(self.view_num):
                self.Data[v] = self.Data[v][self.num:, :]
                print(self.Data[v].shape)
            self.Y = self.Y[self.num:]
            self.num = len(self.Y)
            print(self.Y.shape)

    def __len__(self):
        return self.num

    def __dims__(self):
        return self.view_dims, sum(self.view_dims)

    def __classnum__(self):
        return len(self.classnum)

    def __getitem__(self, idx):
        returndata = []
        for v in range(self.view_num):
            returndata.append(torch.from_numpy(self.Data[v][idx]))
        return returndata, self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset, trainset_rate=0.7, type='train', seed=0, mode='RML', noise_rate=0.5):
    dataset = MyDataset('./data/', trainset_rate=trainset_rate, type=type, seed=seed, dataname=dataset, mode=mode, noise_rate=noise_rate)
    dims, dimss = dataset.__dims__()
    view = len(dims)
    data_size = dataset.__len__()
    class_num = dataset.__classnum__()
    print('N:'+str(data_size), 'K:'+str(class_num), 'M:'+str(view), 'D:'+str(dims))
    return dataset, dims, view, data_size, class_num, dimss
