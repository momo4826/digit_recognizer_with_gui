# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import random
import numpy as np
import torch
from torch.utils import data
import multiprocessing
import matplotlib.pyplot as plt

#constants
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784
NUM_OF_WORKERS = 4  #multiprocessing.cpu_count()

######################## download files ###################################
def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)

####################### load data into arrays ####################################
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    #Interpret a buffer as a 1-dimensional array.
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)


class MNISTData(data.Dataset):
    #image_file is gzip here and we read it to arrays, so if you want to apply transforms to images, you can first apply transforms.ToPILImage() in order to transform arrays to PIL images, and then apply any transforms you want.
    #I decided to not apply any transforms here.
    def __init__(self, mode = 'train', dataset=dataset, transform=None, target_transform=None):
        self.channel_num = 1
        self.image_size = 28
        if mode == 'train':
            self.label_arrays = torch.tensor(dataset['train_label']).to(torch.int64)
            # normalize
            image_arrays = dataset['train_img'].astype(np.float32) / 255.0
            #reshape
            self.image_arrays = torch.tensor(image_arrays.reshape(-1, self.channel_num, self.image_size, self.image_size)).to(torch.float32)
        elif mode == 'test':
            self.label_arrays = torch.tensor(dataset['test_label']).to(torch.int64)
            image_arrays = dataset['test_img'].astype(np.float32) / 255.0
            self.image_arrays = torch.tensor(
                image_arrays.reshape(-1, self.channel_num, self.image_size, self.image_size)).to(torch.float32)
        self.transform = transform
        self.target_transform = target_transform
        self.data_len = len(self.label_arrays)

        #print('Finished reading Dataset ({} samples found)'.format(self.data_len))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image = self.image_arrays[index]
        label = self.label_arrays[index]
        return image, label  # 返回每一个index对应的图片数据和对应的label


def gen_loader(batch_size = 128, mode = 'train'):
    if mode.lower() == 'train':
        training_data = MNISTData('train')
        return data.DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last= True, num_workers=NUM_OF_WORKERS)
    elif mode.lower() == 'test':
        test_data = MNISTData('test')
        return data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=NUM_OF_WORKERS)


def check_the_images(loader):
    images, labels = next(iter(loader))
    print('batch size: {}'.format(len(labels)))
    idx = random.randint(0, len(labels)-1)
    image = images[idx].squeeze()
    label = labels[idx]
    print('the label of this image:{}'.format(label))
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    # download_mnist()

    # #check the dataset type and shape
    # with open(save_file, 'rb') as f:
    #     dataset = pickle.load(f)
    # print(dataset['train_label'].shape)
    # print(dataset['train_label'][0])
    # print(dataset['test_img'].shape)
    # print(dataset['test_img'][0])

    test_size, test_loader = gen_loader(128, 'test')
    check_the_images(test_loader)
    # for i, (X, y) in enumerate(test_loader):
    #     print(X[0].dtype, y[0].dtype)
    #     print(X.shape, y.shape)
    #     break



