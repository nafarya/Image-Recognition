import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys


def getDataSet(name, cifar):
    x = None
    y = None
    l = None

    downloadExtractDataSet()

    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    if (cifar == 10):
        f = open('./data_set/' + folder_name + '/batches.meta', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
        l = datadict['label_names']
    else:
        f = open('./data_set/' + folder_name + '/meta', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
        l = datadict['fine_label_names']


    if name == 'train10':
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name == "test10":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)

    elif name == "train100":
        f = open('./data_set/cifar_100/train', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
        _X = datadict["data"]
        _Y = np.array(datadict['fine_labels'])

        _X = np.array(_X, dtype=float) / 255.0
        _X = _X.reshape([-1, 3, 32, 32])
        _X = _X.transpose([0, 2, 3, 1])
        _X = _X.reshape(-1, 32 * 32 * 3)

        if x is None:
            x = _X
            y = _Y
        else:
            x = np.concatenate((x, _X), axis=0)
            y = np.concatenate((y, _Y), axis=0)

    elif name == "test100":
        f = open('./data_set/cifar_100/test', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
        x = datadict["data"]
        y = np.array(datadict['fine_labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32 * 32 * 3)


    def denseToOneHot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    print(x)
    print(denseToOneHot(y, cifar))
    print(l)

    return x, denseToOneHot(y, cifar), l


def printDownloadProgress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def downloadExtractDataSet():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    cifar_100_directory = main_directory+"cifar_100/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=printDownloadProgress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=printDownloadProgress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.rename(main_directory+"./cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_10)
        os.remove(zip_cifar_100)
