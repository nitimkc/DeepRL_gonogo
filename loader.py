# create iterable of train and test sets for CV

import numpy as np 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# from sklearn.model_selection import KFold 
# from sklearn.model_selection import train_test_split as tts

# # https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

def get_sampling_weights(N, labels, nclasses=2):
        count_per_class = [0] * nclasses
        for image_class in labels:
            count_per_class[image_class] += 1

        weight_per_class = [0.] * nclasses
        for i in range(nclasses):
            weight_per_class[i] = float(N) / float(count_per_class[i])

        weights = [0] * N
        for idx, image_class in enumerate(labels):
            weights[idx] = weight_per_class[image_class]
        return weights

class ImageLoader(Dataset):

    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def show_image(self, idx):
        """Show image with landmarks"""
        image, label = self.__getitem__(idx)
        plt.imshow(image)    
        plt.figure()
        plt.title(f"image label: {label}")
        plt.show()


# class ImageLoader(object):

#     def __init__(self, reader, folds=12, shuffle=True, size=None):
#         self.reader = reader
#         self.folds  = KFold(n_splits=folds, shuffle=shuffle)
#         self.idx = range(0,size) if size is not None else None

#     def images(self, idx=None):
#         items = list(self.reader._read_images(self.reader._fileids))
#         if idx is None:
#             return items
#         return  list(items[i] for i in idx)

#     def labels(self, idx=None):
#         items = list(self.reader.labels(self.reader._fileids))
#         if idx is None:
#             return items
#         return list(items[i] for i in idx)

#     def __iter__(self):
#         i = 1
#         for train_index, test_index in self.folds.split(self.idx):
#             print('iter', i)
#             print(len(train_index), len(test_index))
            
#             i = i+1
#             X_train = self.images(train_index)
#             y_train = self.labels(train_index)
#             X_test  = self.images(test_index)
#             y_test  = self.labels(test_index)

#             yield X_train, X_test, y_train, y_test