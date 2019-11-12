import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2

class ImageDataSet(Dataset):
    def __init__(self, images_dir,fold,transformation=None):
        print('#################',fold,'#############',)
        self.images_dir = images_dir
        if(fold=='test'):
            self.dataframe = pd.read_csv("./dataset/0/test_mapped.csv")
        else:
             self.dataframe = pd.read_csv("./dataset/0/train_mapped.csv") #mapped class labels file
             self.dataframe = self.dataframe[self.dataframe['fold'] == fold] #file contains both val and train as folds

        # if fold == 'train':
        #     self.dataframe = self.dataframe[0:40]
        # if fold == 'val':
        #     self.dataframe = self.dataframe[0:10]

        self.dataframe = self.dataframe.set_index("files")
        self.transform = transformation
        print(self.dataframe)
        print('#######################################')

    # dataset size
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # print('get',idx)
        filename = self.dataframe.index[idx] #file name as index
        image = Image.open(os.path.join(self.images_dir, filename)) #get image
        image = image.resize((224,224))
        image = image.convert('RGB')
        #get label for image
        label = self.dataframe['labels'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        return (image,label,filename)

