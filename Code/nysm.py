
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import DataLoader as tdataloader
import DataLoader as dataloader
import pandas as pd
import numpy as np

tr_loss_track=[]
val_loss_track=[]
num_epochs = 10
batch_size = 32
class_labels=136 #Subject to change
loader_works = 1 #multi-process data loading,#loader worker processes.
learning_rate = 0.01

images_dir='/content/drive/My Drive/25/train/'
images_dir_test='/content/drive/My Drive/25/test/'

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
 device = torch.device('cuda')
else:
 device = torch.device ('cpu')
print('device running on:', device)


def training():
  #For data augmentation
  transform ={'train':transforms.ToTensor(),'val':transforms.ToTensor()}

  #Loading the dataset with train and val as keys
  #Data divided in training and val using separate script

  datasets = {}
  datasets['train'] = dataloader.ImageDataSet(images_dir,fold='train',transformation=transform['train'])
  datasets['val'] = dataloader.ImageDataSet(images_dir,fold='val',transformation=transform['val'])
  print('datasets',len(datasets['train']))
  print('datasets',len(datasets['val']))

  dataloaders = {}
  dataloaders['train'] = tdataloader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=loader_works)
  dataloaders['val'] = tdataloader(datasets['val'], batch_size=batch_size, shuffle=True, num_workers=loader_works)
  print('dataloaders',len(dataloaders['train']))
  print('dataloaders',len(dataloaders['val']))

  #class labels can't be taken dirctly random needs mapping
  #obtaining the pretrained model

  model = models.vgg19_bn(pretrained=True)
  num_features = model.classifier[6].in_features
  print('num_features',num_features)
  model.classifier[6] = nn.Linear(num_features, class_labels+1) #handle 0 to class-1
  model = model.to(device)
  print(model.classifier)
  # print(model)

  criterian = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  #getting size of train and  validation datasets

  #training
  start_time=time.time()
  best_epoch = 0
  train_loss = -1
  best_loss = float('inf')
  best_model=None

  for epoch in range(1,num_epochs+1):
    print('epoch {}/{}'.format(epoch,num_epochs))
    for t in ['train','val']:

        r_loss = 0.0
        if t == 'train':
            model.train(True)
        else:
            model.train(False)
        # print('dataloaders',dataloaders[t])
        for data in dataloaders[t]:
            # print('inner for')
            files,labels = data
            files = Variable(files.to(device)) #to gpu or cpu
            labels = Variable(labels.to(device))

            pred=model(files)
            optimizer.zero_grad() #clearning old gradient from last step

            #loss computation
            loss = criterian(pred,labels)
            
          #backprop gradients at training time
            if t=='train':
                loss.backward()
                optimizer.step()

            r_loss += loss.item() #for all the images- updating on batches

        print(t +' epoch {}:loss {}  '.format(epoch,r_loss))

        if t=='val':
          val_loss_track.append(r_loss)
          if r_loss < best_loss:
            best_loss=r_loss
            best_model = model
        if  t=='train':
          tr_loss_track.append(r_loss)

  total_time =time.time()-start_time
  print('Time taken',total_time)
  return best_model

print('Training began....')
best_model = training()
# print('Best Model',best_model)
# print('tr_loss_track',tr_loss_track)
# print('val_loss_track',val_loss_track)

def predict(model):
    torch.cuda.empty_cache()
    model.train(False)
    model.eval()
    #For data augmentation
    transform ={'test':transforms.ToTensor()}
    datasets = {}
    datasets = dataloader.ImageDataSet(images_dir_test,fold='test',transformation=transform['test'])
    print('datasets',len(datasets))

    dataloaders = {}
    dataloaders = tdataloader(datasets, batch_size=batch_size, shuffle=True, num_workers=loader_works)
    print('dataloaders',len(dataloaders))
    print(dataloaders)
    print(enumerate(dataloaders))
    num_correct=0
    num_samples=0
    for  data in dataloaders:
        files, labels = data
        #load the data to GPU / CPU
        image_data = Variable(files.to(device))
        labels = Variable(labels.to(device))
        output = model(image_data)

        _,prediction =torch.max(output.data,1)
        num_correct += torch.sum(prediction == labels)
        num_samples += prediction.size(0)
    acc = float(num_correct) / num_samples
    print('acc',acc)
  
    # return true_df,pred_df

print('Testing began....')
predict(best_model)

print(tr_loss_track)
print(val_loss_track)

from matplotlib import pyplot as plt
plt.plot(tr_loss_track)
plt.plot(val_loss_track)
plt.show()

"""# New Section"""

# from google.colab import drive
# drive.mount('/content/drive')