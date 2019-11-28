
###### Data storage heirachy for every occlusion type we have four folder of different artifacts (0,1,2,3) containing
# corresponding train and test images.
# We take one occlusion % at a time and one by one all artifacts types to train and run all models.

#Example 23% with 0 will have all models run on it
#23% 1 will have all models and so on

#Output
#23 % has four folder 0,1,2,3

#23->0-> /train, /test , /results and /plots
#/results--> checkpoints are saved in results with model_type and epoch i.e vgg19_3_model.pth for vgg_19 and epoch 3
#for resnet50 epoch 4  resnet50_4_model.pth
#/plots saved as model name_type of plot(3 currently for each model ; training loss-val loss, val-loss and val acc)
#Likewise all

#Maintainig a dict  called all_models for all the models to train with a dict of specs like optim and lr to be used correspondingly
# self.all_models = {        'vgg19':{'lr':0.01, 'optim':"sgd"},
#                            'resnet50':{'lr':0.0001, 'optim':"adam"},
#                            'resnet101':{'lr':0.0001, 'optim':"adam"},
#                            'resnet151':{'lr':0.0001, 'optim':"adam"},
#                            'googlenet':{'lr':0.0001, 'optim':"adam"},
#                            'unpretrained1':{'lr':0.01, 'optim':"adam"},
#                            'unpretrained2':{'lr':0.01, 'optim':"adam"},
#                            }
# #######

import DataLoader as dataloader
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import DataLoader as tdataloader
import copy
from matplotlib import pyplot as plt
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class  Controller: 
    def __init__(self,artifact,artifact_type):
        print(' Controller init ',artifact,artifact_type)
        # track loss and accuracies
        self.num_epochs = 2
        self.batch_size = 32
        self.batch_size_train = 64
        self.class_labels = 177  # Subject to change
        self.loader_works = 1  # multi-process data loading,#loader worker processes.
        self.artifact = artifact # occlusion %
        self.artifact_type = artifact_type #0,1,2,3

        ################ File paths ################

        self.images_dir = './dataset/'+str(artifact)+'/split/'+str(artifact_type)+'/train/'
        self.images_dir_test = './dataset/'+str(artifact)+'/split/'+str(artifact_type)+'/test/'
        self.store_dir = './dataset/'+str(artifact)+'/split/'+str(artifact_type)+'/'
        self.check_point_path = self.store_dir+'results/'
        self.plot = self.store_dir+'plot/'

        self.tr_loss_track, self.val_loss_track, self.tr_acc_track, self.val_acc_track, self.val_acc_history = [], [], [], [], []
        self.USE_GPU = True
        self.datasets = {}
        self.dataloaders = {}
        self.learning_rate = 0.0001


        ####Automation#######
        #dict of dict stating all models and their corresponding parama :lr and optim
        #Eg vgg19 will have lr=0.01 and optim as SGD
        self.all_models = {}

        #####################
        self.optim_type = " "
        self.model_type =" "
        # self.optim_type = "sgd"
        # self.model_type = "resnet101"
        # self.model_type = "vgg19"
        # self.model_type = "resnet50"


    def setDevice(self):
        if self.USE_GPU and torch.cuda.is_available():
                self.device = torch.device('cuda')
        else:
                self.device = torch.device('cpu')
        print('device running on:', self.device)


    def checkpoint(self,model, best_loss, epoch, learning_rate):
        print('$$$$$$$$Saving Checkpoint$$$$$$$$$')
        state = {'model': model, 'best_loss': best_loss, 'epoch': epoch, 'rng_state': torch.get_rng_state(),
             'LR': learning_rate}
        if not os.path.exists(self.check_point_path):
            os.makedirs(self.check_point_path)
        torch.save(state, self.check_point_path+self.model_type+'_'+str(epoch)+"_model.pth")

    def loadmodel(self,model_name):
        # load best model weights to return
        checkpoint_best = torch.load(self.check_point_path +model_name)
        model = checkpoint_best['model']
        return model

    def initialize_data(self):
        print('initialize_data',self.images_dir,' ',self.images_dir_test)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        transform = {'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),
        'test': transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])}
        # Loading the dataset with train and val as keys
        # Data divided in training and val using separate script
        datasets = {}
        datasets['train'] = dataloader.ImageDataSet(self.images_dir, fold='train', transformation=transform['train'])
        datasets['val'] = dataloader.ImageDataSet(self.images_dir, fold='val', transformation=transform['val'])
        datasets['test'] = dataloader.ImageDataSet(self.images_dir_test, fold='test', transformation=transform['test'])
        print('datasets train', len(datasets['train']))
        print('datasets val', len(datasets['val']))
        print('datasets test', len(datasets['test']))

        # Dataloader.util
        self.dataloaders['train'] = tdataloader(datasets['train'], batch_size=self.batch_size_train, shuffle=True, num_workers=self.loader_works)
        self.dataloaders['val'] = tdataloader(datasets['val'], batch_size=self.batch_size, shuffle=True, num_workers=self.loader_works)
        # print('dataloaders train ', len(self.dataloaders['train']))
        # print('dataloaders valida', len(self.dataloaders['val']))
        

        self.dataloaders['test'] = tdataloader(datasets['test'], batch_size=self.batch_size, shuffle=True, num_workers=self.loader_works)
        # print('dataloaders test', len(self.dataloaders['test']))
        print('---------------------------------------------')

        self.all_models = {'vgg19':{'lr':0.01, 'optim':"sgd"},
                           'resnet50':{'lr':0.0001, 'optim':"adam"},
                           'resnet101':{'lr':0.0001, 'optim':"adam"},
                           'resnet151':{'lr':0.0001, 'optim':"adam"},
                           'googlenet':{'lr':0.0001, 'optim':"adam"},
                           'un_vgg19':{'lr':0.01, 'optim':"sgd"},
                           'un_resnet50':{'lr':0.0001, 'optim':"adam"},
                           }

    def train_all_models(self):
        print('train_all_models')
        for model,params in self.all_models.items():
            print('@@@@@ MODEL TYPE ',model,params,'@@@@@')
            self.model_type = model
            self.optim_type = params['optim']
            self.learning_rate = params['lr']
            print('@@@@@ MODEL TYPE End @@@@@',self.model_type,self.optim_type,self.learning_rate)
            self.training()

    def training(self):
        self.tr_loss_track, self.val_loss_track, self.tr_acc_track, self.val_acc_track, self.val_acc_history = [], [], [], [], []

        print('training',self.model_type,self.optim_type,self.learning_rate)
        # Defining pretrained model
        if self.model_type=="vgg19":
            model = models.vgg19(pretrained=True)
            model.classifier[6] = nn.Linear(4096,self.class_labels)
        elif self.model_type=="resnet50":
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.class_labels)
        elif self.model_type =="resnet101":
            model = models.resnet101(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.class_labels)
        elif self.model_type =="resnet151":
            model = models.resnet101(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.class_labels)
        elif self.model_type =="googlenet":
            model = models.googlenet(pretrained=True)
        elif self.model_type =="densenet121":
            model = models.densenet121(pretrained=True)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, self.class_labels)
        # #Without fine tuning
        elif self.model_type == "un_vgg19":
            # print("unpretrained1")
            model = models.vgg19(pretrained=False)
            model.classifier[6] = nn.Linear(4096,self.class_labels)
           # set pretrained =False for models
        elif self.model_type == "un_resnet50":
            # print("unpretrained2")
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.class_labels)

        model = model.to(self.device)
        criterian = nn.CrossEntropyLoss()

        if self.optim_type=="adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optim_type=="sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)

        # training
        start_time = time.time()
        best_loss = float('inf')
        best_model = None
        best_acc = 0.0
        # best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(1, self.num_epochs + 1):
            print('epoch {}/{}'.format(epoch, self.num_epochs))
            for t in ['train', 'val']:
                    num_correct = 0.0
                    num_samples = 0
                    r_loss = 0.0
                    running_corrects = 0

                    if t == 'train':
                        # training mode
                        model.train()
                    else:
                        # evaluate model
                        model.eval()

                    count = 0
                    for data in self.dataloaders[t]:
                        count += 1
                        # data has three types files, labels and filename
                        files, labels, filename = data

                        files = Variable(files.to(self.device))  # to gpu or cpu
                        labels = Variable(labels.to(self.device))

                        optimizer.zero_grad()  # clearning old gradient from last step

                        with torch.set_grad_enabled(t == 'train'):
                            pred = model(files)
                            # loss computation
                            loss = criterian(pred, labels)

                            _, prediction = torch.max(pred, 1)

                            # backprop gradients at training time
                            if t == 'train':
                                loss.backward()
                                optimizer.step()

                            # print(t +' iteration {}:loss {}  '.format(count,r_loss))
                        # statistics
                        r_loss += loss.item() * files.size(0)
                        print(t + ' iteration {}:loss {}  '.format(count, r_loss))
                        running_corrects += torch.sum(prediction == labels.data)
                    epoch_loss = r_loss / len(self.dataloaders[t].dataset)
                    epoch_acc = running_corrects.double() / len(self.dataloaders[t].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(t, epoch_loss, epoch_acc))

                    # print(t +' epoch {}:loss {}  '.format(epoch,r_loss))

                    # deep copy the model
                    # print('epoch_acc',epoch_acc,'best_acc',best_acc)
                    if t == 'val' and epoch_acc > best_acc:
                        print('inside check point if')
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        self.checkpoint(best_model_wts, best_loss, epoch, self.learning_rate)
                    if t == 'val':
                        self.val_acc_history.append(epoch_acc.item())
                        self.val_loss_track.append(epoch_loss)

                    if t == 'train':
                        self.tr_loss_track.append(epoch_loss)
                        self.tr_acc_track.append(epoch_acc.item())

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        # model.load_state_dict(best_model_wts)
        # updating best model in checkpoint

        self.plot_losses_both(self.tr_loss_track,self.val_loss_track)
        self.plot_loss_Val(self.val_loss_track)
        self.plot_loss_Accu(self.val_acc_history)
        # return model

    def predict(self,model):
        torch.cuda.empty_cache()

        num_correct = 0
        num_samples = 0
        count = 0
        for data in self.dataloaders['test']:
            count += 1
            files, labels, filename = data
            # load the data to GPU / CPU
            image_data = Variable(files.to(self.device))
            labels = Variable(labels.to(self.device))
            output = model(image_data)
            # print('output', output)
            _, prediction = torch.max(output.data, 1)
            # print('filename',filename,'\nlabels',labels,'\nprediction',prediction)
            num_correct += torch.sum(prediction == labels)
            num_samples += prediction.size(0)
            print('Iteration', count)
        acc = float(num_correct) / num_samples
        print('Accuracy of test data :', acc)


    def plot_losses_both(self,loss_track1,loss_track2,track1_label="Training Loss",track2_label="Validation loss"):
        ax = plt.subplot(111)
        plt.plot(loss_track1, c='b', label=track1_label)
        plt.plot(loss_track2, c='g', label=track2_label)
        # plt.title(plt_label)
        ax.legend()
        if not os.path.exists(self.plot):
            os.makedirs(self.plot)
        plt.savefig(self.plot+self.model_type+"train_val.png")
        plt.close()



    def plot_loss_Accu(self,val_acc,xlab="Num epochs",ylab="Validation accuracy"):
        plt.plot(val_acc, c='b')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        # plt.title(plt_label)
        if not os.path.exists(self.plot):
            os.makedirs(self.plot)
        plt.savefig(self.plot+self.model_type+"_val_acc.png")
        plt.close()

    def plot_loss_Val(self,val_loss,xlab="Num epochs",ylab="Validation Loss"):
        plt.plot(val_loss, c='b')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        # plt.title(plt_label)
        if not os.path.exists(self.plot):
            os.makedirs(self.plot)
        plt.savefig(self.plot+self.model_type+"_val_loss.png")
        plt.close()


# cont = Controller()
# cont.setDevice()
# cont.initialize_data()

print('Training began....')
#start training for all the data types
#one data type will be trained on all the models
def training_phase():
    print('training phase')
    #possible data combinations
    all_data ={'13':[0,1,2,3],'23':[0,1,2,3],'30':[0,1,2,3],'32':[0,1,2,3]}
    # all_data ={'23':[0,1]}
    for key,value in all_data.items():
        artifact =key
        for artifact_type in value:
            print('########## DATA TYPE #############')
            print('artifact',artifact,'artifact',artifact_type)
            cont = Controller(artifact,artifact_type)
            cont.setDevice()
            cont.initialize_data()
            cont.train_all_models() #for this data type train all models
            print('#####################################')

training_phase()

# print('Testing began....')
