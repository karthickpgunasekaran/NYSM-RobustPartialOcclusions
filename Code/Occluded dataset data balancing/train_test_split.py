#This script divides the images(at /cars_ims) into train and test folders(create folders with names in directory)
#dump is the dump os entire images obtained from .mat file

#train test split of occluded images - based on magic number and 70%-30% split of train and test
#input all images folder divides in train and test

from os import listdir
from os.path import isfile, join
import csv
from statistics import mean
from math import ceil
import cv2


#####Please change these################

artifact = 10
#Change these
path_org = './dataset/'+str(artifact)+'/cars_ims/'
print(path_org)
train_path='./dataset/'+str(artifact)+'/train/'
test_path = './dataset/'+str(artifact)+'/test/'
entire_csv_dump='./dataset/dump.csv'

class_label_count = {}
images = []
file_label={}
occluded_images=[]

#Retrienve all the occluded images- filenames from the folder
for f in listdir(path_org):
    if isfile(join(path_org,f)): #if file
        occluded_images.append(f) #add

print(type(occluded_images))
print('All occluded images',occluded_images)
print(len(occluded_images))

#From the complete csv dump of 16k images-- look for the ones in folder and store in dict with class label as key->
#Gives list of file names correpsonding to images

with open(entire_csv_dump, "r") as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        filepath = row[0] #get file
        label = row[5] #get label
        file_name = filepath.split('/')[1]
        if(file_name in occluded_images): # only add if file is in preprocessed
            if(label in file_label):
                file_label[label].append(file_name)
            else:
                file_label[label] = [file_name]

print('Class- files',file_label)

#How many images per class - stored in dict
class_image_count ={}
for key,value in file_label.items():
    class_image_count[key]=len(value)

print('class_image_count',class_image_count)

# print('min number of images',min(class_image_count.values()),'\nmax number os images',max(class_image_count.values()),'\navg',int(mean(class_image_count.values())))

avg = int(mean(class_image_count.values()))
#get all the classes with labels +- magic number than average
filtered_dict = {}
index_train ={}
index_test = {}

magic_number =10 #How much deviation from the average

for key,value in class_image_count.items():
    if(class_image_count[key]>=avg-magic_number): # minum number of images in class
        if(class_image_count[key]>avg+magic_number):  #cap it to avg+ magic number to take that class into accoount- this is done to avoid ignoring classes that had more number of images than avg-magic_number
            #cap to average+magic_number
            filtered_dict[key] = avg+magic_number # retrieve all the labels
        else: #if lies between avg-magic_number  and avg+magic_number
            filtered_dict[key] = class_image_count[key] # retrieve all the labels
        index_train[key] = ceil(0.7*filtered_dict[key]) # 70% of images per class to train rest to test
    else:
        continue

print('filtered_dict',filtered_dict)
print('filtered_dict',len(filtered_dict),'keys',len(filtered_dict.keys()))

print('index_train',index_train)
print('index_train',len(index_train),'keys',len(index_train.keys()))

training_images = {}
test_images = {}
#Segregating the training and the test data
for key,value in index_train.items():
    training_images[key] = file_label[key][:index_train[key]]
    test_images[key] = file_label[key][index_train[key]:filtered_dict[key]] #upto capped value if number too high

print('training_images',len(training_images))
print(training_images)

print('test_images',len(test_images),'num images')
print(test_images)



#Now write trainign and test images in separate folders

for key,values in training_images.items():
     for image in training_images[key]:
         img = cv2.imread(path_org+'/'+image,cv2.IMREAD_UNCHANGED)
         cv2.imwrite(train_path+image,img)

for key,values in test_images.items():
    for image in test_images[key]:
         img = cv2.imread(path_org+'/'+image,cv2.IMREAD_UNCHANGED)
         cv2.imwrite(test_path+image,img)






