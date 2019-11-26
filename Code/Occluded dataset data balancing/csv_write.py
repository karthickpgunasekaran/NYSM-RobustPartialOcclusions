#once the images are in respective train and test folders this script is used to generate csv for train and test

# you will have to call write_csv function  once with train folder and train csv
# and once again with test images folder and test csv

#Input - train adnd test folder
#Output write to train adn test csvs
#funtion input(folder name, file name)

from os import listdir
from os.path import isfile, join
import csv



##### Chnage these paths#############
artifact = 10

path_org = './dataset/'+str(artifact)+'/'
train_path1='./dataset/'+str(artifact)+'/train/'
csv_path1='./dataset/'+str(artifact)+'/train.csv'


test_path = './dataset/'+str(artifact)+'/test/'
csv_test_path='./dataset/'+str(artifact)+'/test.csv'

entire_csv_dump='./dataset/dump.csv'


def write_csv(folder_path,file_path):
	# if(flag=='train'):
	# 	train_path = train_path1
	# else:
	# 	train_path = test_path
	train_path = folder_path
	org_images = [] #all images in training/test folder
	for f in listdir(train_path):
		if isfile(join(train_path,f)): #if file
			org_images.append(f) #add
	print(type(org_images))
	print(org_images)

	file_label={}
	entry=[]
	#for these images write all in train.csv/test.csv
	with open(entire_csv_dump, "r") as f:
		reader = csv.reader(f, delimiter=',')
		next(reader)
		for row in reader:
			ls=[]
			filepath = row[0] #get file
			label = row[5] #get label
			file_name = filepath.split('/')[1]
			if(file_name in org_images): # only add if file is in preprocessed
				if(label in file_label):
					file_label[label].append(file_name)
				else:
					file_label[label] = [file_name]
				ls.append(file_name)
				ls.append(label)
				# ls.append(label)
				entry.append(ls)

	csv_path = file_path
	header = [['files','labels','fold']]
	with open(csv_path, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(header)
		writer.writerows(entry)

# write_csv(train_path1,csv_path1)

write_csv(test_path,csv_test_path)
