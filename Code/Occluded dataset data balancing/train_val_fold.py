import csv
from math import ceil
#To be considered
#The files divides the training data into train-val 80% train and 20% val
#also does class mapping starting from 0  and writes to mapping file

# Takes train, test and dump as input
#train- val split with mappeing into file train_mapped
#test file after mapping into test_mapped
#mapped - to map old and new labels just for us to keep track


artifact = 10

csv_train_path='./dataset/'+str(artifact)+'/train.csv' #read from this
csv_train_path_fold='./dataset/'+str(artifact)+'/train_mapped.csv' #write to this
csv_test_path = './dataset/'+str(artifact)+'/test.csv' # read from this
csv_test_mapped='./dataset/'+str(artifact)+'/test_mapped.csv' #write to this
csv_mapped = './dataset/'+str(artifact)+'/mapped.csv' # label mapping track


def train_val_split_map():
	file_label ={} #count
	counter=0
	#get all the files in the particular label
	with open(csv_train_path, "r") as f:
		reader = csv.reader(f, delimiter=',')
		next(reader) #skips header row
		for row in reader:
			counter+=1
			label = row[1] #get label
			file_name = row[0] #get file
			if(label in file_label):
				file_label[label].append(file_name)
			else:
				file_label[label] = [file_name]

	#get how many number of images are suitable for validation data
	num_train = counter #total number of training images
	print('num_train',num_train)
	percent_val = 0.2 #80% train and 20 %val
	count =int(num_train*percent_val) #get number that is x% of entire training data those many val
	#get total number of classes
	num_classes =len(file_label.keys())
	print('num_classes',num_classes)
	#equally get val data from all the classes
	num_val_each_class = ceil(count/num_classes) #taking these many number of images from each class for validation
	print('num_val_each_class',num_val_each_class)

	val_={}
	train_={}
	val_files_temp=[]
	val_files=[]
	mapping={}
	countt=0
	for key,value in file_label.items():
		#for mapping each class given to incremmental counter
		mapping[key] =countt
		countt+=1
		val_[key] = file_label[key][0:num_val_each_class] # first num_val_each_class to val test data
		val_files_temp.append(val_[key])
		train_[key] =file_label[key][num_val_each_class:] # rest to train to evenly distribute

	print('mapping',mapping)
	print('mapping len',len(mapping))

	for i in val_files_temp:
		for j in i:
			val_files.append(j) #all the val files for all the classes from val_files_temp dict

	print('val_files',len(val_files))

	# print(val_files)
	# print('file_label',file_label)
	# print('val',val_)
	# print('train',train_)

	#getting the files folds as train and val
	entry=[]
	with open(csv_train_path, "r") as f:
		reader = csv.reader(f, delimiter=',')
		next(reader)
		for row in reader:
			ls=[]
			file_name = row[0]
			label = row[1]
			if(file_name in val_files): #if file name in val set fold in file to be val else train
				fold='val'
			else:
				fold='train'
			ls.append(file_name)
			# ls.append(label)
			ls.append(mapping[label]) #instead of old class label add a new label starting from 0 in incremental way
			ls.append(fold)
			entry.append(ls)

	#writing th file with mapped labels
	header = [['files','labels','fold']]
	with open(csv_train_path_fold, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(header)
		writer.writerows(entry)


	#chaging the test file mapping
	#train
	entry=[]
	with open(csv_test_path, "r") as f:
		reader = csv.reader(f, delimiter=',')
		next(reader)
		for row in reader:
			ls=[]
			file_name = row[0]
			label = row[1]
			ls.append(file_name)
			# ls.append(label)
			ls.append(mapping[label]) #instead of old class label add a new label starting from 0 in incremental way
			entry.append(ls)

	#writing th file with mapped labels
	#test
	header = [['files','labels']]
	with open(csv_test_mapped, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(header)
		writer.writerows(entry)


	#writing mapping file to track
	#writing to  train csv
	#mapping
	header = [['old_label','new_label']]
	with open(csv_mapped, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(header)
		for key, val in mapping.items():
			writer.writerow([key, val])

train_val_split_map()
