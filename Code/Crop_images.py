#Crop images as per the boundind boxes
#input dump file
# src intact folder of 16k images which is cropped to bounding box and resized
# output cropped 16k images need to use stanford_split to segregate into train and rest folders
# cropped image

#referered to stanford paper idea to crop images as per bounding boxes
from os import listdir
from os.path import isfile, join
import csv
import cv2
import os


######CHange these variables###############
entire_csv_dump='./dataset/dump.csv'
artifact = -1

src = './dataset/car_ims/'
dst = './dataset/cropped/cropped_car_ims/'

img_height = 224
img_width = 224

#getting bounding boxes for all the images from the entire data dump.csv

def getBoundingBox():
	bbox = {}
	with open(entire_csv_dump, "r") as f:
			reader = csv.reader(f, delimiter=',')
			next(reader)
			for row in reader:
				filepath = row[0] #get file
				label = row[5] #get label
				# get bounding box coordinates
				x1 = int(row[1])
				y1 = int(row[2])
				x2 = int(row[3])
				y2 = int(row[4])
				bb_box = (x1,y1,x2,y2)
				# print('get_bounding_box',bb_box)
				file_name = filepath.split('/')[1]
				bbox[file_name] = bb_box
				# break


	return bbox #file name with corresponding bb_box



def cropImages(src_folder_path,dst_folder_path,img_height,img_width):

	#get images from relevant folder
	#get bounding boxes of corresponding images
	#crop the images and move to respective folders
	# the cropping technique is inspired from stanford paper cited
	images_to_crop = []
	for f in listdir(src_folder_path):
		if isfile(join(src_folder_path,f)): #if file
			images_to_crop.append(f) #add

	# print(type(images_to_crop))
	# print('All images',images_to_crop)
	print('Total images to crop: ',len(images_to_crop))

	num_images = len(images_to_crop)

	bb_box = getBoundingBox() #get bounding box corresponding to all images

	for i in images_to_crop:
		# print(i)
		if('.jpg' in i):
			img = cv2.imread(src_folder_path+i,cv2.IMREAD_UNCHANGED)
			h = img.shape[0]
			w = img.shape[1]

		# as per the paper
		if(i in bb_box ): # if file exists
			# print(type(bb_box[i]))
			(x1, y1, x2, y2) = bb_box[i]
			# print('inside crop',x1, y1, x2, y2)
			margin = 16 #given as per suggested in paper to maintain some context after cropping
			x1 = max(0, x1 - margin)
			y1 = max(0, y1 - margin)
			x2 = min(x2 + margin, w)
			y2 = min(y2 + margin, h)

			dst_path = os.path.join(dst_folder_path)
			if not os.path.exists(dst_path):
				os.makedirs(dst_path)
			dst_path = os.path.join(dst_path, i)

			crop_image = img[y1:y2, x1:x2]
			print(i,crop_image.shape)
			dst_img = cv2.resize(src=crop_image, dsize=(img_height, img_width))
			cv2.imwrite(dst_path, dst_img)


cropImages(src,dst,img_height,img_width)
