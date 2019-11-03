import scipy.io
import csv

path = '/Users/nikita/Documents/pythonScripts/cs682Project'
car_annos = scipy.io.loadmat(path+'/cars_annos.mat') #we can do with entire file instead of devkit

print(car_annos.keys())

print('annos',type(car_annos))

print('car_annos["annotations"]',len(car_annos["annotations"][0]))

print('car_annos["annotations"]',car_annos["annotations"][0].dtype)

print('car_annos["annotations"]',car_annos["annotations"][0][0]["bbox_x1"],car_annos["annotations"][0][0]["bbox_x2"], car_annos["annotations"][0][0]["relative_im_path"])
print('car_annos["annotations"]',car_annos["annotations"][0][0]["bbox_y1"],car_annos["annotations"][0][0]["bbox_y2"], car_annos["annotations"][0][0]["relative_im_path"])


#196 classes
  #From devkit
  # Contains the variable 'annotations', which is a struct array of length
  # num_images and where each element has the fields:
  #   bbox_x1: Min x-value of the bounding box, in pixels
  #   bbox_x2: Max x-value of the bounding box, in pixels
  #   bbox_y1: Min y-value of the bounding box, in pixels
  #   bbox_y2: Max y-value of the bounding box, in pixels
  #   class: Integral id of the class the image belongs to.
  #   fname: Filename of the image within the folder of images.

#Extracting details for each image

# headings =[['bbox_x1','bbox_y1','bbox_x2','bbox_y2','class','fname']] # For devkit train

headings =[['relative_im_path','bbox_x1','bbox_y1','bbox_x2','bbox_y2','class','test']] #For all
entry=[] # all the image data with each row containing the entire detail as mentioned in headings
for i in car_annos['annotations'][0]:
    # print(i)
    key=[]
    for j in i:
        key.append(j.flat[0])
    entry.append(key)

print(len(entry))
print(entry[0])

# Writing to csv
with open(path+"/cars_annos.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(headings)
    writer.writerows(entry)
