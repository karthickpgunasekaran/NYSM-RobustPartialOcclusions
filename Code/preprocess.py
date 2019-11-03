import cv2
from os import listdir
from os.path import isfile, join

path = '/Users/nikita/Documents/pythonScripts/cs682Project/'

new_dims = (256, 256) # standard what they expect for pretrained

path_org = path+'Convert/'
path_resized = path+'Resized/'

org_images = []

for f in listdir(path_org):
    if isfile(join(path_org,f)): #if file
        org_images.append(f) #add

print(type(org_images))
print(org_images)


#REading file from original folder and resizing
for i in org_images:
    if('.jpg' in i):
        img = cv2.imread(path_org+i,cv2.IMREAD_UNCHANGED)
        # print(i+':  Original Dimensions : ',img.shape)
        h,w,_ = img.shape
        if(h>150 and w >150): #resize only if height and width >150
            resized = cv2.resize(img, new_dims, interpolation = cv2.INTER_AREA)
            # print('Resized Dimensions : ',resized.shape)
            cv2.imwrite(path_resized+i,resized)

