import random
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.io
from scipy.misc import imresize
from PIL import Image

patch_size = {30:[166,165],25:[130,127],20:[115,113],15:[100,99],10:[80,81]}

patch_size_new = {}
bounding_box = {}
#SET THE FOLLOWING VARIABLES
folder_in = "../dataset/car_ims"
artifact_size = 15 #in terms of percentage between 30 and 10 increments of 5
folder_out = "../dataset/"+str(artifact_size)+"/cars_ims/"
artifact_type = 3 #0 -random values for different pixels, 1 - contant value for all pixels, 2- contant value choosen randomly, 3 - image artifacts
artifact_color = 150
image_artifact_images_dir = "../dataset/image_artifact_processed/"
image_artifact_input_dir = "../dataset/image_artifact/"
bb_file = "../dataset/cars_annos.mat"
artifact_images =[]
artifact_images_count=0
############################

def initBoundingBoxes(filename):
    car_annos = scipy.io.loadmat(filename)  # we can do with entire file instead of devkit

    # 196 classes
    # From devkit
    # Contains the variable 'annotations', which is a struct array of length
    # num_images and where each element has the fields:
    #   bbox_x1: Min x-value of the bounding box, in pixels
    #   bbox_x2: Max x-value of the bounding box, in pixels
    #   bbox_y1: Min y-value of the bounding box, in pixels
    #   bbox_y2: Max y-value of the bounding box, in pixels
    #   class: Integral id of the class the image belongs to.
    #   fname: Filename of the image within the folder of images.

    # Extracting details for each image

    # headings =[['bbox_x1','bbox_y1','bbox_x2','bbox_y2','class','fname']] # For devkit train

    headings = [['relative_im_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'test']]
    entry = []  # all the image data with each row containing the entire detail as mentioned in headings
    for i in car_annos['annotations'][0]:
        key = []
        for j in i:
            key.append(j.flat[0])
        entry.append(key)
        file_id = key[0].split("/")[1]
        file_id = file_id.lstrip("0")
        #print("key:", file_id)
        bounding_box[file_id] = (key[2],key[1],key[4],key[3])
#This function takes any artifact image and converts it into 256*256 image size and saves it to another folder set in image_artifact_images_dir
def preprocessImageArtifacts():
    i =0
    files_list = allFiles(image_artifact_input_dir)
    for file in files_list:
        #print(file)
        img = readImage(image_artifact_input_dir+file)
        re_img = reshapeImage(img)
        img = Image.fromarray(re_img)
        img.save(image_artifact_images_dir + str(i)+ ".jpg")
        i=i+1
def loadArtifactsImages():
    global artifact_images_count
    files_list = allFiles(image_artifact_images_dir)
    for file in files_list:
        img = readImage(image_artifact_images_dir+file)
        artifact_images.append(img)
        artifact_images_count=artifact_images_count+1

def readImage(filename):
    img = plt.imread(filename)
    return img

def getRandomStartPoints(x_size,artifact_x,y_size,artifact_y):
    return random.randint(0,x_size-artifact_x-1),random.randint(0,y_size-artifact_y-1)

def randomArtifact(artifact_x,artifact_y,channels):
    return np.random.randint(0,255,size=(artifact_x,artifact_y,channels))

def constantArtifact(artifact_x,artifact_y,value,channels):
    if value>=0 and value<=255:
              return np.ones(shape=(artifact_x,artifact_y,channels))*value
    return np.ones(shape=(artifact_x,artifact_y,channels))*random.randint(0,255)
def imageArtifact(artifact_x,artifact_y,channels):
    global artifact_images_count
    select_id = random.randint(0,artifact_images_count-1)
    #print("select id:",select_id," artifact cnt:",artifact_images_count)
    if channels==1:
        return artifact_images[select_id][0:artifact_x,0:artifact_y,0]
    return artifact_images[select_id][0:artifact_x,0:artifact_y,:]

def getPatchSize(x_size,y_size):
    if x_size in patch_size_new:
        patch_size_y =patch_size_new[x_size]
        if y_size in patch_size_y[y_size]:
            return patch_size_new[x_size][y_size]

    target = int((x_size * y_size)/100)*artifact_size
    x = x_size
    y =y_size

    x_inc = int(x_size/y_size)
    y_inc = int(y_size/x_size)
    if x_inc==0:
         x_inc = 1
    if y_inc ==0:
         y_inc =1
    #print("target:",target)
    curr = x*y
    while curr > target:
         #print("curr:",x*y)
         x=x-x_inc
         y=y-y_inc
         curr = x*y
    #print("curr:", x * y)
    patch_size_new[x_size,y_size] = [x,y]
    return [x,y]

def getFixedPatchSize():
    return patch_size[artifact_size]

def createArtifacts(img):
    #Get image shapes and no of pixels
    x_size = img.shape[0]
    y_size = img.shape[1]
    if len(img.shape)==3:
        z_axis = 3
    else:
        z_axis = 1
    total_pixels = x_size*y_size

    arti_arr = getFixedPatchSize()
    artifact_x, artifact_y = arti_arr[0],arti_arr[1]


    #Get random start points for the artifact
    start_x, start_y = getRandomStartPoints(x_size,artifact_x,y_size,artifact_y)

    #Create an artifact using random values from 0-255
    if artifact_type==0:
        artifact = randomArtifact(artifact_x,artifact_y,z_axis)
    elif artifact_type==1:
        artifact = constantArtifact(artifact_x,artifact_y,artifact_color,z_axis)
    elif artifact_type==2:
        artifact = constantArtifact(artifact_x,artifact_y,-1,z_axis)
    elif artifact_type==3:
        artifact = imageArtifact(artifact_x,artifact_y,z_axis)
    #print("artifact:",artifact.shape," ",img[start_x:start_x+artifact_x,start_y:start_y+artifact_y,:].shape )
    #print(artifact.shape)
    if z_axis>1:
        img[start_x:start_x+artifact_x,start_y:start_y+artifact_y,:] = artifact[:,:,:]
    else:
        img[start_x:start_x + artifact_x, start_y:start_y + artifact_y] = artifact[:, :,0]

    return img


def allFiles(folder_in):
    fileNames = [f for f in listdir(folder_in) if isfile(join(folder_in, f))]
    return fileNames

def checkBoundingBoxSize(img,filename):
    x,y,x1,y1 = bounding_box[filename.lstrip("0")]
    if x1>img.shape[0] or y1>img.shape[1]:
        print("bb mismatch:",filename," expected:",x," ",y," ",x1," ",y1," img:",img.shape)
        return True
    if abs(x1-x)>=245 and abs(y1-y)>=245:
        return False
    print("bb size small:", filename)
    return True

def getBoundingBox(img,filename):
    x, y, x1, y1 = bounding_box[filename.lstrip("0")]
    pad_percent =7
    pad_val_x = int(pad_percent*((x1-x)/100))
    pad_val_y = int(pad_percent *((y1-y) / 100))
    #print("x:",x," y:",y," x1:",x1," y1:",y1," pad_x:",pad_val_x," pad_y:",pad_val_y)
    lx = x - pad_val_x
    rx = x1 + pad_val_x
    ty = y - pad_val_y
    by = y1 + pad_val_y
    if lx<0:
        lx =0
    if  ty<0:
        ty=0
    if rx >img.shape[0]:
        rx =img.shape[0]
    if by > img.shape[1]:
        by = img.shape[1]
    return img[lx:rx,ty:by]
def reshapeImage(img):
    #print("size:",img.shape)
    return imresize(img,(256,256))

def controller():
    filesList = allFiles(folder_in)

    print(filesList)
    filesList.sort()
    #return
    count =0
    for filename in filesList:
        #print("File name:",filename)
        #if checkBoundingBoxSize(filename):
        #    continue
        img = readImage(folder_in+"/"+filename).copy()
        if count%100==0:
            print("Count:",count)
        #check if the bounding box size is within the image size and check check if its more than 256*256 pixels
        if checkBoundingBoxSize(img,filename):
            continue

        #plt.imshow(img)
        #plt.show()

        #print("file name strip: ",filename.lstrip("0"))
        #print("img shape:",img.shape)
        #get the bounding box pixels image from the original image
        bb_img = getBoundingBox(img,filename)
        #print("bb img shape:", bb_img.shape)

        #reshape to 256*256
        re_img =reshapeImage(bb_img)
        print("shape:",re_img.shape)
        arti_img = createArtifacts(re_img)
        #print("arti shape:", arti_img.shape)
        img = Image.fromarray(arti_img)
        img.save(folder_out+filename)
        #plt.imshow(arti_img)
        #plt.show()
        #plt.savefig( folder_out + filename)
        count=count+1
        del arti_img
        del bb_img
        del re_img
        del img
#preprocessImageArtifacts()
loadArtifactsImages()
initBoundingBoxes(bb_file)
controller()
