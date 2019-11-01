import random
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

patch_size = {40:[300,320],35:[250,336],30:[240,300],25:[200,300],15:[150,240],10:[120,200]}

patch_size_new = {}

#SET THE FOLLOWING VARIABLES
folder_in = "../dataset/cars_train"

artifact_size = 15 #interms of percentage
folder_out = "../dataset/"+str(artifact_size)+"/cars_train/"
artifact_type = 0 #0 -random values for different pixels, 1 - contant value for all pixels, 2- contant value choosen randomly
artifact_color = 150

def readImage(filename):
    img = plt.imread(folder_in+"/"+filename)
    return img

def getRandomStartPoints(x_size,artifact_x,y_size,artifact_y):
    return random.randint(0,x_size-artifact_x-1),random.randint(0,y_size-artifact_y-1)

def randomArtifact(artifact_x,artifact_y,channels):
    return np.random.randint(0,255,size=(artifact_x,artifact_y,channels))

def constantArtifact(artifact_x,artifact_y,value,channels):
    if value>=0 and value<=255:
              return np.ones(shape=(artifact_x,artifact_y,channels))*value
    return np.ones(shape=(artifact_x,artifact_y,channels))*random.randint(0,255)

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


#x-100,y-100, xy -10000, target -2500, new_x = 100/2, new_y =100/2
#x-100,y-100, xy -10000, target -5000, new_x =
def createArtifacts(filename):
    #patch_size.get(artifact_size)[0],patch_size.get(artifact_size)[1]
    img1 = readImage(filename)
    img = img1.copy()

    #Get image shapes and no of pixels
    x_size = img.shape[0]
    y_size = img.shape[1]
    if len(img.shape)==3:
        z_axis = 3
    else:
        z_axis = 1
    total_pixels = x_size*y_size

    arti_arr = getPatchSize(x_size,y_size)
    artifact_x, artifact_y = arti_arr[0],arti_arr[1]

    print("Image:",img.shape," type:",type(img))

    print("arti x:",artifact_x," arti y:",artifact_y)
    #Get random start points for the artifact
    start_x, start_y = getRandomStartPoints(x_size,artifact_x,y_size,artifact_y)
    print("start x:",start_x," start y:",start_y)
    #Create an artifact using random values from 0-255
    if artifact_type==0:
        artifact = randomArtifact(artifact_x,artifact_y,z_axis)
    elif artifact_type==1:
        artifact = constantArtifact(artifact_x,artifact_y,artifact_color,z_axis)
    elif artifact_type==2:
        artifact = constantArtifact(artifact_x,artifact_y,-1,z_axis)
    print(artifact.shape)
    if z_axis>1:
        img[start_x:start_x+artifact_x,start_y:start_y+artifact_y,:] = artifact[:,:,:]
    else:
        img[start_x:start_x + artifact_x, start_y:start_y + artifact_y] = artifact[:, :,0]

    return img


def allFiles():
    fileNames = [f for f in listdir(folder_in) if isfile(join(folder_in, f))]
    return fileNames


def controller():
    filesList = allFiles()
    print(filesList)
    #return
    count =0
    for filename in filesList:
        print("fn:",filename)
        img = createArtifacts(filename)
        plt.imshow(img)
        plt.savefig( folder_out + filename)
        count=count+1

controller()