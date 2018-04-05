import sys

print (sys.version)


import tensorflow as tf; 

print(tf.__version__)


import math
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle as pk
#import cv2
import random
import os

#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

#mydevice = '/gpu:0'

path = 'Training/'
img_w = 160
img_h = 160
n_labels = 2 #6
n_of_images= 191*2
data_shape = img_w*img_h



#import matplotlib.pyplot as plt
import tensorlayer as tl


NoLabel = [0,0,0]

Water = [255,100,100] 

Sky = [153,17,105] 

Mountain = [0,255,0] 

Boat = [200,8,21] 

Other = [255,255,255] 

label_colours = np.array([NoLabel, Water, Sky, Mountain, Boat,
                          Other])

def visualize(temp, plot=False):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,5):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    rgb[:,:,0] = (b)#[:,:,0]
    rgb[:,:,1] = (g)#[:,:,1]
    rgb[:,:,2] = (r)#[:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb
    
def normalized(bgr):
    return bgr/255.0
    #norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    #b=rgb[:,:,0]
    #g=rgb[:,:,1]
    #r=rgb[:,:,2]

    #norm[:,:,0]=cv2.equalizeHist(b)
    #norm[:,:,1]=cv2.equalizeHist(g)
    #norm[:,:,2]=cv2.equalizeHist(r)

    #return norm

    ##b, g, r = cv2.split(bgr)
    ##red = cv2.equalizeHist(r)
    ##green = cv2.equalizeHist(g)
    ##blue = cv2.equalizeHist(b)
    ##return cv2.merge((blue, green, red))

def binarylab(labels):
    x = np.zeros([img_h,img_w,n_labels])  
    
    for i in range(img_h):
        for j in range(img_w):
            x[i,j,labels[i][j]]=1
    return x

"""def prep_data(path, augmentation1, augmentation2):
    train_data = []
    train_label = []
    train_data_path = []
    train_label_path = []
    import os
    with open(path+'data.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        image= cv2.imread(os.getcwd() + "/" + path + txt[i][0])
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h,w,d=image.shape
        annotation=cv2.imread(os.getcwd() + "/" + path + txt[i][1])
        h_a,w_a,d_a=annotation.shape
        if(augmentation1==True):
            number_of_crop=int(w/img_w)
            for j in range(number_of_crop):
                Angle=random.randrange(-10,10)
                M = cv2.getRotationMatrix2D((h/2,w/2),Angle,1)
                image_crop = image[int((h/2)-(img_h/2)):int((h/2)+(img_h/2)),int((j*img_w)):int((j+1)*img_w)]
                annotation_crop = annotation[int((h/2)-(img_h/2)):int((h/2)+(img_h/2)),int((j*img_w)):int((j+1)*img_w)]
                #if(j>1 and j<number_of_crop-1):
                #    image_crop = cv2.warpAffine(image_crop,M,(img_w,img_h))
                #    annotation_crop = cv2.warpAffine(annotation_crop,M,(img_w,img_h))
                train_data.append(normalized(image_crop))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
                train_label.append(binarylab(annotation_crop[:,:,0]))
        #Angle=random.randrange(-10,10)
        #M = cv2.getRotationMatrix2D((w/2,h/2),Angle,1)
        #image_rotated = cv2.warpAffine(image,M,(w,h))
        #image_rotated = image_rotated[int((h/2)-(h/2.6)):int((h/2)+(h/2.6)), int((w/2)-(w/2.6)):int((w/2)+(w/2.6))]
        #image_rotated = cv2.resize(image_rotated,(img_w,img_h))
        #annotation_rotated = cv2.warpAffine(annotation,M,(w_a,h_a))
        #annotation_rotated = annotation_rotated[int((h/2)-(h/2.6)):int((h/2)+(h/2.6)), int((w/2)-(w/2.6)):int((w/2)+(w/2.6))]
        #annotation_rotated = cv2.resize(annotation_rotated,(img_w,img_h))
        
        if(augmentation2==True):
            image_flip=cv2.flip(image,1)
            annotation_flip=cv2.flip(annotation,1)
            image_flip=cv2.resize(image_flip,(img_w,img_h))
            annotation_flip=cv2.resize(annotation_flip,(img_w,img_h))
            train_data.append(normalized(image_flip))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
            train_label.append(binarylab(annotation_flip[:,:,0]))
            
            image_rotated,annotation_rotated= tl.prepro.rotation_multi([image, annotation],rg=20, is_random=True, fill_mode='reflect') 
            image_rotated=cv2.resize(image_rotated,(img_w,img_h))
            annotation_rotated=cv2.resize(annotation_rotated,(img_w,img_h))
            train_data.append(normalized(image_rotated))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
            train_label.append(binarylab(annotation_rotated[:,:,0]))
            
            image_shear,annotation_shear= tl.prepro.shear_multi([image, annotation], 0.05, is_random=True, fill_mode='reflect')
            image_shear=cv2.resize(image_shear,(img_w,img_h))
            annotation_shear=cv2.resize(annotation_shear,(img_w,img_h))
            train_data.append(normalized(image_shear))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
            train_label.append(binarylab(annotation_shear[:,:,0]))
            
            image_zoom,annotation_zoom=tl.prepro.zoom_multi([image, annotation],zoom_range=[0.5, 0.7], is_random=True, fill_mode='reflect')
            image_zoom=cv2.resize(image_zoom,(img_w,img_h))
            annotation_zoom=cv2.resize(annotation_zoom,(img_w,img_h))
            train_data.append(normalized(image_zoom))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
            train_label.append(binarylab(annotation_zoom[:,:,0]))
            
            image_Bright=tl.prepro.brightness(image, gamma=1, gain=1, is_random=True)
            image_Bright=cv2.resize(image_Bright,(img_w,img_h))
            annotation_Bright=cv2.resize(annotation,(img_w,img_h))
            train_data.append(normalized(image_Bright))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
            train_label.append(binarylab(annotation_Bright[:,:,0]))
            #randomRoulette=random.randrange(0,20)
            #if(randomRoulette==10):
            #    image_flipUpDown=cv2.flip(image,0)
            #    annotation_flipUpDown=cv2.flip(annotation,0)
            #    train_data.append(normalized(image_flipUpDown))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
            #    train_label.append(binarylab(annotation_flipUpDown[:,:,0]))
                
            #randomRoulette=random.randrange(0,5)
            #if(randomRoulette==3):
            #    image_blur=cv2.blur(image,(10,10))
            #    annotation_blur=cv2.blur(annotation,(10,10))
            #    train_data.append(normalized(image_blur))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
            #    train_label.append(binarylab(annotation_blur[:,:,0]))
                
                
                
            
        
        image= cv2.resize(image,(img_w,img_h))
        annotation= cv2.resize(annotation,(img_w,img_h))
        train_data.append(normalized(image))#normalized(cv2.imread(os.getcwd() + "/Intcatch_Dataset/" + txt[i][0])))
        train_label.append(binarylab(annotation[:,:,0]))
        print(os.getcwd() + "/" + path + txt[i][0])
        print(os.getcwd() + "/" + path + txt[i][1])
        train_data_path.append(os.getcwd() + "/" + path + txt[i][0])
        train_label_path.append(os.getcwd() + "/" + path + txt[i][1])
    return np.array(train_data), np.array(train_label)

train_data, train_label = prep_data( path, True, True )
#print(len(train_data))
#train_label = np.reshape(train_label,(len(train_data),data_shape,n_labels))
path2= 'Validation/'
val_data, val_label = prep_data( path2, False, False )
#val_label = np.reshape(val_label,(len(val_data),data_shape,n_labels))




print(len(train_data))"""

print("loading training data")
train_data = pk.load(open("train_data.pk", "rb"))
print("loading training labels")
train_label = pk.load(open("train_label.pk", "rb"))
print("loading validation data")
val_data = pk.load(open("val_data.pk", "rb"))
print("loading validation labels")
val_label = pk.load(open("val_label.pk", "rb"))

"""plt.imshow(train_data[0])
plt.show()
plt.imshow(np.argmax(train_label[0],axis=-1))
plt.show()
plt.imshow(train_data[1])
plt.show()
plt.imshow(np.argmax(train_label[1],axis=-1))
plt.show()
plt.imshow(train_data[2])
plt.show()
plt.imshow(np.argmax(train_label[2],axis=-1))
plt.show()
plt.imshow(train_data[3])
plt.show()
plt.imshow(np.argmax(train_label[3],axis=-1))
plt.show()
plt.imshow(train_data[4])
plt.show()
plt.imshow(np.argmax(train_label[4],axis=-1))
plt.show()
plt.imshow(train_data[5])
plt.show()
plt.imshow(np.argmax(train_label[5],axis=-1))
plt.show()
plt.imshow(train_data[6])
plt.show()
plt.imshow(np.argmax(train_label[6],axis=-1))
plt.show()
plt.imshow(train_data[7])
plt.show()
plt.imshow(np.argmax(train_label[7],axis=-1))
plt.show()
plt.imshow(train_data[8])
plt.show()
plt.imshow(np.argmax(train_label[8],axis=-1))
plt.show()
plt.imshow(train_data[9])
plt.show()
plt.imshow(np.argmax(train_label[9],axis=-1))
plt.show()"""





"""fig = plt.figure(figsize=(10, 3))
fig.suptitle("Origin",fontsize=16)
sub1 = plt.subplot(1,1,1)
sub1.imshow(train_data[0])
sub1.axis('off')
fig = plt.figure(figsize=(10, 3))
fig.suptitle("Zoom and Crop over horizont",fontsize=16)
sub2 = plt.subplot(1, 2, 1)
sub2.imshow(train_data[1])
sub2.axis('off')
sub3 = plt.subplot(1, 2, 2)
sub3.imshow(train_data[2])
sub3.axis('off')
plt.show()

import tensorlayer as tl
x=train_data[0]
fig = plt.figure(figsize=(10, 3))
fig.suptitle("Rotation",fontsize=16)
sub1 = plt.subplot(1,1,1)
x1=tl.prepro.rotation(x,rg=20, is_random=False, fill_mode='reflect') # left right
sub1.imshow(x1)
sub1.axis('off')
plt.show()
fig = plt.figure(figsize=(10, 3))
fig.suptitle("Flip",fontsize=16)
sub1 = plt.subplot(1,1,1)
x2=tl.prepro.flip_axis(x,axis=1, is_random=False) # left right
sub1.imshow(x2)
sub1.axis('off')
plt.show()
fig = plt.figure(figsize=(10, 3))
fig.suptitle("Shear",fontsize=16)
sub1 = plt.subplot(1,1,1)
x3= tl.prepro.shear(x, 0.3, is_random=False, fill_mode='reflect')
sub1.imshow(x3)
sub1.axis('off')
plt.show()
fig = plt.figure(figsize=(10, 3))
fig.suptitle("Zoom",fontsize=16)
sub1 = plt.subplot(1,1,1)
x4 = tl.prepro.zoom(x,zoom_range=[0.5, 0.6], is_random=False, fill_mode='reflect')
sub1.imshow(x4)
sub1.axis('off')
plt.show()
fig = plt.figure(figsize=(10, 3))
fig.suptitle("Brightness",fontsize=16)
sub1 = plt.subplot(1,1,1)
x5 = tl.prepro.brightness(x, gamma=1, gain=1, is_random=True)
sub1.imshow(x5)
sub1.axis('off')
plt.show()"""





import os
import tensorlayer as tl
with open(path+'data.txt') as f:
    txt = f.readlines()
    txt = [line.split(' ') for line in txt]

#image= cv2.imread(os.getcwd() + "/" + path + txt[1][0])
#image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#annotation=cv2.imread(os.getcwd() + "/" + path + txt[1][1])

"""plt.imshow(image)
plt.show()
plt.imshow(visualize(annotation[:,:,0]))
plt.show()

x,y=tl.prepro.rotation_multi([image, annotation],rg=20, is_random=True, fill_mode='reflect') # left right
plt.imshow(x)
plt.show()
plt.imshow(visualize(y[:,:,0]))
plt.show()

x,y=tl.prepro.flip_axis_multi([image, annotation],axis=1, is_random=True) # left right
plt.imshow(x)
plt.show()
plt.imshow(visualize(y[:,:,0]))
plt.show()

x, y = tl.prepro.shear_multi([image, annotation], 0.05, is_random=True, fill_mode='reflect')
plt.imshow(x)
plt.show()
plt.imshow(visualize(y[:,:,0]))
plt.show()

x, y = tl.prepro.zoom_multi([image, annotation],zoom_range=[0.5, 0.7], is_random=True, fill_mode='reflect')
plt.imshow(x)
plt.show()
plt.imshow(visualize(y[:,:,0]))
plt.show()

x,y = tl.prepro.brightness_multi([image, annotation], gamma=1, gain=1, is_random=True)
plt.imshow(x)
plt.show()
plt.imshow(visualize(y[:,:,0]))
plt.show()"""





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.10, random_state=42)
X_val=val_data
y_val=val_label

del train_data
del train_label
del val_data
del val_label






print(X_train[0].shape, " --- ", y_train[0].shape )
print("--------")
print(len(X_train), " --- ", len(y_train))






from tensorlayer.layers import *
def u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')
	
	
	"""weights = tf.get_variable('conv_5_2/W_conv2d')

	# scale weights to [0 255] and convert to uint8 (maybe change scaling?)
	x_min = tf.reduce_min(weights)
	x_max = tf.reduce_max(weights)
	weights_0_to_1 = (weights - x_min) / (x_max - x_min)
	weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)

	# to tf.image_summary format [batch_size, height, width, channels]
	weights_transposed = tf.transpose (weights_0_to_255_uint8, [3, 0, 1, 2])

	# this will display random 3 filters from the 64 in conv1
	tf.image_summary('conv5_2/filters', weights_transposed, max_images=3)
"""
        up4 = DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
        with tf.variable_scope('conv5_2', reuse = True) as scope_conv:
			weights = tf.get_variable('W_conv2d')

			# scale weights to [0 255] and convert to uint8 (maybe change scaling?)
			x_min = tf.reduce_min(weights)
			x_max = tf.reduce_max(weights)
			weights_0_to_1 = (weights - x_min) / (x_max - x_min)
			weights_0_to_255_uint8 = tf.image.convert_image_dtype (weights_0_to_1, dtype=tf.uint8)
            weights_remapped = tf.reshape(weights_0_to_255_uint8, [3,3,1,-1])

			# to tf.image_summary format [batch_size, height, width, channels]
			weights_transposed = tf.transpose (weights_remapped, [3, 0, 1, 2])

			# this will display random 3 filters from the 64 in conv1
			tf.summary.image('conv5_2/filters', weights_transposed, max_outputs=3)
    return conv1

def reduced_u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        inputs = InputLayer(x, name='inputs')
        conv1 = Conv2d(inputs, 4, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = Conv2d(conv1, 4, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = Conv2d(pool1, 8, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = Conv2d(conv2, 8, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = Conv2d(pool2, 16, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = Conv2d(conv3, 16, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = Conv2d(pool3, 32, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = Conv2d(conv4, 32, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = Conv2d(pool4, 64, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = Conv2d(conv5, 64, (3, 3), act=tf.nn.relu, name='conv5_2')

        up4 = DeConv2d(conv5, 32, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = Conv2d(up4, 32, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = Conv2d(conv4, 32, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = DeConv2d(conv4, 16, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = Conv2d(up3, 16, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = Conv2d(conv3, 16, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = DeConv2d(conv3, 8, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = Conv2d(up2, 8, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = Conv2d(conv2, 8, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = DeConv2d(conv2, 4, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 4, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = Conv2d(conv1, 4, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    return conv1






###======================== HYPER-PARAMETERS ============================###
batch_size = 1
lr = 0.0001 
# lr_decay = 0.5
# decay_every = 100
beta1 = 0.9
n_epoch = 50
print_freq_step = 1
summary_record_step = 4





import tensorlayer as tl
import os, time
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
###======================== DEFIINE MODEL =======================###
t_image = tf.placeholder('float32', [batch_size, img_h, img_w, 3], name='input_image')
## labels are either 0 or 1
t_seg = tf.placeholder('float32', [batch_size, img_h, img_w, 2], name='target_segment')
## train inference
net = u_net(t_image, is_train=True, reuse=False, n_out=2) #reduced_u_net
## test inference
net_test = u_net(t_image, is_train=False, reuse=True, n_out=2) #reduced_u_net
###======================== DEFINE LOSS =========================###
## train losses
out_seg = net.outputs
print(tf.shape(out_seg))

# scale weights to [0 255] and convert to uint8 (maybe change scaling?)
x_min = tf.reduce_min(out_seg)
x_max = tf.reduce_max(out_seg)
out_seg_0_to_1 = (out_seg - x_min) / (x_max - x_min)
out_seg_0_to_255_uint8 = tf.image.convert_image_dtype (out_seg_0_to_1, dtype=tf.uint8)

output_image_l1 = tf.slice(out_seg_0_to_255_uint8, [0,0,0,0], [batch_size, img_h, img_w, 1])
output_image_l2 = tf.slice(out_seg_0_to_255_uint8, [0,0,0,1], [batch_size, img_h, img_w, 1])
# to tf.image_summary format [batch_size, height, width, channels]
#out_seg_transposed = tf.transpose (out_seg_0_to_255_uint8, [2, 0, 1])

# this will display random 3 filters from the 64 in conv1
tf.summary.image('output_layer_1', output_image_l1, max_outputs=1)
tf.summary.image('output_layer_2', output_image_l2, max_outputs=1)


with tf.name_scope('train_loss'):
	with tf.name_scope('dice_soft'):
		dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
                tf.summary.scalar("dice_loss", dice_loss)
	with tf.name_scope('iou_loss'):
		iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3])
                tf.summary.scalar("iou_loss", iou_loss)
	with tf.name_scope('dice_hard'):
		dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3])
                tf.summary.scalar("dice_hard", dice_hard)
loss = dice_loss

## test losses
test_out_seg = net_test.outputs
with tf.name_scope('test_loss'):
	with tf.name_scope('dice_soft'):
		test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
	with tf.name_scope('iou_loss'):
		test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3])
	with tf.name_scope('dice_hard'):
		test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3])

###======================== DEFINE TRAIN OPTS =======================###
t_vars = tl.layers.get_variables_with_name('u_net', True, True)
with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr, trainable=False)

with tf.name_scope('adam_optimizer'):
	train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)
###======================== LOAD MODEL ==============================###
merged = tf.summary.merge_all()
sum_writer = tf.summary.FileWriter("logs", sess.graph)
tl.layers.initialize_global_variables(sess)
## load existing model if possible
## tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}.npz'.format(task), network=net)




#Training
for epoch in range(0, n_epoch+1):
    epoch_time = time.time()
    total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
    i = 0;
    for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                batch_size=batch_size, shuffle=True):
        images, labels = batch
        step_time = time.time()
        # Data augmentation
        #data = tl.prepro.threading_data([_ for _ in zip(images, labels)], distort_img)
        #images, labels = data.transpose((1,0,2,3,4))
        
        ## update network
        _dice = 0; _iou = 0; _diceh = 0; out = 0;
        if i % summary_record_step == 0:
            _, _dice, _iou, _diceh, out, summy = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs, merged],
                    {t_image: images, t_seg: labels})
            sum_writer.add_summary(summy, i);
        

        else:
            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    {t_image: images, t_seg: labels})



        total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
        n_batch += 1
        if n_batch % print_freq_step == 0:
            print("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs"
            % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))
            """plt.imshow(images[0])
            plt.show()
            plt.imshow(np.argmax(labels[0],axis=-1))
            plt.show()
            plt.imshow(np.argmax(out[0],axis=-1))
            plt.show()"""
        

        ## check model fail
        if np.isnan(_dice):
            exit(" ** NaN loss found during training, stop training")
        if np.isnan(out).any():
            exit(" ** NaN found in output images during training, stop training")
        
        i += 1

    print(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs" %
            (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

    ###======================== EVALUATION TEST SET ==========================###
    total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
    for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                    batch_size=batch_size, shuffle=True):
        b_images, b_labels = batch
        _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                test_iou_loss, test_dice_hard, net_test.outputs],
                {t_image: b_images, t_seg: b_labels})
        total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
        n_batch += 1

    print(" **"+" "*12+"test 1-dice: %f hard-dice: %f iou: %f" %
            (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
    
    
    # save the test model as 'u_net_test_model'
    saver = tf.train.Saver()
    saver.save(sess, './Tensorflow_models/u_net_test_model')









