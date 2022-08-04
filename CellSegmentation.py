# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 23:57:43 2022

@author: user

Project 4 : Cell Nuclei Detection using Image Segmentation

"""
#1. Import Necessary Packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers,callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import io
import glob, os

#%%

#2. Load dataset
train_path = r"C:\Users\user\Desktop\AI07\Projects\P4\data-science-bowl-2018-2\test"
test_path = r"C:\Users\user\Desktop\AI07\Projects\P4\data-science-bowl-2018-2\train"

#Load images
def load_images(file_path):
    images=[]
    for image_file in os.listdir(file_path):
        img = cv2.imread(os.path.join(file_path,image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        images.append(img)
    return images

train_image_dir = os.path.join(train_path,'inputs')
train_images = load_images(train_image_dir)
test_image_dir = os.path.join(test_path,'inputs')
test_images = load_images(test_image_dir)

#Load masks
def load_masks(file_path):
    masks=[]
    for mask_file in os.listdir(file_path):
        mask = cv2.imread(os.path.join(file_path,mask_file),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(128,128))
        masks.append(mask)
    return masks
    
train_mask_dir = os.path.join(train_path,'masks')
train_masks = load_masks(train_mask_dir)
test_mask_dir = os.path.join(test_path,'masks')
test_masks = load_masks(test_mask_dir)

#%%

#Convert the lists into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%

#Check some examples
plt.figure(figsize=(10,5))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = train_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()   

plt.figure(figsize=(10,5))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = train_masks[i]
    plt.imshow(mask_plot, cmap='gray')
    plt.axis('off')
plt.show()   

#%%

#2. Data preprocessing
#Expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)
#Check the mask output
print(train_masks[0].min(), train_masks[0].max)

#%%

#Convert the mask values into class labels
converted_masks_train = np.round(train_masks_np_exp/255).astype(np.int64)
converted_masks_test = np.round(test_masks_np_exp/255).astype(np.int64)
#Check the mask output
print(np.unique(converted_masks_train[0]))

#%%

#Normalize image pixels value
converted_images_train = train_images_np / 255.0
converted_images_test = test_images_np / 255.0

#%%

#Perform train-validation split
from sklearn.model_selection import train_test_split

SEED = 12345
x_train,x_val,y_train,y_val = train_test_split(converted_images_train,converted_masks_train,test_size=0.2,random_state=SEED)

#%%

#Convert the numpy arrays into tensor 
train_x = tf.data.Dataset.from_tensor_slices(x_train)
val_x = tf.data.Dataset.from_tensor_slices(x_val)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
val_y = tf.data.Dataset.from_tensor_slices(y_val)
test_x = tf.data.Dataset.from_tensor_slices(converted_images_test)
test_y = tf.data.Dataset.from_tensor_slices(converted_masks_test)

#%%

#Combine the images and masks using zip
train = tf.data.Dataset.zip((train_x,train_y))
val = tf.data.Dataset.zip((val_x,val_y))
test = tf.data.Dataset.zip((test_x,test_y))

#%%

#Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 800//BATCH_SIZE
VALIDATION_STEPS = 200//BATCH_SIZE
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.batch(BATCH_SIZE).repeat()
val = val.prefetch(buffer_size=AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


#%%

#Visualize some examples
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
#%%

for images, masks in train.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
    
#%%

#3. Create image segmentation model
#Use a pretrained model as the feature extraction layers
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    #Apply functional API to construct U-Net
    #Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    #This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

#%%

#Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
#Compile the model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)
model.summary()

#%%

#Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])
        
#%%

#Test out the show_prediction function
show_predictions()

#%%

#Create a callback to help display results during model training
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
        
#%%

#4. Model training
#Hyperparameters for the model
EPOCHS = 100
history = model.fit(train,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,batch_size=BATCH_SIZE,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=val,
                    callbacks=[DisplayCallback()])

#%%

#5. Deploy model
show_predictions(test,3)

#%%

#Test evaluation
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss = {test_loss}")
print(f"Test accuracy = {test_accuracy}")
