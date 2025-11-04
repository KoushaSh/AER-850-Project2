# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 22:55:39 2025

@author: Kousha
"""

# AER850 - Project 2
# Step 1: Data loading + basic augmentation

import os



import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


base_dir = os.path.join(os.getcwd(), "Data")

train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

# sanity check 
print("Base data dir:", base_dir)
print("Train exists:", os.path.isdir(train_dir))
print("Valid exists:", os.path.isdir(valid_dir))

# Image + batch settings

IMG_HEIGHT = 500   
IMG_WIDTH = 500
CHANNELS = 3       
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

BATCH_SIZE = 32   

# Data augmentation

# Train: rescale + some light augmentation so the model doesn't just memorize
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10,        
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True      
)

# Validation: ONLY rescaling 
valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)



train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  
    shuffle=True,
    seed=SEED
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False              
)

# number of classes 
num_classes = len(train_generator.class_indices)
print("Class indices (label mapping):", train_generator.class_indices)
print("Number of classes:", num_classes)
print("Train samples:", train_generator.samples)
print("Validation samples:", valid_generator.samples)
