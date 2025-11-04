# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 22:55:39 2025

@author: Kousha
"""

# AER850 - Project 2


#==========================================
# Step 1: Data loading + basic augmentation
#==========================================

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

# =====================================================
# Step 2: CNN Model 1 
# =====================================================

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

def build_model_1(input_shape, num_classes):
    """
    First CNN attempt for the project.
    Nothing crazy, just 3 conv blocks + a small dense head.
    """
    model = models.Sequential()

    # --- Conv block 1 ---
    model.add(layers.Conv2D(32, (3, 3),
                            activation='relu',
                            padding='same',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Conv block 2 ---
    model.add(layers.Conv2D(64, (3, 3),
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Conv block 3 ---
    model.add(layers.Conv2D(128, (3, 3),
                            activation='relu',
                            padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Dense head ---
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))         # trying to keep overfitting under control
    model.add(layers.Dense(num_classes,
                           activation='softmax'))  

    return model


# building the model
model_1 = build_model_1(INPUT_SHAPE, num_classes)

# compiling the model

model_1.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)


print(model_1.summary())

# =====================================================
# Step 3: Training for Model 1 
# =====================================================

EPOCHS = 15

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history_1 = model_1.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=[early_stop],
    verbose=1
)


# history_1 will be used later to plot accuracy/loss (Step 4)
