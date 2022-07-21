# %%
import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt

from numba import jit
from numba import cuda
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# %%
# GRADED FUNCTION: split_data
def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  """
  Splits the data into train and test sets
  
  Args:
    SOURCE_DIR (string): directory path containing the images
    TRAINING_DIR (string): directory path to be used for training
    VALIDATION_DIR (string): directory path to be used for validation
    SPLIT_SIZE (float): proportion of the dataset to be used for training
    
  Returns:
    None
  """

  ### START CODE HERE
  file_list = []
  for i in os.listdir(SOURCE_DIR):
    if os.path.getsize(os.path.join(SOURCE_DIR, i)) == 0:
      print(i, 'is zero length, so ignoring.')
      continue

    file_list.append(i)

  random_sample = random.sample(file_list, len(file_list))
  indices = int(split_size*len(random_sample))
  for i in random_sample[:indices]:
    copyfile(os.path.join(SOURCE_DIR, i), os.path.join(TRAINING_DIR, i))
  for i in random_sample[indices:]:
    copyfile(os.path.join(SOURCE_DIR, i), os.path.join(VALIDATION_DIR, i))
  ### END CODE HERE

# %%
# Define root directory
root_dir = './malaria/'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)

# GRADED FUNCTION: create_train_val_dirs
def create_train_val_dirs(root_path):
  """
  Creates directories for the train and test sets
  
  Args:
    root_path (string) - the base directory path to create subdirectories from
  
  Returns:
    None
  """  
  ### START CODE HERE

  # HINT:
  # Use os.makedirs to create your directories with intermediate subdirectories
  # Don't hardcode the paths. Use os.path.join to append the new directories to the root_path parameter
  os.makedirs(os.path.join(root_path, "training/Parasitized"))
  os.makedirs(os.path.join(root_path, "training/Uninfected"))
  os.makedirs(os.path.join(root_path, "validation/Uninfected"))
  os.makedirs(os.path.join(root_path, "validation/Parasitized"))

  ### END CODE HERE

  
try:
  create_train_val_dirs(root_path=root_dir)
except FileExistsError:
  print("You should not be seeing this since the upper directory is removed beforehand")

# %%
parasitised = "./cell_images/Parasitized/"
normal = "./cell_images/Uninfected/"

training_dir = "./malaria/training/"
validation_dir = "./malaria/validation"


# %%
# GRADED FUNCTION: split_data
@jit
def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  """
  Splits the data into train and test sets
  
  Args:
    SOURCE_DIR (string): directory path containing the images
    TRAINING_DIR (string): directory path to be used for training
    VALIDATION_DIR (string): directory path to be used for validation
    SPLIT_SIZE (float): proportion of the dataset to be used for training
    
  Returns:
    None
  """

  ### START CODE HERE
  file_list = []
  for i in os.listdir(SOURCE_DIR):
    if os.path.getsize(os.path.join(SOURCE_DIR, i)) == 0:
      print(i, 'is zero length, so ignoring.')
      continue

    file_list.append(i)

  random_sample = random.sample(file_list, len(file_list))
  indices = int(split_size*len(random_sample))
  for i in random_sample[:indices]:
    copyfile(os.path.join(SOURCE_DIR, i), os.path.join(TRAINING_DIR, i))
  for i in random_sample[indices:]:
    copyfile(os.path.join(SOURCE_DIR, i), os.path.join(VALIDATION_DIR, i))
  ### END CODE HERE


# %%
training_dir_parasitised = os.path.join(training_dir, "Parasitized")
training_dir_normal = os.path.join(training_dir, "Uninfected")
validation_dir_parasitised = os.path.join(validation_dir, "Parasitized")
validation_dir_normal = os.path.join(validation_dir, "Uninfected")

# %%
# Empty directories in case you run this cell multiple times
if len(os.listdir(training_dir_normal)) > 0:
  for file in os.scandir(training_dir_normal):
    os.remove(file.path)
if len(os.listdir(training_dir_parasitised)) > 0:
  for file in os.scandir(training_dir_parasitised):
    os.remove(file.path)
if len(os.listdir(validation_dir_normal)) > 0:
  for file in os.scandir(validation_dir_normal):
    os.remove(file.path)
if len(os.listdir(validation_dir_parasitised)) > 0:
  for file in os.scandir(validation_dir_parasitised):
    os.remove(file.path)

# Define proportion of images used for training
split_size = .9

# Run the function
# NOTE: Messages about zero length images should be printed out
split_data(normal, training_dir_normal, validation_dir_normal, split_size)
split_data(parasitised, training_dir_parasitised, validation_dir_parasitised, split_size)


# %%
size = 600
batch_size = 40

# %%
# GRADED FUNCTION: train_val_generators
@jit
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators
  
  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  train_datagen = ImageDataGenerator(rescale = 1.0/255.)

  # Pass in the appropiate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=batch_size,
                                                      class_mode='binary',
                                                      target_size=(size, size),
                                                      shuffle = True)

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=batch_size,
                                                                class_mode='binary',
                                                                target_size=(size, size))
  ### END CODE HERE
  return train_generator, validation_generator


# %%
train_generator, validation_generator = train_val_generators(training_dir, validation_dir)

# %%
# GRADED FUNCTION: create_model
model = None

def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE

  model = tf.keras.models.Sequential([ 
      tf.keras.layers.Conv2D(16, (4,4), activation = 'relu', padding = 'same', input_shape = (size, size, 3)),
      tf.keras.layers.MaxPooling2D((3,3)),

      tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same'),
      tf.keras.layers.MaxPooling2D((2,2)),

      tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
      tf.keras.layers.MaxPooling2D((2,2)),

      tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
      tf.keras.layers.MaxPooling2D((2,2)),

      tf.keras.layers.Conv2D(128, (2,2), activation = 'relu', padding = 'same'),
      tf.keras.layers.MaxPooling2D((2,2)),




      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(256, activation = 'relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(512, activation = 'relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(1, activation = 'sigmoid')
  ])

  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                loss='binary_crossentropy',
                metrics=['accuracy']) 
    
  ### END CODE HERE

  return model



device = cuda.get_current_device()
device.reset()

# Get the untrained model
model = create_model()
model.summary()



# %%
# Train the model
# Note that this may take some time.
with tf.device("GPU:0"):
    history = model.fit(train_generator,
                        epochs=25,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2)])

# %%
model.save("high_acc_res")
model.save_weights("high_acc_res_weights")

# %%
device = cuda.get_current_device()
device.reset()


