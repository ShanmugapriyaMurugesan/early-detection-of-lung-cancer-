# Define paths to the training, validation, and test datasets
train_folder = 'train_folder/filepath/'
test_folder = 'test_folder/filepath'
validate_folder = 'validate_folder/filepath'

# Define paths to the specific classes within the dataset
normal_folder =' normal_folder/filepath'
adenocarcinoma_folder = 'adenocarcinoma/filepath'
large_cell_carcinoma_folder = 'large_cell_carcinoma/filepath'
squamous_cell_carcinoma_folder = 'squamous_cell_carcinoma/filepath'
# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import utils
from tensorflow.keras.models import load_model


print("Libraries Imported")

# Set the image size for resizing
IMAGE_SIZE = (350, 350)
BATCH_SIZE = 16  # or 8 if limited by memory


# Initialize the image data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define the batch size for training
batch_size = 8

# Create the training data generator
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

# Create the validation data generator
validation_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_folder,        # path to your test dataset directory
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)


OUTPUT_SIZE = train_generator.num_classes

# Load Xception base model pretrained on ImageNet, exclude top layers
base_model = tf.keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# Fine-tune top layers of Xception for your data
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Create the final model architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(OUTPUT_SIZE, activation='softmax')
])

# Compile model with Adam optimizer at a low learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks for training optimization
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=4,
    verbose=2,
    factor=0.5,
    min_lr=1e-6
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=2,
    restore_best_weights=True
)
checkpointer = ModelCheckpoint(
    filepath='best_model.keras',
    verbose=2,
    save_best_only=True
)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[learning_rate_reduction, early_stopping, checkpointer]
)

# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Evaluate the model on the test data
test_scores = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_scores[1]:.4f}")

# Function for prediction from an image path
# Function to load and preprocess an image for prediction
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Load, preprocess, and predict the class of an image
img_path = 'test/squamous.cell.carcinoma/000108 (6).png'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[predicted_class]

print(f"The image belongs to class: {predicted_label}")

# Display the image with the predicted class
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

# Repeat the process for additional images
img_path = 'adenocarcinoma/000108 (3).png'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class]
print(f"The image belongs to class: {predicted_label}")
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

img_path = 'large.cell.carcinoma/000108.png'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class]
print(f"The image belongs to class: {predicted_label}")
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

img_path = 'normal/6 - Copy (2).png'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class]
print(f"The image belongs to class: {predicted_label}")
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

# Save the improved model
model.save('/dataset_improved.h5')
