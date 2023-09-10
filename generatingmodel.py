import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

# Define parameters
num_classes = len(mudra_names)

# Load MobileNetV2 from TensorFlow Hub
base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5", trainable=False)

# Build the model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load and preprocess the dataset
data_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_data = data_gen.flow_from_directory(
    path_to_dataset,
    target_size=input_shape,
    subset="training"
)

valid_data = data_gen.flow_from_directory(
    path_to_dataset,
    target_size=input_shape,
    subset="validation"
)

# Train the model
model.fit(train_data, validation_data=valid_data, epochs=10, batch_size=40)

model.save(path_to_model)
