
import ddim
import constants as const
import utils as U

import keras
from keras import ops
import tensorflow as tf

import os
import datetime


# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
image_channel = X_train.shape[-1]

# Normalize data to (0, 1)
x_train = U.normalize_batch(X_train, low_s=0, high_s=255, low_t=0, high_t=1)

# Use only a subset of the data
# num_samples = 1000
num_samples = len(x_train)
x_train = x_train[:num_samples]

# Resize images
x_train = ops.image.resize(x_train, (const.IMAGE_DIM, const.IMAGE_DIM))

# Constract TF dataset
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(buffer_size=1024).batch(const.BATCH_SIZE)

# Define diffusion model
diffmodel = ddim.create_model(const.IMAGE_DIM, image_channel, const.WIDTHS, const.BLOCK_DEPTH)

# Compute normalization statistics
diffmodel.normalizer.adapt(dataset)

# Check if MODEL_DIR exists, create if not
if not os.path.exists(const.MODEL_DIR):
    os.makedirs(const.MODEL_DIR)

# Create callbacks for saving model and display samples
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join(const.MODEL_DIR, f"ddim-cifar10-keras-{current_time}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

display_callback = ddim.DisplayCallback(
    diffusion_steps=const.PLOT_DIFFUSION_STEPS,
    checkpoint_dir=checkpoint_dir,
)

# checkpoint_path = os.path.join(checkpoint_dir, "model.keras")
checkpoint_path = os.path.join(checkpoint_dir, "ddim_model.weights.h5")
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq="epoch",
    save_weights_only=True,
)
# Build the model by specifying the input shape
diffmodel.build(input_shape=(None, const.IMAGE_DIM, const.IMAGE_DIM, const.IMAGE_CHANNEL))

# Train the model
diffmodel.fit(
    dataset,
    epochs=const.NUM_EPOCHS,
    # validation_data=val_dataset,
    callbacks=[
        display_callback, 
        checkpoint_callback,
    ],
)