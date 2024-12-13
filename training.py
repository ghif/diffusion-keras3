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
num_samples = 10000
x_train = x_train[:num_samples]

# Resize images
x_train = ops.image.resize(x_train, (const.IMAGE_DIM, const.IMAGE_DIM))

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(buffer_size=1024).batch(const.BATCH_SIZE)

diffmodel = ddim.DDIM(const.IMAGE_DIM, image_channel, const.WIDTHS, const.BLOCK_DEPTH)

diffmodel.summary(expand_nested=True)

loss_fn = keras.losses.MeanAbsoluteError()
diffmodel.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY
    ),
    loss=loss_fn,
)

diffmodel.normalizer.adapt(dataset)

# run training and plot generated images periodically
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Check if MODEL_DIR exists, create if not
if not os.path.exists(const.MODEL_DIR):
    os.makedirs(const.MODEL_DIR)

checkpoint_dir = os.path.join(const.MODEL_DIR, f"ddim-cifar10-keras-{current_time}")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

display_callback = ddim.DisplayCallback(
    diffusion_steps=const.PLOT_DIFFUSION_STEPS,
    checkpoint_dir=checkpoint_dir,
)

diffmodel.fit(
    dataset,
    epochs=const.NUM_EPOCHS,
    # validation_data=val_dataset,
    callbacks=[
        display_callback
    ]
    # callbacks=[
    #     display_callback, 
    #     checkpoint_callback,
    # ],
)