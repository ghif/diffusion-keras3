import ddim
import constants as const
import utils as U

import tensorflow as tf
import keras
from keras import ops
import numpy as np
import os

def forward_diffusion(diffmodel, xo, noises, steps=5):
    """
    Perform the forward diffusion process on the input image.
    Args:
        diffmodel: The diffusion model object that contains the diffusion schedule and normalization methods.
        xo: The original input image.
        noises: The noise to be added to the image at each step.
        steps (int, optional): The number of diffusion steps. Default is 5.
    Returns:
        list: A list of noisy images generated at each diffusion step, including the original image.
    """
    
    # Progressive schedule
    diffusion_times_t = np.linspace(0, 1, steps)
    noise_rates_t, signal_rates_t = diffmodel.diffusion_schedule(diffusion_times_t, const.MIN_SIGNAL_RATE, const.MAX_SIGNAL_RATE)

    # Normalize the image to have zero mean and unit variance
    xo_n = diffmodel.normalizer(xo, training=False)

    # Forward diffusion
    noisy_images_list = [xo]
    for t in range(steps):
        # Mix the images with noises accordingly
        noise_rates = noise_rates_t[t]
        signal_rates = signal_rates_t[t]
        noisy_images = signal_rates * xo_n + noise_rates * noises

        # Denormalize the images
        xt = diffmodel.denormalize(noisy_images).numpy()
        noisy_images_list.append(xt[0])

    return noisy_images_list



# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
image_channel = X_train.shape[-1]

# Normalize data to (0, 1)
x_train = U.normalize_batch(X_train, low_s=0, high_s=255, low_t=0, high_t=1)

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(buffer_size=1024).batch(const.BATCH_SIZE)

# Get batched data
batch_size = 20
x_batch = x_train[:batch_size]

# Define diffusion model
diffmodel = ddim.create_model(const.IMAGE_DIM, image_channel, const.WIDTHS, const.BLOCK_DEPTH)


diffmodel.normalizer.adapt(dataset)

# normalize images to have standard deviation of 1, like the noises
batch_size = ops.shape(x_batch)[0]
image_size = ops.shape(x_batch)[1]
image_ch = ops.shape(x_batch)[-1]

# noises = keras.random.normal(shape=(batch_size, image_size, image_size, image_ch))
# noises = np.linspace(0, 1, batch_size * image_size * image_size * image_ch).reshape(batch_size, image_size, image_size, image_ch)

# # sample uniform random diffusion times
# diffusion_times = keras.random.uniform(
#     shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
# )

# noise_rates, signal_rates = diffmodel.diffusion_schedule(diffusion_times, const.MIN_SIGNAL_RATE, const.MAX_SIGNAL_RATE)

# # mix the images with noises accordingly
# noisy_images = signal_rates * x_batch + noise_rates * noises

# noises = np.random.rand(batch_size, image_size, image_size, image_ch)

# # Forwrad diffusion
# image = x_batch[11]
# steps = 5
# noises = keras.random.normal(shape=(batch_size, image_size, image_size, image_ch))
# noisy_images_list = forward_diffusion(diffmodel, image, noises, steps=steps)
# U.visualize_imgrid(noisy_images_list, title="Forward Diffusion", plot_dim=(1, steps+1))

# Load model from .keras checkpoint
diffmodel.build(input_shape=(None, const.IMAGE_DIM, const.IMAGE_DIM, const.IMAGE_CHANNEL))
checkpoint_dir = os.path.join(const.MODEL_DIR, f"ddim-cifar10-keras-20250116-100337")
checkpoint_path = os.path.join(checkpoint_dir, "ddim_model.weights.h5")
print(f"Load model from checkpoint: {checkpoint_path}")
diffmodel.load_weights(checkpoint_path)

# Reverse diffusion

diffusion_steps = const.PLOT_DIFFUSION_STEPS

