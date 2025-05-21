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
    q(x_t | x_0) = \sqrt(\bar{\alpha}_t) * x_0 + \sqrt(1 - \bar{\alpha}_t) * \epsilon

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

def reverse_diffusion(diffmodel, initial_noise, steps=5):
    """
    Perform reverse diffusion process to generate images from initial noise.

    Args:
        diffmodel (object): The diffusion model used for denoising and scheduling.
        initial_noise (numpy.ndarray): The initial noise tensor to start the reverse diffusion process.
        steps (int, optional): The number of reverse diffusion steps. Default is 5.
    Returns:
        list: A list of predicted images at each step of the reverse diffusion process.
    """
    num_images = initial_noise.shape[0]
    step_size = 1.0 / steps

    next_noisy_images = initial_noise

    pred_images_list = [diffmodel.denormalize(initial_noise).numpy()[0]]
    for step in range(steps):
        noisy_images = next_noisy_images
        diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
        noise_rates, signal_rates = diffmodel.diffusion_schedule(
            diffusion_times, 
            const.MIN_SIGNAL_RATE, 
            const.MAX_SIGNAL_RATE
        )

        pred_noises, pred_images = diffmodel.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        pred_images_list.append(diffmodel.denormalize(pred_images).numpy()[0])

        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffmodel.diffusion_schedule(
            next_diffusion_times,
            const.MIN_SIGNAL_RATE,
            const.MAX_SIGNAL_RATE
        )
        next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
    
    return pred_images_list

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

# normalize images to have standard deviation of 1, like the noises
batch_size = ops.shape(x_batch)[0]
image_size = ops.shape(x_batch)[1]
image_ch = ops.shape(x_batch)[-1]

# Load model from .keras checkpoint
diffmodel = ddim.create_model(const.IMAGE_DIM, image_channel, const.WIDTHS, const.BLOCK_DEPTH)
diffmodel.normalizer.adapt(dataset)
diffmodel.build(input_shape=(None, const.IMAGE_DIM, const.IMAGE_DIM, const.IMAGE_CHANNEL))

checkpoint_dir = os.path.join(const.MODEL_DIR, f"ddim-cifar10-keras-best")
checkpoint_path = os.path.join(checkpoint_dir, "ddim_model.weights.h5")
print(f"Load model from checkpoint: {checkpoint_path}")
diffmodel.load_weights(checkpoint_path)



# Forwrad diffusion
print(f"Forward diffusion ...")
image = x_batch[9]
forward_steps = 10
noises = keras.random.normal(shape=(batch_size, image_size, image_size, image_ch))
noisy_images_list = forward_diffusion(diffmodel, image, noises, steps=forward_steps)
# figpath = os.path.join("images", "forward_diffusion.png")
# figpath = None
# U.visualize_imgrid(noisy_images_list, title=f"Forward Diffusion (steps: {forward_steps})", plot_dim=(1, forward_steps+1), figpath=figpath)
U.create_animated_gif(noisy_images_list, output_gif_path="images/animated_forward_diffusion_9.gif")

# # Reverse diffusion
# print(f"Reverse diffusion ...")
# reverse_steps = 10
# initial_noise = keras.random.normal(shape=(1, image_size, image_size, image_ch))
# pred_images_list = reverse_diffusion(diffmodel, initial_noise, reverse_steps)
# # figpath = os.path.join("images", "reverse_diffusion.png")
# figpath = None
# U.visualize_imgrid(pred_images_list, title=f"Reverse Diffusion (steps: {reverse_steps})", figpath=figpath)
