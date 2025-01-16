import keras
from keras import ops
import tensorflow as tf
import os

import architectures as arch
import constants as const
import utils as U
from time import process_time


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, latent_variable=None, diffusion_steps=5, checkpoint_dir="checkpoints"):
        """
        Initialize the display callback.
        """
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.train_steps = 0
        self.checkpoint_dir = checkpoint_dir

        if latent_variable is not None:
            self.latent_variable = latent_variable
        else:
            self.latent_variable = keras.random.normal(
                shape=(const.NUM_SAMPLES, const.IMAGE_DIM, const.IMAGE_DIM, const.IMAGE_CHANNEL)
            )

    def on_train_batch_end(self, batch, logs=None):
        if self.train_steps % 100 == 0:
            generated_images = self.model.generate(
                self.latent_variable,
                self.diffusion_steps
            )
            checkpoint_path = os.path.join(self.checkpoint_dir, f"denoised_images_step-{self.train_steps}.jpg")
            print(f" ... [{self.train_steps}] Visualize {checkpoint_path}")
            U.visualize_grid(
                generated_images, 
                figpath=checkpoint_path
            )

        self.train_steps += 1

@keras.saving.register_keras_serializable()
class DDIM(keras.Model):
    def __init__(self, image_size, image_channel, widths, block_depth):
        """
        Initialize the DDIM model.

        Args:
            image_size (int): The image size.
            image_channel (int): The image channel.
            widths (list[int]): The width of each layer.
            block_depth (int): The depth of each block.
        """
        super().__init__()
        self.normalizer = keras.layers.Normalization()
        self.network = arch.create_unet(image_size, image_channel, widths, block_depth)

        # Exponential moving average network for smoother sampling
        self.ema_network = arch.create_unet(image_size, image_channel, widths, block_depth)
    
    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.MeanAbsoluteError(name="n_loss")
        self.image_loss_tracker = keras.metrics.MeanAbsoluteError(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]
    
    def denormalize(self, images):
        """
        Convert the pixel values back to 0-1 range

        Args:
            images (tf.Tensor): The images to denormalize.
        
        Returns:
            The denormalized images.
        """
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return ops.clip(images, 0.0, 1.0)
    
    def diffusion_schedule(self, diffusion_times, min_signal_rate, max_signal_rate):
        """
        Create the diffusion schedule.

        Args:
            diffusion_times (tf.Tensor): The diffusion timenoses.
            min_signal_rate (float): The minimum signal rate.
            max_signal_rate (float): The maximum signal rate.

        Returns:
            The noise rates and signal rates.
        """
        start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)

        return noise_rates, signal_rates
    
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """
        Denoise the noisy images.

        Args:
            noisy_images (tf.Tensor): The noisy images.
            noise_rates (tf.Tensor): The noise rates.
            signal_rates (tf.Tensor): The signal rates.
            training (bool): Whether the model is training.
        Returns:
            The denoised images.
        """
        if training:
            network = self.network
        else:
            network = self.ema_network
        
        # Predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """
        Reverse the diffusion process to generate images from noise
        
        Args:
            initial_noise (Tensor): the initial noise tensor (n, dim1, dim2, ch)
            diffusion_steps (int): the number of diffusion steps to reverse

        Returns:
            pred_images: the predicted images
        """

        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # Important: at the first sampling step, the "noisy image" is purse noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(
                diffusion_times, 
                const.MIN_SIGNAL_RATE, 
                const.MAX_SIGNAL_RATE
            )

            # network used in eval mode
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times,
                const.MIN_SIGNAL_RATE,
                const.MAX_SIGNAL_RATE
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images
    
    def generate(self, initial_noise, diffusion_steps):
        """
        Generate images from noise

        Args:
            initial_noise (tf.Tensor): the initial noise tensor
            diffusion_steps (int): the number of diffusion steps to reverse
        
        Returns:
            generated_images (tf.Tensor): the generated images
        """
        # noise -> images -> denormalized images
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        """
        Train the diffusion model on a batch of images

        Args:
            images (tf.Tensor): the batch of images
        
        Returns:
            metrics (dict): the metrics to log
        """
        # normalize images to have standard deviation of 1, like the noises
        batch_size = ops.shape(images)[0]
        image_size = ops.shape(images)[1]
        image_ch = ops.shape(images)[-1]

        images = self.normalizer(images, training=True)
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, image_ch))

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )

        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times, const.MIN_SIGNAL_RATE, const.MAX_SIGNAL_RATE)

        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises) # used for training
            # image_loss = self.loss(images, pred_images) # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noises, pred_noises)
        self.image_loss_tracker.update_state(images, pred_images)

        # track the exponential moving average of the network weights
        for weight, ema_weights in zip(self.network.weights, self.ema_network.weights):
            ema_weights.assign(const.EMA * ema_weights + (1-const.EMA) * weight)

        # KID is not measured during the training phase for computational efficiency
        # return {m.name: m.result() for m in self.metrics[:-1]}
        return {m.name: m.result() for m in self.metrics}

    
def create_model(image_dim, image_channel, widths, block_depth):
    diffmodel = DDIM(image_dim, image_channel, widths, block_depth)
    diffmodel.summary(expand_nested=True)

    loss_fn = keras.losses.MeanAbsoluteError()
    diffmodel.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY
        ),
        loss=loss_fn,
    )
    return diffmodel

if __name__ == "__main__":
    image_ch = 3
    diffmodel = DDIM(const.IMAGE_DIM, image_ch, const.WIDTHS, const.BLOCK_DEPTH)
    print(diffmodel.summary())

    # Test diffusion schedule
    batch_size = 10
    diffusion_times = keras.random.uniform(
        shape=(batch_size, 1, 1, 1), minval=0, maxval=1.0
    )

    noise_rates, signal_rates = diffmodel.diffusion_schedule(diffusion_times, const.MIN_SIGNAL_RATE, const.MAX_SIGNAL_RATE)
    print(f"diffusion_times: {diffusion_times}")

    # Test reverse diffusion
    print("Test reverse diffusion")
    initial_noise = keras.random.uniform(
        shape=(batch_size, const.IMAGE_DIM, const.IMAGE_DIM, image_ch), minval=-1, maxval=1
    )
    start_t = process_time()
    pred_images = diffmodel.reverse_diffusion(initial_noise, const.PLOT_DIFFUSION_STEPS)
    elapsed_t = process_time() - start_t
    print(f" -- Elapsed time: {elapsed_t:.2f} secs")