import keras
from keras import ops
import math
import numpy as np
import matplotlib.pyplot as plt
import constants as const

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    """
    Create a sinusoidal embedding of the given tensor.

    Args:
        x: A tensor to embed.

    Returns:
        A tensor with the sinusoidal embedding of the given tensor.
    """

    frequencies = ops.exp(
        ops.linspace(
            ops.log(const.EMBEDDING_MIN_FREQUENCY),
            ops.log(const.EMBEDDING_MAX_FREQUENCY),
            const.EMBEDDING_DIMS // 2,
        )
    )

    angular_speeds = ops.cast(2.0 * math.pi * frequencies, dtype="float32")
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def residual_block(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = keras.layers.Conv2D(width, kernel_size=1)(x)
        
        x = keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = keras.layers.Add()([x, residual])
        return x
    
    return apply

def down_block(width, block_depth):
    """
    Create a down-sampling block with the given parameters.

    Args:
        width (int): The width of the block.
        block_depth (int): The depth of the block.
    Return:
        A function that applies the down-sampling block to the given input.
    """
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = residual_block(width)(x)
            skips.append(x)
        x = keras.layers.AveragePooling2D(pool_size=2)(x)
        return x
    
    return apply

def up_block(width, block_depth):
    def apply(x):
        x, skips = x
        x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = keras.layers.Concatenate()([x, skips.pop()])
            x = residual_block(width)(x)
        return x 
    
    return apply


def create_unet(image_dim, image_channel, widths, block_depth):
    """
    Create a U-Net architecture with the given parameters.
    Args:
        image_dim (dim): The image dimension.
        image_channel (int): The number of image channels.
        widths (list[int]): The width of each layer.
        block_depth: The depth of each block.

    Returns:
        A U-Net model (keras.Model).
    """
    # Input placeholders
    noisy_images = keras.Input(shape=(image_dim, image_dim, image_channel))
    noise_rates = keras.Input(shape=(1, 1, 1)) # noise_rates

    # Embeddings for the images
    x = keras.layers.Conv2D(widths[0], kernel_size=1)(noisy_images)

    # Embeddings for the noise rates
    e = keras.layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, image_dim))(noise_rates)
    e = keras.layers.UpSampling2D(size=image_dim, interpolation="nearest")(e)

    # Concatenate both embeddings
    x = keras.layers.Concatenate()([x, e])

    skips = []

    # Encoder
    for width in widths[:-1]:
        x = down_block(width, block_depth)([x, skips])
    
    # Middle / Bottleneck block
    for _ in range(block_depth):
        x = residual_block(widths[-1])(x)

    # Decoder
    for width in reversed(widths[:-1]):
        x = up_block(width, block_depth)([x, skips])
    
    x = keras.layers.Conv2D(image_channel, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(inputs=[noisy_images, noise_rates], outputs=x, name="residual_unet")
                                                                           
    

if __name__ == "__main__":
    # # Test sinusoidal embedding
    # x = np.linspace(0, 1, 10).reshape(-1, 1, 1, 1)
    # embeddings = sinusoidal_embedding(x)
    # # embeddings = embeddings.numpy()


    # Test U-Net
    model = create_unet(const.IMAGE_DIM, 3, const.WIDTHS, const.BLOCK_DEPTH)
    print(model.summary(expand_nested=True))
    keras.utils.plot_model(model, to_file="diff_arch.png", expand_nested=True, show_shapes=True)

    # Inference with unet model
    n_samples = 1000
    noisy_images = np.random.rand(n_samples, const.IMAGE_DIM, const.IMAGE_DIM, 3)
    noise_rates = np.random.rand(n_samples, 1, 1, 1)
    denoised_images = model.predict([noisy_images, noise_rates])
