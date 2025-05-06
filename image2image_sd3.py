import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import time

import numpy as np

import keras_hub
from PIL import Image
from keras import layers

from utils import display_generated_images


IMAGE_SHAPE = (256, 256, 3)
IMAGE_FOLDER = "images"

# Open and preprocess an image
input_path = f"{IMAGE_FOLDER}/cat1.jpg"
image = Image.open(input_path).convert("RGB")
image = image.resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
width, height = image.size
image_arr = np.array(image, dtype=np.float32)

# Standardize the image values to the range [-1.0, 1.0]
rescale = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
pimage_arr = rescale(image_arr)

backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
    "stable_diffusion_3_medium", dtype="float16", image_shape=IMAGE_SHAPE
)
preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
    "stable_diffusion_3_medium"
)
image_to_image = keras_hub.models.StableDiffusion3ImageToImage(backbone, preprocessor)

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, "
prompt += "adorable, Pixar, Disney, 8k"

start_t = time.time()
generated_image = image_to_image.generate(
    {
        "images": pimage_arr,
        "prompts": prompt,
    }
)
elapsed_t = time.time() - start_t
print(f"Elapsed time: {elapsed_t} (secs)")
print("Generated image shape: ", generated_image.shape)

# Display the generated image
display_generated_images(generated_image)

# Save the generated image
filename = os.path.splitext(os.path.basename(input_path))[0] # Take the filename of input_path
timestamp = int(time.time())
output_path = f"{IMAGE_FOLDER}/{filename}_styled_sd3_{IMAGE_SHAPE[0]}_{timestamp}.png"
image = Image.fromarray(generated_image)
image.save(output_path)
print(f"Image saved to {output_path}")


