import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import time
import numpy as np

import keras_hub
from PIL import Image
from utils import display_generated_images

# Constants
IMAGE_SHAPE = (256, 256, 3)
NUM_STEPS = 28 # default number: 28
STORE_INTERMEDIATE = True

def diffusion_callback(step, timestep, latents):
    """Callback function to store or display intermediate images during the diffusion process."""
    if STORE_INTERMEDIATE and step % 5 == 0:
        try:
            decoded_image = backbone.vae.decode(latents)
            print(f"decoded_image: {decoded_image.shape}")

        except Exception as e:
            print(f"Error during decoding: {e}")


backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
    "stable_diffusion_3_medium", dtype="float16", image_shape=IMAGE_SHAPE
)

preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
    "stable_diffusion_3_medium"
)

text_to_image = keras_hub.models.StableDiffusion3TextToImage(backbone=backbone, preprocessor=preprocessor)

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

prompt = "High quality 8K painting impressionist style of a Indonesian modern city street with a boy on the foreground wearing a traditional dress, staring at the sky, daylight"


intermediate_steps = list(np.linspace(0, NUM_STEPS, num=5, dtype=int))

timestamp = int(time.time())

for step in intermediate_steps:
    print(f"Step: {step}")
    start_t = time.time()
    generated_image = text_to_image.generate(
        prompt,
        num_steps=step,
    )
    elapsed_t = time.time() - start_t
    print(f"[Step-{step}] Elapsed time: {elapsed_t} (secs)")
    print("Generated image shape: ", generated_image.shape)
    # display_generated_images(generated_image)

    # Save the generated image
    save_folder = "images"
    
    image_path = os.path.join(save_folder, f"sd3_text2img_{IMAGE_SHAPE[0]}_{timestamp}_step-{step}.png")
    image = Image.fromarray(generated_image)
    image.save(image_path)
    print(f"Image saved to {image_path}")





