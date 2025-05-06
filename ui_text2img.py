import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import gradio as gr
import keras_hub
import tensorflow as tf
import numpy as np
from PIL import Image
import time


# --- Configuration ---
MODEL_PRESET = "stable_diffusion_3_medium"
IMAGE_SHAPE = (256, 256, 3) # Adjust as needed, larger sizes need more VRAM/time
DEFAULT_STEPS = 28
DTYPE = "float16" # Use "float16" for faster inference on compatible GPUs, "float32" otherwise

# --- Global Model Loading (Load once at startup) ---
print("Loading Stable Diffusion 3 models... This might take a while.")
try:
    # Ensure necessary environment variables are set if required by backend (e.g., Keras backend)
    # os.environ["KERAS_BACKEND"] = "tensorflow" # Or "jax" or "torch"

    backbone = keras_hub.models.StableDiffusion3Backbone.from_preset(
        MODEL_PRESET, dtype=DTYPE, image_shape=IMAGE_SHAPE
    )

    preprocessor = keras_hub.models.StableDiffusion3TextToImagePreprocessor.from_preset(
        MODEL_PRESET
    )

    text_to_image = keras_hub.models.StableDiffusion3TextToImage(
        backbone=backbone,
        preprocessor=preprocessor
    )
    print("Models loaded successfully.")
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models: {e}")
    print("Gradio UI will launch, but generation will fail.")
    MODELS_LOADED = False
    text_to_image = None # Ensure it's None if loading failed

# --- Generation Function ---
def generate_image(prompt, final_step, num_steps, seed=None):
    """Generates an image based on the prompt using the loaded SD3 model."""
    if not MODELS_LOADED or text_to_image is None:
        raise gr.Error("Models failed to load. Cannot generate image.")

    print(f"Generating image for prompt: '{prompt}'")
    

    # Handle seed: Use None for random seed if input is -1 or invalid
    if not isinstance(seed, int) or seed < 0:
        seed = None
        print("Using random seed.")
    else:
        print(f"Using seed: {seed}")

    try:
        intermediate_steps = np.linspace(0, final_step, num=num_steps, dtype=int)
        for step in intermediate_steps:
            # Generate image with the specified number of steps
            start_time = time.time()
            generated_images = text_to_image.generate(
                prompt,
                num_steps=step, # Ensure steps is int
                seed=seed,
            )
            end_time = time.time()
            print(f"[Step-{step+1}] Generation took {end_time - start_time:.2f} seconds.")

            # Assuming the output is a NumPy array [batch, height, width, channels]
            # and needs conversion to PIL Image for Gradio
            print(f"Generated images: {generated_images.dtype}, shape: {generated_images.shape}")
            if isinstance(generated_images, (np.ndarray, tf.Tensor)): # Check if Tensor too
                # Convert tensor to numpy if necessary
                if hasattr(generated_images, 'numpy'):
                    img_array = generated_images.numpy()
                else:
                    img_array = generated_images

                pil_image = Image.fromarray(img_array)
                yield pil_image
            else:
                # Handle cases where the output might already be a PIL image or other format
                print(f"Unexpected output type: {type(generated_images)}")
                # Attempt to return directly if it looks like an image
                if hasattr(generated_images, 'save'): # Duck-typing for PIL Image
                    # return generated_images
                    yield generated_images
                else:
                    raise gr.Error("Failed to process generated image format.")


    except Exception as e:
        print(f"Error during generation: {e}")
        # Raise a Gradio-specific error to display it nicely in the UI
        raise gr.Error(f"Generation failed: {e}")

# --- Gradio Interface Definition ---
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion 3 Text-to-Image with Keras Hub")
    gr.Markdown("Enter a prompt to generate an image using the SD3 Medium model.")

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")
            steps_input = gr.Slider(minimum=1, maximum=100, value=DEFAULT_STEPS, step=1, label="Inference Steps")
            seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
            intermediate_display_steps = gr.Number(label="Intermediate Steps (for display)", value=5, precision=0, visible=True)
            submit_button = gr.Button("Generate Image", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil") # Expect PIL image

    # Link button click to function
    submit_button.click(
        fn=generate_image,
        inputs=[prompt_input, steps_input, intermediate_display_steps, seed_input],
        outputs=output_image
    )

    gr.Examples(
        examples=[
            ["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", 28, -1],
            ["A photorealistic painting of a cat riding a bicycle on the moon", 28, 12345],
            ["Impressionist painting of a rainy street in Paris", 35, 54321],
            ["High-tech cityscape at sunset, cyberpunk style", 50, -1],
        ],
        inputs=[prompt_input, steps_input, seed_input],
        outputs=output_image,
        fn=generate_image, # Make examples clickable
        cache_examples=False # Don't cache results for dynamic generation
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()