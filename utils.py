import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

def normalize_batch(X_batch, low_s=0, high_s=255, low_t=-1, high_t=1):
    """
    Normalize the batch of images X_batch to the range [low_t, high_t]

    Args:
    - X_batch: np.ndarray, shape=(N, H, W, C)
    - low_s: float, minimum value in the input data
    - high_s: float, maximum value in the input data
    - low_t: float, minimum value in the normalized data
    - high_t: float, maximum value in the normalized data

    Returns:
    - X_batch: np.ndarray, shape=(N, H, W, C)
    """

    Xn = 1.0 * (X_batch - low_s) / (high_s - low_s)
    Xn = Xn * (high_t - low_t) + low_t
    return Xn

def visualize_imgrid(img_list, title="Untitled", figpath=None, plot_dim=None):
    """
    Visualize a list of images in a grid

    Args:
        - img_list: list[np.ndarray], list of images
        - title: str, title of the plot
        - figpath: str, path to save the figure (if None, show the plot)
        - plot_dim: tuple, dimension of the plot (if None, use the square root of the number of images

    """
    # plt.figure(figsize=(6, 6))
    # plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    plt.clf()
    n_samples = len(img_list)
    
    if plot_dim is None:
        # decide subplot dimensions
        d = np.sqrt(n_samples)
        d = np.ceil(d).astype("uint8")
        dx = dy = d
    else:
        dx, dy = plot_dim

    plt.title(title)
    plt.axis("off")

    for i in range(n_samples):
        ax = plt.subplot(dx, dy, i + 1)
        ax.axis("off")

        img = img_list[i]
    
        vmin = np.min(img)
        vmax = np.max(img)
        
        plt.imshow(img, vmin=vmin, vmax=vmax)
    
    if figpath is not None:
        plt.savefig(figpath)
    else:
        plt.show()

def visualize_grid(X, figpath=None, suptitle=None):
    """
    Visualizes a grid of images.

    Args:
        X (numpy.ndarray): A batch of images to be visualized.
        figpath (str, optional): Path to save the figure. If None, the figure is shown instead. Default is None.
        vmin (int, optional): Minimum value for normalization. Default is -1.
        vmax (int, optional): Maximum value for normalization. Default is 1.
        suptitle (str, optional): Super title for the figure. Default is None.
    """
    fig = plt.figure(figsize=(6, 6))
    if suptitle is not None:
        fig.suptitle(suptitle)

    # decide grid dimension
    n_samples = X.shape[0]

    d = np.sqrt(n_samples)
    d = np.ceil(d).astype("uint8")

    grid = ImageGrid(fig, 
        111, # similar to subplot(111)
        nrows_ncols=(d, d), # creates d x d grid
        axes_pad=0.05, # pad between axes in inch.
    )

    # Xn = normalize_batch(
    #     X, low_s=np.min(X), high_s=np.max(X), low_t=0, high_t=1
    # )
    # for ax, im in zip(grid, Xn):
    #     ax.imshow(im)
    #     ax.axis("off")
    
    for ax, im in zip(grid, X):
        ax.imshow(im, vmin=np.min(X), vmax=np.max(X))
        ax.axis("off")
    
    if figpath is not None:
        plt.savefig(figpath)
    else:
        plt.show()


def display_generated_images(images):
    """Helper function to display the images from the inputs.

    This function accepts the following input formats:
    - 3D numpy array.
    - 4D numpy array: concatenated horizontally.
    - List of 3D numpy arrays: concatenated horizontally.
    """
    display_image = None
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            display_image = Image.fromarray(images)
        elif images.ndim == 4:
            concated_images = np.concatenate(list(images), axis=1)
            display_image = Image.fromarray(concated_images)
    elif isinstance(images, list):
        concated_images = np.concatenate(images, axis=1)
        display_image = Image.fromarray(concated_images)

    if display_image is None:
        raise ValueError("Unsupported input format.")

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(display_image)
    plt.show()
    plt.close()

def create_animated_gif(img_list, output_gif_path="images/animated_forward_diffusion.gif"):
    """
    Create an animated GIF from a list of images.
    Args:
        img_list (list): List of images to be included in the GIF (float32 ranged from 0 to 1).
    """
    pil_images = []
    for img_in in img_list:
        # Convert the image to uint8 format
        img_array = np.clip(img_in, 0.0, 1.0)
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img_uint8))

    if pil_images:
        pil_images[0].save(
            output_gif_path,
            save_all=True,
            append_images=pil_images[1:], # Append the rest of the images
            duration=300,# Duration for each frame in milliseconds
            loop=0, # Loop indefinitely (0 means loop forever)
        )
        print(f"Animated GIF saved at {output_gif_path}")
    else:
        print("The list of images is empty. No GIF created.")