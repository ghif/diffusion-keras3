import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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

def visualize_grid(X, figpath=None, vmin=-1, vmax=1, suptitle=None):
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

    Xn = normalize_batch(
        X, low_s=np.min(X), high_s=np.max(X), low_t=0, high_t=1
    )
    for ax, im in zip(grid, Xn):
        ax.imshow(im)
        ax.axis("off")
    
    if figpath is not None:
        plt.savefig(figpath)
    else:
        plt.show()