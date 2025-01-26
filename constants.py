# Sinusoidal embedding constants
EMBEDDING_DIMS = 32
EMBEDDING_MIN_FREQUENCY = 1.0
EMBEDDING_MAX_FREQUENCY = 1000.0
WIDTHS = [32, 64, 96, 128]
BLOCK_DEPTH = 2 # residual block depth in downsampling and upsampling blocks

# Data
IMAGE_DIM = 32
IMAGE_CHANNEL = 3

# Sampling
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95
PLOT_DIFFUSION_STEPS = 20

# Optimizations
EMA = 0.999
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50

# Visualization
NUM_SAMPLES = 64

# Paths
MODEL_DIR = "models"