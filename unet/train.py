import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
# from utils import (
    # load_checkpoint,
    # save_checkpoint,
    # get_loaders,
    # check_accuracy,
    # save_predictions_as_imgs,
# )

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160 # 1280 originally
IMAGE_WIDTH = 240 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "training_data/training_set/train_images/"
TRAIN_MASK_DIR = "training_data/training_set/train_masks/"
VAL_IMG_DIR = "training_data/training_set/val_images/"
VAL_MASK_DIR = "training_data/training_set/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    pass
