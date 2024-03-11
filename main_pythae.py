from pathlib import Path
import argparse

import torchio as tio 
import torch
from torch.utils.data import DataLoader
import wandb

from data.dataset import IRCAD, IRCAD_pythae
from data.config import ConfigIRCAD
from data.util import *

from pythae.models import VAEConfig, VAE
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.trainers.training_callbacks import WandbCallback
from pythae.pipelines.training import TrainingPipeline


# ARGUMENTS
parser = argparse.ArgumentParser(description='My first experiment with Torchio, Pythae & IRCAD Dataset')

parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--dims', type=str, default="slice", choices=["2d", "slices", "3d", "volume"])
parser.add_argument('--input_dims', nargs='+', type=int, default=[64, 64, 16])

args = parser.parse_args()

input_dims = (64,64,16)

# PATHS
mount_dir = Path("/mnt/Shared/") if Path.exists(Path("/mnt/Shared/")) else Path.home() / "data"
root_data_dir = mount_dir / "3Dircadb1"

# TRANSFORMS
transforms = [
    tio.ToCanonical(),  # to RAS
    tio.Clamp(-150, 250),
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.Resample((1, 1, 2)),  # to 1 mm iso
    #tio.CropOrPad(input_dims, mask_name='liver'),
    ]


# DATA
dataset_config = ConfigIRCAD(root_data_dir)

# TRAIN
# TODO make compatible with queue, maybe come back to IRCAD simple 
train_dataset = IRCAD_pythae(
    config=dataset_config,
    subset='train',
    transform=tio.Compose(transforms),
)

train_patches_queue = tio.Queue(
    train_dataset,
    max_length=100,
    samples_per_volume=20,
    sampler=tio.data.UniformSampler(patch_size=input_dims),
    num_workers=0,
)

train_patches_loader = DataLoader(
    train_patches_queue,
    batch_size=args.batch_size,
    num_workers=0,  # this must be 0
)

# VALIDATION
eval_dataset = IRCAD_pythae(
    config=dataset_config,
    subset='test',
    transform=tio.Compose(transforms),
)

eval_patches_queue = tio.Queue(
    eval_dataset,
    max_length=100,
    samples_per_volume=20,
    sampler=tio.data.UniformSampler(patch_size=input_dims),
    num_workers=0,
)

eval_patches_loader = DataLoader(
    eval_patches_queue,
    batch_size=args.batch_size,
    num_workers=0,  # this must be 0
)

# MODEL
model_config = VAEConfig(
    input_dim=input_dims,
    latent_dim=8
)

model = VAE(
    model_config=model_config
)

# TRAINING CONFIG
training_config = BaseTrainerConfig(
    output_dir='experiments',
    num_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
)

# WEIGHTS AND BIASES log
wandb_cb = WandbCallback()
wandb_cb.setup(
    training_config=training_config, # training config
    model_config=model_config, # model config
    project_name="my-first-project", # specify your wandb project
)

wandb.config.update({
    "dataset_config": dataset_config.to_dict(),
    "data_transforms": list_torchio_transforms_to_dict(transforms),
    })

callbacks = []
callbacks.append(wandb_cb)


# TRAINING PIEPLINE
pipeline = TrainingPipeline(
    training_config=training_config,
    model=model
)


pipeline(
    train_data=train_patches_loader,
    eval_data=eval_patches_loader,
    callbacks=callbacks # pass the callbacks to the TrainingPipeline and you are done!
)