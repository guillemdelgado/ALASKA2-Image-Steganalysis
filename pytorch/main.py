from utils.utils import seed_everything
seed_everything()
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from albumentations.pytorch import ToTensor
from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F

from data_loader.alaska import Alaska
from data_loader.generator import Alaska2Dataset
from model.network import Net
from utils.metrics import alaska_weighted_auc
from utils.data_augmentation import get_transforms
from data_loader.dataset_retriever import DatasetRetriever
from trainer.fitter import Fitter

import json



config_json = "./config/baseline.json"
with open(config_json) as f:
  config = json.load(f)
config['config_json'] = config_json
PATH = config["train_reader"]["input_path"]

IMG_SIZE = 512
if "ratio" in config["train_config"]:
    train_val_ratio = config["train_config"]["ratio"]
else:
    train_val_ratio = 0.9
color_mode = "RGB"
mode = "jpegfactor"
fold_number = config["train_config"]["fold"]
nclasses = config["train_config"]["nclasses"]
mode = config["train_config"]["mode"] if "mode" in config["train_config"] else None
multiclass_df = config["train_config"]["multiclass_df"] if "multiclass_df" in config["train_config"] else None
device = 'cuda'

alaska_data = Alaska(PATH, train_val_ratio, mode, multiclass_file=multiclass_df)
#alaska_data = Alaska(PATH, train_val_ratio, mode, multiclass_file='./multiclass_stega_df_ricard.csv')

dataset = alaska_data.build_kfold(5)
AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_transforms()


train_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] != fold_number].kind.values,
    image_names=dataset[dataset['fold'] != fold_number].image_name.values,
    labels=dataset[dataset['fold'] != fold_number].label.values,
    root_path=config["train_reader"]["input_path"],
    nclasses=nclasses,
    transforms=AUGMENTATIONS_TRAIN
)

validation_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] == fold_number].kind.values,
    image_names=dataset[dataset['fold'] == fold_number].image_name.values,
    labels=dataset[dataset['fold'] == fold_number].label.values,
    transforms=AUGMENTATIONS_TEST,
    root_path=config["train_reader"]["input_path"],
    nclasses=nclasses
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),
    batch_size=config["train_config"]["batch_size"],
    pin_memory=False,
    drop_last=True,
    num_workers=config["train_config"]["num_workers"],
)
val_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=config["train_config"]["batch_size"],
    num_workers=config["train_config"]["num_workers"],
    shuffle=False,
    sampler=SequentialSampler(validation_dataset),
    pin_memory=False,
)


model = Net(num_classes=nclasses)
fitter = Fitter(model=model, device=device, config=config)
fitter.fit(train_loader, val_loader)


