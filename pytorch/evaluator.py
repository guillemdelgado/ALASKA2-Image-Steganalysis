from utils.utils import seed_everything
seed_everything()
import warnings
warnings.filterwarnings("ignore")
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
import glob

from model.network import Net
from utils.data_augmentation import get_transforms
from data_loader.dataset_retriever import DatasetSubmissionRetriever

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


AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST = get_transforms()



model = Net(num_classes=nclasses)
if torch.cuda.device_count() > 1 and device == 'cuda':
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model = nn.DataParallel(model).to(device)
checkpoint = torch.load(config["test_config"]["checkpoint"])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_ids = os.listdir(config['test_config']['input_path'])
for i in range(len(test_ids)):
    test_ids[i] = os.path.join(config['test_config']['input_path'], test_ids[i])

dataset = DatasetSubmissionRetriever(
    image_names=np.array(test_ids),
    transforms=AUGMENTATIONS_TEST,
)


data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

result = {'Id': [], 'Label': []}
tk0 = tqdm(data_loader, total=int(len(data_loader)))
for image_names, images in tk0:
    if 'TTA' in config['test_config'] and config['test_config']['TTA']:
        im = images.flip(2)
        outputs = model(im.cuda())
        im = images.flip(3)
        outputs = (0.25 * outputs + 0.25 * model(im.cuda()))
        y_pred = model(images.cuda())
        y_pred = (outputs + 0.5 * y_pred)
    else:
        y_pred = model(images.cuda())
    y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]

    result['Id'].extend(image_names)
    result['Label'].extend(y_pred)

submission = pd.DataFrame(result)
submission.to_csv('submission30.csv', index=False)
submission.head()


