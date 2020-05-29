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
from albumentations import VerticalFlip, HorizontalFlip, ToFloat, Compose
import torch.nn.functional as F

from data_loader.alaska import Alaska
from data_loader.generator import Alaska2Dataset
from model.network import Net
from utils.metrics import alaska_weighted_auc

#PATH = "D:\\Data\\alaska2-image-steganalysis\\"
PATH = "/export/home/scratch/rdg/data/"

IMG_SIZE = 512
train_val_ratio = 0.9
batch_size = 8
num_workers = 8
epochs = 20
color_mode = "RGB"
mode = "multiclass"

if mode == "multiclass":
    n_classes = 10
else:
    n_classes = 1

#alaska_data = Alaska(PATH, train_val_ratio, mode, multiclass_file='./multiclass_stega_df.csv')
alaska_data = Alaska(PATH, train_val_ratio, mode, multiclass_file='./multiclass_stega_df_ricard.csv')

data = alaska_data.build()
IMAGE_IDS_train = data[0]
IMAGE_LABELS_train = data[1]
IMAGE_IDS_val = data[2]
IMAGE_LABELS_val = data[3]

AUGMENTATIONS_TRAIN = Compose([
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    ToFloat(max_value=255),
    ToTensor()
], p=1)


AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=255),
    ToTensor()
], p=1)


train_dataset = Alaska2Dataset(IMAGE_IDS_train, IMAGE_LABELS_train, augmentations=AUGMENTATIONS_TRAIN, color_mode=color_mode)
valid_dataset = Alaska2Dataset(IMAGE_IDS_val, IMAGE_LABELS_val, augmentations=AUGMENTATIONS_TEST, color_mode=color_mode)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)
device = 'cuda'

if torch.cuda.device_count() > 1 and device == 'cuda':
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model = Net(num_classes=n_classes)
# pretrained model in my pc. now i will train on all images for 2 epochs
model.load_state_dict(torch.load('./epoch_5_val_loss_7.03_auc_0.844.pth'))
model = nn.DataParallel(model).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

train_loss, val_loss = [], []

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)
    model.train()
    running_loss = 0
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for im, labels in tk0:
        inputs = im["image"].to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tk0.set_postfix(loss=(loss.item()))

    epoch_loss = running_loss / (len(train_loader) / batch_size)
    train_loss.append(epoch_loss)
    print('Training Loss: {:.8f}'.format(epoch_loss))

    tk1 = tqdm(valid_loader, total=int(len(valid_loader)))
    model.eval()
    running_loss = 0
    y, preds = [], []
    with torch.no_grad():
        for (im, labels) in tk1:
            inputs = im["image"].to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            y.extend(labels.cpu().numpy().astype(int))
            preds.extend(F.softmax(outputs, 1).cpu().numpy())
            running_loss += loss.item()
            tk1.set_postfix(loss=(loss.item()))

        epoch_loss = running_loss / (len(valid_loader) / batch_size)
        val_loss.append(epoch_loss)
        preds = np.array(preds)
        # convert multiclass labels to binary class
        y = np.array(y)
        labels = preds.argmax(1)
        for class_label in np.unique(y):
            idx = y == class_label
            acc = (labels[idx] == y[idx]).astype(np.float).mean() * 100
            print('accuracy for class', class_label, 'is', acc)

        acc = (labels == y).mean() * 100
        new_preds = np.zeros((len(preds),))
        temp = preds[labels != 0, 1:]
        new_preds[labels != 0] = temp.sum(1)
        new_preds[labels == 0] = 1 - preds[labels == 0, 0]
        y = np.array(y)
        y[y != 0] = 1
        auc_score = alaska_weighted_auc(y, new_preds)
        print(
            f'Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}')
    torch.save(model.state_dict(),
               f"epoch_{epoch}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}_rgb.pth")

test_ids = os.listdir(os.path.join(PATH, 'Test'))
for i in range(len(test_ids)):
    test_ids[i] = os.path.join(os.path.join(PATH, 'Test'), test_ids[i])


test_dataset = Alaska2Dataset(test_ids, None, augmentations=AUGMENTATIONS_TEST, test=True, color_mode=color_mode)
batch_size = 16
num_workers = 0

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False,
                                          drop_last=False)
model.eval()
preds = []
tk0 = tqdm(test_loader)

with torch.no_grad():
    for i, im in enumerate(tk0):
        inputs = im["image"].to(device)
        # flip vertical
        im = inputs.flip(2)
        outputs = model(im)
        # fliplr
        im = inputs.flip(3)
        outputs = (0.25*outputs + 0.25*model(im))
        outputs = (outputs + 0.5*model(inputs))
        preds.extend(F  .softmax(outputs, 1).cpu().numpy())

preds = np.array(preds)
labels = preds.argmax(1)
new_preds = np.zeros((len(preds),))
new_preds[labels != 0] = preds[labels != 0, 1:].sum(1)
new_preds[labels == 0] = 1 - preds[labels == 0, 0]

test_dataset.data['Id'] = test_dataset.data['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_dataset.data['Label'] = new_preds

test_df = test_dataset.data.drop('ImageFileName', axis=1)
test_df.to_csv('submission_eb0_epoch5_ycbcr.csv', index=False)

