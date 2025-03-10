import warnings
warnings.filterwarnings("ignore")
import time
import torch
import os
from datetime import datetime
import glob
import torch.nn as nn
from model.regularization import LabelSmoothing
from utils.metrics import AverageMeter, RocAucMeter
from tqdm import tqdm
from shutil import copyfile
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


class Fitter:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        if "output_path" in self.config["output_writer"]:
            self.base_dir = self.config["output_writer"]["output_path"]
            os.makedirs(self.base_dir, exist_ok=True)
        else:
            self.base_dir = './'
        copyfile(self.config['config_json'], self.base_dir+"/"+os.path.basename(config['config_json']))
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.device = device

        if torch.cuda.device_count() > 1 and device == 'cuda':
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        self.model = nn.DataParallel(model).to(device)

        if "metrics" in self.config["train_config"] and self.config["train_config"]["metrics"]:
            self.metrics = True
        else:
            self.metrics = False


        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["train_config"]["lr"])
        if "reduceLROnPlateau" in self.config["train_config"]:
            scheduler_params = dict(
                mode='min',
                factor=0.5,
                patience=1,
                verbose=False,
                threshold=0.0001,
                threshold_mode='abs',
                cooldown=0,
                min_lr=1e-8,
                eps=1e-08
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)

        if "finetune_checkpoint" in config:
            self.load(config["finetune_checkpoint"])

        #TODO: Set this on dataparallel?
        self.criterion = LabelSmoothing().to(self.device)

    def fit(self, train_loader, validation_loader):
        for e in range(self.config["train_config"]["epochs"]):
            if self.config["train_config"]["verbose"]:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, final_scores = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f},'
                f' final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            if validation_loader is not None:
                t = time.time()
                summary_loss, final_scores = self.validation(validation_loader)

                self.log(
                    f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f},'
                    f' final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')
                if summary_loss.avg < self.best_summary_loss:
                    self.best_summary_loss = summary_loss.avg
                    self.model.eval()
                    self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

                if self.config["train_config"]["validation_scheduler"]:
                    self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        gt = None
        pred = None
        for step, (images, targets) in enumerate(val_loader):
            if self.config["train_config"]["verbose"]:
                if step % self.config["train_config"]["verbose_step"] == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                #TODO: Make this parallel?
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                # Make confusion matrix
                if self.metrics:
                    if gt is None and pred is None:
                        gt = targets.cpu().numpy()
                        pred = outputs.cpu().numpy()
                    else:
                        gt = np.concatenate((gt, targets.cpu().numpy()))
                        pred = np.concatenate((pred, outputs.cpu().numpy()))

                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)
        if self.metrics:

            report = classification_report(np.argmax(gt, axis=1), np.argmax(pred, axis=1),
                                           labels=list(range(self.config["train_config"]["nclasses"])))
            cm = confusion_matrix(np.argmax(gt, axis=1), np.argmax(pred, axis=1),
                                  labels=list(range(self.config["train_config"]["nclasses"])))
            print(report)
            print(cm)
        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        step = 0
        for images, targets in tk0:
        #for step, (images, targets) in enumerate(train_loader):

            if self.config["train_config"]["verbose"]:
                if step % self.config["train_config"]["verbose_step"] == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\n'
                    )
            step += 1
            # TODO: Make this parallel?
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config["train_config"]["step_scheduler"]:
                self.scheduler.step()

        return summary_loss, final_scores

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config["train_config"]["verbose"]:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')