"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

We suggest not changing anything in this file.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from scipy.stats import wasserstein_distance
from transformers import BertTokenizerFast

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset,
                    learning_rate, max_epochs, batch_size, experiment_name):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.ckpt_path = experiment_name + '_params'

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        if self.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            logger.info("saving %s", self.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.ckpt_path)

    def train(self):
        model = self.model

        # create the loss function and optimizer
        loss_function = torch.nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.AdamW(params=model.parameters(), lr=self.learning_rate)

        def run_epoch(epoch, split):
            is_train = split == 'train'
            model.train(is_train)
            dataset = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(dataset, batch_size=self.batch_size)

            losses = []
            outputs = []
            all_labels = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, data in pbar:
                # place data on the correct device
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device)
                all_labels.extend(labels)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    output = model(input_ids, attention_mask)
                    outputs.extend(output)

                    logits = torch.log(output)
                    loss = loss_function(logits, labels)
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}.")

            writer.add_scalar("Loss/train", np.mean(losses), epoch)

            if not is_train:
                # sanity check
                assert(len(outputs) == len(all_labels))

                tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
                distances = []

                for i in range(len(outputs)):
                    distances.append(wasserstein_distance(outputs[i].cpu(), all_labels[i].cpu()))

                logger.info("test loss: %f", np.mean(losses))
                logger.info("Wasserstein distance: %f", np.mean(distances))

                writer.add_scalar("Loss/validation", np.mean(losses), epoch)
                writer.add_scalar("Distance", np.mean(distances), epoch)

        for epoch in range(self.max_epochs):
            run_epoch(epoch, 'train')
            if self.test_dataset is not None:
                run_epoch(epoch, 'test')

            self.save_checkpoint()
            
        writer.flush()
        writer.close()
