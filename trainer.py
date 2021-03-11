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
from transformers import DistilBertTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#LEARNING_RATE = 1e-05
#MAX_EPOCHS = 2
#BATCH_SIZE = 32

class Trainer:

    def __init__(self, model, train_dataset, test_dataset,
                    learning_rate, max_epochs, experiment_name):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = 32
        self.ckpt_path = experiment_name + '_params'
        #self.log_softmax = torch.nn.LogSoftmax(dim=1)

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

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            dataset = self.train_dataset if is_train else self.test_dataset
            #print(len(dataset))
            loader = DataLoader(dataset, batch_size=self.batch_size)
            #print(BATCH_SIZE)
            #print(len(loader))

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            #print(enumerate(loader))
            for it, data in pbar:
                #thing = thing.to(self.device)
                #print(thing)
                # place data on the correct device
                #x = x.to(self.device)
                #y = y.to(self.device)
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                labels = data['labels'].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    output = model(input_ids, attention_mask)
                    logits = torch.log(output)
                    loss = loss_function(logits, labels)
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    '''
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    '''

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}.")

            if not is_train:
                tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
                distances = []
                for index, item in enumerate(data['input_ids']):
                    tokens = tokenizer.convert_ids_to_tokens(item)
                    #print(tokenizer.convert_tokens_to_string(tokens))
                    #print(output[index])
                    #print(labels[index])
                    distances.append(wasserstein_distance(output[index], labels[index]))
                #distance = wasserstein_distance(output, labels)
                logger.info("test loss: %f", np.mean(losses))
                logger.info("Wasserstein distance: %f", np.mean(distances))
                #print("test loss: %f", np.mean(losses))
                #print("mean of Wasserstein distances: %f", np.mean(distances))

        #self.tokens = 0 # counter used for learning rate decay

        for epoch in range(self.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                run_epoch('test')

            self.save_checkpoint()
