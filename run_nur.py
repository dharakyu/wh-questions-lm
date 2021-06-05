import pandas as pd
import ast
from transformers import BertTokenizer
from datasets import Dataset
import argparse
import random

parser = argparse.ArgumentParser(
		description='Run ...')
parser.add_argument('--train_data_path', dest='train_data_path')
parser.add_argument('--eval_data_path', dest='eval_data_path')
parser.add_argument('--model_name', dest='model_name', default='model')
parser.add_argument('--checkpoints_dir', dest='checkpoints_dir', default='checkpoints')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=2)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=5e-05)
parser.add_argument('--save_steps', dest='save_steps', type=int, default=2000)
opt = parser.parse_args()
print(opt)

K_CANDIDATES = 4
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
batch_size = opt.batch_size

train_df = pd.read_csv(opt.train_data_path)
train_labels = []
ending0 = []
ending1 = []
ending2 = []
ending3 = []
train_df['context'] = ["".join(ast.literal_eval(item)) for item in train_df['context']]
train_df['candidate_responses'] = [ast.literal_eval(item)[:K_CANDIDATES] for item in train_df['candidate_responses']]
for response in train_df['candidate_responses']:
	shuffled_indices = list(range(K_CANDIDATES))
	random.shuffle(shuffled_indices)
	label = shuffled_indices.index(0)
	train_labels.append(label)
	ending0.append(response[shuffled_indices[0]])
	ending1.append(response[shuffled_indices[1]])
	ending2.append(response[shuffled_indices[2]])
	ending3.append(response[shuffled_indices[3]])

train_data = {'context': train_df['context'],
		'utterance': train_df['utterance'],
		'ending0': ending0,
		'ending1': ending1,
		'ending2': ending2,
		'ending3': ending3,
		'label': train_labels
		}

new_train_df = pd.DataFrame(train_data)
new_train_df = new_train_df.sample(frac=1)

eval_df = pd.read_csv(opt.eval_data_path)
eval_labels = []
ending0 = []
ending1 = []
ending2 = []
ending3 = []
eval_df['context'] = ["".join(ast.literal_eval(item)) for item in eval_df['context']]
eval_df['candidate_responses'] = [ast.literal_eval(item)[:K_CANDIDATES] for item in eval_df['candidate_responses']]
for response in eval_df['candidate_responses']:
	shuffled_indices = list(range(K_CANDIDATES))
	random.shuffle(shuffled_indices)
	label = shuffled_indices.index(0)
	eval_labels.append(label)
	ending0.append(response[shuffled_indices[0]])
	ending1.append(response[shuffled_indices[1]])
	ending2.append(response[shuffled_indices[2]])
	ending3.append(response[shuffled_indices[3]])

eval_data = {'context': eval_df['context'],
		'utterance': eval_df['utterance'],
		'ending0': ending0,
		'ending1': ending1,
		'ending2': ending2,
		'ending3': ending3,
		'label': eval_labels
		}

new_eval_df = pd.DataFrame(eval_data)

def show_one(idx):
	context_as_string = ("".join([item.strip() for item in df['context'][idx]])).strip()
	utterance = df["utterance"][idx].strip()
	print('CONTEXT:', context_as_string)
	print('UTTERANCE:', utterance)
	for j in range(K_CANDIDATES):
		option = df['candidate_responses'][idx][j].strip()
		print('OPTION', j, ':', option)


ending_names = ["ending0", "ending1", "ending2", "ending3"]

def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * 4 for context in examples["context"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["utterance"]
    second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
    
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    print(first_sentences)
    print(second_sentences)
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

train_features = Dataset.from_pandas(new_train_df)
print(train_features)
train_dataset = train_features.map(preprocess_function, batched=True)

eval_features = Dataset.from_pandas(new_eval_df)
print(eval_features)
eval_dataset = eval_features.map(preprocess_function, batched=True)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

accepted_keys = ["input_ids", "attention_mask", "label"]
features = [{k: v for k, v in train_dataset[i].items() if k in accepted_keys} for i in range(10)]
batch = DataCollatorForMultipleChoice(tokenizer)(features)

from transformers import BertForMultipleChoice, TrainingArguments, Trainer

model = BertForMultipleChoice.from_pretrained('bert-base-uncased')

args = TrainingArguments(
    output_dir=opt.checkpoints_dir,
    overwrite_output_dir=True,
    save_total_limit=1,
    evaluation_strategy = "steps",
    save_strategy='steps',
    logging_strategy='steps',
    save_steps=opt.save_steps,
    eval_steps=opt.save_steps,
    learning_rate=opt.learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=opt.num_epochs,
    weight_decay=0.01,
)

import numpy as np

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

bert_only = model.bert
bert_only.save_pretrained(opt.model_name)

