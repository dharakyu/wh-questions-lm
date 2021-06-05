import pandas as pd
import ast
from transformers import BertTokenizer
from datasets import Dataset
import argparse

parser = argparse.ArgumentParser(
		description='Run ...')
parser.add_argument('--train_data_path', dest='train_data_path')
parser.add_argument('--eval_data_path', dest='eval_data_path')
parser.add_argument('--model_name', dest='model_name', default='model')
parser.add_argument('--checkpoints_dir', dest='checkpoints_dir', default='checkpoints')
#parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=2)
#parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-04)
#parser.add_argument('--train_data_reader_batch_size', dest='train_data_reader_batch_size', type=int, default=6)
#parser.add_argument('--eval_data_reader_batch_size', dest='eval_data_reader_batch_size', type=int, default=5)
opt = parser.parse_args()
print(opt)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
#batch_size = opt.batch_size

train_df = pd.read_csv(opt.train_data_path, usecols=['utterance'])
train_dataset = Dataset.from_pandas(train_df)

eval_df = pd.read_csv(opt.eval_data_path, usecols=['utterance'])
eval_dataset = Dataset.from_pandas(eval_df)

def tokenize_function(examples):
    exs = []
    for ex in examples["utterance"]:
        if isinstance(ex, str):
            exs.append(ex)
    return tokenizer(exs)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["utterance"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["utterance"])

block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_train_dataset = tokenized_train_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
)

lm_eval_dataset = tokenized_eval_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
)

from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir=opt.checkpoints_dir,
    overwrite_output_dir=True,
    evaluation_strategy = "epoch",
    logging_strategy='epoch',
    save_total_limit=1,
    save_strategy='epoch',
    learning_rate=opt.learning_rate,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_train_dataset,
    data_collator=data_collator,
)

trainer.train()

bert_only = model.bert
bert_only.save_pretrained(opt.model_name)
