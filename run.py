import pandas as pd
import logging
from datetime import datetime
import argparse

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from dataset import WhQuestionsDataset
from model import DistilBertForWhQuestionInference
from trainer import Trainer

import numpy as np
from scipy.stats import wasserstein_distance

import torch

def read_dataset_split(path_to_dataset):
	df = pd.read_csv(path_to_dataset, sep='\t')

	sentences = df['Question'].tolist()

	every_probs = list(df['Every'])
	a_probs = list(df['A'])
	the_probs = list(df['The'])
	other_probs = list(df['Other'])

	labels = [[every_probs[i], a_probs[i], the_probs[i], other_probs[i]] for i in range(len(sentences))]

	return sentences, labels

def main():
	parser = argparse.ArgumentParser(
		description='Run ...')
	parser.add_argument('--experiment_name', dest='experiment_name', default='')
	parser.add_argument('--num_epochs', dest='num_epochs', default=2)
	parser.add_argument('--learning_rate', dest='learning_rate', default=1e-05)
	opt = parser.parse_args()
	print(opt)

	if opt.experiment_name == '':
		opt.experiment_name = datetime.now().strftime('%m_%d_%H_%M')

	train_sentences, train_labels = read_dataset_split('datasets/wh-questions-2/train_db.csv')
	val_sentences, val_labels = read_dataset_split('datasets/wh-questions-2/valid_db.csv')
	test_sentences, test_labels = read_dataset_split('datasets/wh-questions-2/test_db.csv')

	tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

	train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
	val_encodings = tokenizer(val_sentences, truncation=True, padding=True)
	test_encodings = tokenizer(test_sentences, truncation=True, padding=True)

	train_dataset = WhQuestionsDataset(train_encodings, train_labels)
	val_dataset = WhQuestionsDataset(val_encodings, val_labels)
	test_dataset = WhQuestionsDataset(test_encodings, test_labels)

	model = DistilBertForWhQuestionInference()

	trainer = Trainer(model, train_dataset, val_dataset, opt.learning_rate, opt.num_epochs,
						opt.experiment_name)
	trainer.train()


if __name__ == '__main__':
    main()