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

def compute_trivial_baseline():
	val_sentences, val_labels = read_dataset_split('datasets/wh-questions-2/valid_db.csv')
	avg_0 = np.mean([label[0] for label in val_labels])
	avg_1 = np.mean([label[1] for label in val_labels])
	avg_2 = np.mean([label[2] for label in val_labels])
	avg_3 = np.mean([label[3] for label in val_labels])
	#baseline = [avg_0, avg_1, avg_2, avg_3]
	baseline = [0.25, 0.25, 0.25, 0.25]

	distances = []
	for i in range(len(val_labels)):
		distances.append(wasserstein_distance(baseline, val_labels[i]))

	print(baseline)
	print(np.mean(distances))

compute_trivial_baseline()