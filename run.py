import pandas as pd
import logging
from datetime import datetime
import argparse
import os

from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

from dataset import WhQuestionsDataset
from models import DistilBertForWhQuestionInference, BertForWhQuestionInference
from trainer import Trainer

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div

import torch
from torch.utils.data.dataloader import DataLoader

from utils import *

def main():
	parser = argparse.ArgumentParser(
		description='Run ...')
	parser.add_argument('--mode', dest='mode', choices=['train', 'eval'], default='train')
	parser.add_argument('--experiment_name', dest='experiment_name', default='')
	parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=2)
	parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-06)
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
	parser.add_argument('--model', dest='model', choices=['distilbert', 'bert'], default='bert')
	parser.add_argument('--from_pretrained', dest='from_pretrained', default=None)
	parser.add_argument('--use_context', dest='use_context', action='store_true')
	parser.add_argument('--path_to_datasets', dest='path_to_datasets', default=os.path.join('datasets', 'wh-questions-question-context'))
	parser.add_argument('--path_to_params', dest='path_to_params')
	parser.add_argument('--eval_dataset', dest='eval_dataset', choices=['test', 'valid'], default='valid')
	parser.add_argument('--write_to_file', dest='write_to_file', action='store_true')
	parser.add_argument('--num_train_examples', dest='num_train_examples', type=int, default=None)
	opt = parser.parse_args()
	print(opt)

	# initialize model
	if opt.model == 'distilbert':
		model = DistilBertForWhQuestionInference()
	else:
		if opt.from_pretrained:
			model = BertForWhQuestionInference(opt.from_pretrained)
		else:
			model = BertForWhQuestionInference(None)

	# initialize tokenizer
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	if opt.mode == 'train':
		if opt.experiment_name == '':
			opt.experiment_name = datetime.now().strftime('%m_%d_%H_%M')

		train_path = os.path.join(opt.path_to_datasets, 'train_db.csv')
		valid_path = os.path.join(opt.path_to_datasets, 'valid_db.csv')
		test_path = os.path.join(opt.path_to_datasets, 'test_db.csv')

		if opt.use_context:
			if opt.num_train_examples is None:
				train_sentences, train_labels = read_dataset_split_with_context(train_path)
			else:
				train_sentences, train_labels = read_dataset_split_with_context(train_path, opt.num_train_examples)

			val_sentences, val_labels = read_dataset_split_with_context(valid_path)
			test_sentences, test_labels = read_dataset_split_with_context(test_path)
			
		else:
			train_sentences, train_labels = read_dataset_split_sentence_only(train_path)
			val_sentences, val_labels = read_dataset_split_sentence_only(valid_path)
			test_sentences, test_labels = read_dataset_split_sentence_only(test_path)

		print(train_sentences)
		train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
		val_encodings = tokenizer(val_sentences, truncation=True, padding=True)
		test_encodings = tokenizer(test_sentences, truncation=True, padding=True)

		train_dataset = WhQuestionsDataset(train_encodings, train_labels)
		val_dataset = WhQuestionsDataset(val_encodings, val_labels)
		test_dataset = WhQuestionsDataset(test_encodings, test_labels)

		trainer = Trainer(model, train_dataset, val_dataset, opt.learning_rate, opt.num_epochs,
							opt.batch_size, opt.experiment_name)
		trainer.train()
	
	else:
		# load params from a model we already trained
		model.load_state_dict(torch.load(opt.path_to_params, map_location=torch.device('cpu')))

		eval_path = os.path.join(opt.path_to_datasets, opt.eval_dataset + '_db.csv')

		if opt.use_context:
			eval_sentences, eval_labels = read_dataset_split_with_context(eval_path)
		else:
			eval_sentences, eval_labels = read_dataset_split_sentence_only(eval_path)

		eval_encodings = tokenizer(eval_sentences, truncation=True, padding=True)
		eval_dataset = WhQuestionsDataset(eval_encodings, eval_labels)


		# check if we're on a gpu
		if torch.cuda.is_available():
			device = torch.cuda.current_device()
			model = torch.nn.DataParallel(model).to(device)
		else:
			device = torch.device("cpu")

		model.eval()
		loader = DataLoader(eval_dataset, batch_size=opt.batch_size)

		encodings = []
		outputs = []
		all_labels = []
		distances = []
		l2_distances = []
		kl_divergences = []

		kl = torch.nn.KLDivLoss(reduction='batchmean')

		for it, data in enumerate(loader):
			# place data on the correct device
			input_ids = data['input_ids'].to(device)
			encodings.extend(input_ids)

			attention_mask = data['attention_mask'].to(device)
			labels = data['labels'].to(device)
			all_labels.extend(labels)

			with torch.no_grad():
				output = model(input_ids, attention_mask)
				outputs.extend(output)
		
		mkdir_p('outputs_for_analysis')

		rows = []
		sentences = []
		for i in range(len(all_labels)):
			distances.append(wasserstein_distance(outputs[i].cpu(), all_labels[i].cpu()))
			l2_distances.append(np.linalg.norm(outputs[i].cpu() - all_labels[i].cpu()))
			logits = torch.log(outputs[i])
			loss = kl(logits, all_labels[i])
			kl_divergences.append(loss.item())
			tokens = tokenizer.convert_ids_to_tokens(encodings[i])
			sentence = tokenizer.convert_tokens_to_string(tokens)
			sentence = sentence.replace('[CLS]', '')
			sentence = sentence.replace('[PAD]', '')
			sentence = sentence.replace('[SEP]', '')
			sentences.append(sentence)
			rounded_output = str([round(i, 4) for i in outputs[i].tolist()])
			rounded_label = str([round(i, 4) for i in all_labels[i].tolist()])
			row = sentence + '\t' + rounded_output + '\t' + rounded_label + '\t' + str(distances[i]) + '\n'
			rows.append(row)
			print(outputs[i])
			if i % 20 == 0:
				print('sentence:\n', sentence)
				print('prediction:\n', outputs[i])
				print('gold:\n', all_labels[i])
				print('distance:\n', distances[i])
				print('------')

		print('Wasserstein distance:', np.mean(distances))
		print('L2 distance:', np.mean(l2_distances))
		print('KL divergence', np.mean(kl_divergences))
		for i in range(4):
			predictions = torch.Tensor([row[i] for row in outputs]).cpu()
			labels = torch.Tensor([row[i] for row in all_labels]).cpu()
			corr = np.corrcoef(predictions, labels)
			print(corr[0, 1])

		if opt.write_to_file:
			if opt.use_context:
				write_path = os.path.join('outputs_for_analysis', opt.path_to_params + '_with_context_' + opt.eval_dataset + '_db.csv')
			else:
				write_path = os.path.join('outputs_for_analysis', opt.path_to_params + '_' + opt.eval_dataset + '_db.csv')
			f = open(write_path, 'w')
			head_line = "Sentence\tOutput\tLabel\tDistance\n"
			f.write(head_line)
			for row in rows:
				f.write(row)
			f.close()


if __name__ == '__main__':
	main()