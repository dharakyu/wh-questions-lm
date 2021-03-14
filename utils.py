import os
import errno
import pandas as pd
import re

def mkdir_p(path):
	"""Create a directory if not exist"""
	try:
		os.makedirs(path)
	except OSError as exc:
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise
	return

def clean_sentence(sentence):
	"""Get rid of trace characters"""
	cleaned_sentence = re.sub(r'\*\S*', '', sentence)
	cleaned_sentence = cleaned_sentence.replace('0', '')
	return cleaned_sentence

def format_context(dialogue):
	"""Format a context dialogue, indicating alternating speakers with [SEP] token"""
	#cleaned_dialogue = re.sub(r'speaker[a-z][0-9][0-9]?[0-9]?\.', ' [SEP] ', dialogue)
	cleaned_dialogue = re.sub(r'speaker[a-z][0-9][0-9]?[0-9]?\.', '', dialogue)
	cleaned_dialogue = cleaned_dialogue.replace('###', '')
	cleaned_dialogue = clean_sentence(cleaned_dialogue)

	return cleaned_dialogue

def read_dataset_split_sentence_only(path_to_dataset):
	"""Given a path to a dataset, return the sentence inputs and the probability labels"""
	df = pd.read_csv(path_to_dataset, sep='\t')

	sentences = df['Question'].tolist()

	every_probs = list(df['Every'])
	a_probs = list(df['A'])
	the_probs = list(df['The'])
	other_probs = list(df['Other'])

	labels = [[every_probs[i], a_probs[i], the_probs[i], other_probs[i]] for i in range(len(sentences))]

	return sentences, labels

def read_dataset_split_with_context(path_to_dataset):
	"""Given a path to a dataset, return the complete dialogue inputs and the probability labels"""
	df = pd.read_csv(path_to_dataset, sep='\t')

	contexts = list(df['PrecedingContext'])
	sentences = list(df['Question'])

	dialogues = [sentences[i] + ' [SEP] ' + contexts[i] for i in range(len(contexts))]

	every_probs = list(df['Every'])
	a_probs = list(df['A'])
	the_probs = list(df['The'])
	other_probs = list(df['Other'])

	labels = [[every_probs[i], a_probs[i], the_probs[i], other_probs[i]] for i in range(len(sentences))]

	return dialogues, labels