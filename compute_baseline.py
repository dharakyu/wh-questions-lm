import argparse
import os

import numpy as np
from scipy.stats import wasserstein_distance
from utils import read_dataset_split_sentence_only

def compute_baseline(dataset):
	_, labels = read_dataset_split_sentence_only(os.path.join('datasets', 'wh-questions-2', dataset + '_db.csv'))
	avg_0 = np.mean([label[0] for label in labels])
	avg_1 = np.mean([label[1] for label in labels])
	avg_2 = np.mean([label[2] for label in labels])
	avg_3 = np.mean([label[3] for label in labels])
	baseline = [avg_0, avg_1, avg_2, avg_3]
	#baseline = [0.25, 0.25, 0.25, 0.25]

	distances = []
	for i in range(len(labels)):
		distances.append(wasserstein_distance(baseline, labels[i]))

	print(baseline)
	print(np.mean(distances))

def main():
	parser = argparse.ArgumentParser(
		description='Compute baseline Wasserstein distance values for validation and test datasets ...')
	parser.add_argument('--dataset', dest='dataset', choices=['valid', 'test'], default='valid')
	opt = parser.parse_args()
	print('Computing baseline for', opt.dataset, 'dataset')
	compute_baseline(opt.dataset)

if __name__ == '__main__':
    main()