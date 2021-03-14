import pandas as pd
import os
from utils import *

"""


def main():
	corpus_df = pd.read_csv('corpus_data/lm_data_cleaned.csv')
	tiny_df = pd.read_csv('datasets/wh-questions-2/tiny_db.csv', sep='\t')
	#print(tiny_df.head())
	for tgrep_id in list(tiny_df['TGrepID']):
		#print(corpus_df[corpus_df['tgrep_id'] == tgrep_id].iloc[0])
		context = corpus_df[corpus_df['tgrep_id'] == tgrep_id].iloc[0]['PreceedingContext']
		cleaned_context = format_context(context)
		print(context)
		print(cleaned_context)
		print('---------')
"""


def main():
	filenames = ['tiny', 'train', 'valid', 'test', 'all']
	#filenames = ['tiny']
	corpus_path = 'corpus_data/lm_data_cleaned.csv'
	corpus_df = pd.read_csv(corpus_path)
	for file in filenames:
		input_path = 'datasets/wh-questions-2/' + file + '_db.csv'
		df = pd.read_csv(input_path, sep='\t')

		contexts = []
		for tgrep_id in list(df['TGrepID']):
			context = corpus_df[corpus_df['tgrep_id'] == tgrep_id].iloc[0]['PreceedingContext']
			cleaned_context = format_context(context)
			contexts.append(cleaned_context)
		df['PrecedingContext'] = contexts
		df.to_csv('datasets/wh-questions-question-context/' + file + '_db.csv', sep='\t')



if __name__ == '__main__':
    main()