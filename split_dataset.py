import argparse
import logging
import math
import random
import re

from utils import mkdir_p

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

WH_WORDS = ['who', 'what', 'when', 'where', 'why', 'how']

def split_train_valid_test(seed_num, save_path, ratio=0.7, input='./corpus_data/lm_data.csv',
                     verbose=True):
    """Split the corpus into training and test sets with a given split ratio

    Arguments:
    seed_num -- the random seed we want to use
    save_path -- where we store the new training/test files
    ratio -- split ratio
    input -- path to the corpus that we want to split
    verbose -- if true, will print message to screen
    """
    if verbose:
        print(f"Spit data into training/test sets with split ratio={ratio}\n=====================")
        print(f"Using random seed {seed_num}, file loaded from {input}")

    # set random seed
    random.seed(seed_num)
    # read in the file
    input_df = pd.read_csv(input, sep=',', low_memory=False)

    # map from tgrep id to item
    ratings_dict = {}
    modal_present_dict = {}
    wh_dict = {}
    question_dict = {}

    for unique_id in input_df['tgrep_id'].unique():

        curr_df = input_df[input_df['tgrep_id'] == unique_id]
        question_dict[unique_id] = curr_df['Question'].tolist()[0]
        #print(sentence_dict[unique_id])
        paraphrases_to_rating = curr_df[['paraphrase', 'rating']].groupby('paraphrase')['rating'].apply(list).to_dict()
        paraphrases_to_modal_present = curr_df[['paraphrase', 'ModalPresent']].groupby('paraphrase')['ModalPresent'].apply(list).to_dict()
        paraphrases_to_wh = curr_df[['paraphrase', 'Wh']].groupby('paraphrase')['Wh'].apply(list).to_dict()
        #print(paraphrase_to_rating)
        ratings_dict[unique_id] = paraphrases_to_rating
        modal_present_dict[unique_id] = paraphrases_to_modal_present
        wh_dict[unique_id] = paraphrases_to_wh

    if verbose:
        print(f"New files can be found in this directory: {save_path}")
        print(f"Out of total {total_num_examples} entries, {train_num_examples} will be in training"
            + f" set and {test_num_examples} will be in test set.\n=====================")

    count_every = 0
    count_a = 0
    count_the = 0
    count_other = 0

    tgrep_ids = []
    questions = []
    every_avgs = []
    a_avgs = []
    the_avgs = []
    other_avgs = []
    wh_words = []
    modal_presents = []

    for (k, values_dict) in ratings_dict.items():
        tgrep_ids.append(k)
        cleaned_question = re.sub(r'\*\S*', '', question_dict[k])
        cleaned_question = cleaned_question.replace('0', '')
        questions.append(cleaned_question)
        print(cleaned_question)

        for item in values_dict:
            if " every " in item:
                every_avgs.append(np.mean(np.array(values_dict[item])))
                count_every += 1
            elif " a " in item:
                a_avgs.append(np.mean(np.array(values_dict[item])))
                count_a += 1
            elif " the " in item:
                the_avgs.append(np.mean(np.array(values_dict[item])))
                count_the += 1
            else:
                other_avgs.append(np.mean(np.array(values_dict[item])))
                count_other += 1

            wh_word = wh_dict[k][item][0]
            modal_present = 1 if modal_present_dict[k][item][0] == 'yes' else 0

        wh_words.append(wh_word)
        modal_presents.append(modal_present)

    all_output_df = pd.DataFrame(data=np.column_stack([tgrep_ids, questions, every_avgs, a_avgs, the_avgs, other_avgs, wh_words, modal_presents]),
                                    columns=["TGrepID", "Question", "Every", "A", "The", "Other", "Wh", "ModalPresent"])

    all_train_df, test_df = train_test_split(all_output_df, test_size=0.3)
    train_df, valid_df = train_test_split(all_train_df, test_size=0.2)

    mkdir_p(save_path)
    
    all_output_df.to_csv(save_path + '/all_db.csv', sep='\t')
    train_df.to_csv(save_path + '/train_db.csv', sep='\t')
    train_df.head(10).to_csv(save_path + '/tiny_db.csv', sep='\t')
    valid_df.to_csv(save_path + '/valid_db.csv', sep='\t')
    test_df.to_csv(save_path + '/test_db.csv', sep='\t')

def main():
    parser = argparse.ArgumentParser(
        description="Creating data splits ...")
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--path", dest="path", type=str, default="./datasets/wh-questions-2")
    parser.add_argument("--ratio", dest="ratio", type=float, default=0.7)
    parser.add_argument("--file", dest="input", type=str,
        default="./corpus_data/lm_data.csv")
    parser.add_argument("--verbose", dest="verbose", action='store_true')
    opt = parser.parse_args()
    split_train_valid_test(opt.seed, opt.path, opt.ratio, opt.input, opt.verbose)

if __name__ == '__main__':
    main()
