import argparse
import logging
import math
import random
import os
import errno

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

WH_WORDS = ['who', 'what', 'when', 'where', 'why', 'how']

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

def split_train_test(seed_num, save_path, ratio=0.7, input='./corpus_data/lm_data.csv',
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
    input_df = pd.read_csv(input, sep=',')

    # map from tgrep id to item
    ratings_dict = {}
    modal_present_dict = {}
    wh_dict = {}
    sentence_dict = {}

    for unique_id in input_df['tgrep_id'].unique():

        curr_df = input_df[input_df['tgrep_id'] == unique_id]
        sentence_dict[unique_id] = curr_df['Sentence'].tolist()[0]
        #print(sentence_dict[unique_id])
        paraphrases_to_rating = curr_df[['paraphrase', 'rating']].groupby('paraphrase')['rating'].apply(list).to_dict()
        paraphrases_to_modal_present = curr_df[['paraphrase', 'ModalPresent']].groupby('paraphrase')['ModalPresent'].apply(list).to_dict()
        paraphrases_to_wh = curr_df[['paraphrase', 'Wh']].groupby('paraphrase')['Wh'].apply(list).to_dict()
        #print(paraphrase_to_rating)
        ratings_dict[unique_id] = paraphrases_to_rating
        modal_present_dict[unique_id] = paraphrases_to_modal_present
        wh_dict[unique_id] = paraphrases_to_wh

    #print(ratings_dict)
    list_new_value = []

    total_num_examples = len(input_df['tgrep_id'].unique())
    train_num_examples = math.ceil(ratio*len(input_df['tgrep_id'].unique()))
    test_num_examples = total_num_examples - train_num_examples
    
    if verbose:
        print(f"New files can be found in this directory: {save_path}")
        print(f"Out of total {total_num_examples} entries, {train_num_examples} will be in training"
            + f" set and {test_num_examples} will be in test set.\n=====================")

    count_every = 0
    count_a = 0
    count_the = 0
    count_other = 0

    for (k, values_dict) in ratings_dict.items():
        tgrep_id = k
        avg_ratings = [0] * 4
        wh_word = None
        modal_present = None
        sentence = sentence_dict[k]

        for item in values_dict:
            if " every " in item:
                avg_ratings[0] = np.mean(np.array(values_dict[item]))
                count_every += 1
            elif " a " in item:
                avg_ratings[1] = np.mean(np.array(values_dict[item]))
                count_a += 1
            elif " the " in item:
                avg_ratings[2] = np.mean(np.array(values_dict[item]))
                count_the += 1
            else:
                avg_ratings[3] = np.mean(np.array(values_dict[item]))
                count_other += 1

            wh_word = wh_dict[k][item][0]
            modal_present = 1 if modal_present_dict[k][item][0] == 'yes' else 0

        l = tgrep_id + '\t' + sentence + '\t' + str(avg_ratings[0]) + '\t' + str(avg_ratings[1]) + '\t' \
                            + str(avg_ratings[2]) + '\t' + str(avg_ratings[3]) + '\t' \
                            + wh_word + '\t' + str(modal_present)


        list_new_value.append(l)

    # sanity check
    assert total_num_examples == len(list_new_value)

    # shuffle
    ids = list(range(0, total_num_examples))
    random.shuffle(ids)
    train_ids = ids[:train_num_examples]
    test_ids = ids[train_num_examples:]
    mkdir_p(save_path)



    f_all = open(save_path + '/all_db.csv', 'w')
    f = open(save_path + '/train_db.csv', 'w')
    head_line = "TGrepID" + "\t" + "Sentence" + "\t" \
    + "Every" + "\t" + "A" + "\t" + "The" + "\t" + "Other" + "\t" \
    + "Wh" + "\t" + "ModalPresent" + "\n"
    f_all.write(head_line)
    f.write(head_line)
    for i in train_ids:
        f_all.write(list_new_value[i]+"\n")
        f.write(list_new_value[i]+"\n")
    f.close()
    f = open(save_path + '/test_db.csv', 'w')
    f.write(head_line)
    for i in test_ids:
        f_all.write(list_new_value[i]+"\n")
        f.write(list_new_value[i]+"\n")
    f_all.close()
    f.close()


def k_folds_idx(k, total_examples, seed_num):
    """Create K folds

    Arguments:
    k -- number of folds we want
    total_examples -- total number of examples we want to split
    seed_num -- the random seed we want to use

    Return:
    output -- k (train_idx, val_idx) pairs
    """
    all_inds = list(range(total_examples))
    cv = KFold(n_splits=k, shuffle=True, random_state=seed_num)
    output = cv.split(all_inds)
    return output

def main():
    parser = argparse.ArgumentParser(
        description="Creating data splits ...")
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--path", dest="path", type=str, default="./datasets/wh-questions")
    parser.add_argument("--ratio", dest="ratio", type=float, default=0.7)
    parser.add_argument("--file", dest="input", type=str,
        default="./corpus_data/lm_data.csv")
    parser.add_argument("--verbose", dest="verbose", action='store_true')
    opt = parser.parse_args()
    split_train_test(opt.seed, opt.path, opt.ratio, opt.input, opt.verbose)

if __name__ == '__main__':
    main()
