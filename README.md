# wh-questions-lm

## Set up virtual environment
To create a virtual environment with all the necessary packages, run `conda env create -f env.yml`.

Then activate with `conda activate cs224nproject`.

## Create a new dataset train/validation/test split
This is not strictly necessary, since the dataset splits already exist, but if you want to create a new split, run `python split_dataset.py`.

## Finetune model on dataset

## How to view run on Tensorboard

1. Start running the experiment on the VM
2. In another window, log into the VM `ssh -L 16006:127.0.0.1:6006 [username]@[host]`
  
  Then do the following:
  ```
  conda activate cs224nproject
  cd wh-questions-lm
  tensorboard --logdir=runs
  ```
3. In your browser, go to http://127.0.0.1:16006
