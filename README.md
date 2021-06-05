# wh-questions-lm

## Set up virtual environment
To create a virtual environment with all the necessary packages, run `conda env create -f env.yml`.

Then activate with `conda activate cs224nproject`.

## Create a new dataset train/validation/test split
This is not strictly necessary, since the dataset splits already exist, but if you want to create a new split, run `python split_dataset.py`.

## Pretrain model on conversational dataset
Run `python run_nor.py --help` or `python run_mlm.py --help` to see options for performing model pretraining.

Warning: unless you have a massive GPU, use a very small batch size for training (batch size = 2 should work).

## Finetune model on dataset
Run `python run.py --help` to see all options.

`--mode {train,eval}` train model or evaluate an existing model.

`--experiment_name EXPERIMENT_NAME` experiment name which will be used to save parameters from trained model

`--num_epochs NUM_EPOCHS` number of epochs for which to train

`--learning_rate LEARNING_RATE` learning rate for training

`--batch_size BATCH_SIZE` batch size for training and validation

`--model {distilbert,bert}` which BERT variant to use (use DistilBERT for faster training and evaluation)

`--use_context` flag to determine if training or evaluation should incorporate preceding contextual information

`--path_to_datasets PATH_TO_DATASETS` path to train/validation datasets for training (only needed for training)

`--path_to_params PATH_TO_PARAMS` path to trained model parameters (only needed for evaluation)

`--eval_dataset {test,valid}` 

`--write_to_file` flag to determine if evaluated outputs are written to a file in `outputs_for_analysis`

### How to view run on Tensorboard

1. Start running the experiment on the VM
2. In another window, log into the VM `ssh -L 16006:127.0.0.1:6006 [username]@[host]`
  
  Then do the following:
  ```
  conda activate cs224nproject
  cd wh-questions-lm
  tensorboard --logdir=runs
  ```
3. In your browser, go to http://127.0.0.1:16006
