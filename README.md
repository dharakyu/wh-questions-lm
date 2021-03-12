# wh-questions-lm

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
