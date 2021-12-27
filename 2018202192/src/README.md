# EmbedKGQA

## Overview of the Model

![model](model.png)

## Instructions

First, build a new environment by running:

```bash
conda create --name <env> --file requirements.txt
```

In order to run the code, download `pretrained_model.zip` from [here](https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing). Unzip it in the `src`directory.

Move to directory `src/scripts/KGQA`. Following is an example command to run the training code:

```bash
python3 main.py --mode train --relation_dim 200 --do_batch_norm 1 --use_cuda 1 --gpu 0 --freeze 1 --batch_size 16 --validate_every 10 --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 --decay 1.0 --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200
```
This will train a model and save it in `checkpoints/roberta_finetune/best_score_model.pt`. Validation will be done during the training process.

Files `log` is an example output file for the the command above.
