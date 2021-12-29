# A simple chatbot based on pytorch

Creating a simple Python chatbot using sequence to sequence model.

## Example
```
$ python main.py
Start preparing training data ...
Reading lines...
Read 221282 sentence pairs
Trimmed to 184550 sentence pairs
Counting words...
Counted words: 40151
keep_words 21855 / 40148 = 0.5444
Trimmed from 184550 pairs to 163084, 0.8837 of total
Building encoder and decoder ...
Models built and ready to go!
Building optimizers ...
Starting Training!
Initializing ...
Training...
> hello
Bot: hello miss daniels
> where am I ?
Bot: you don t know . . . . . . . . . . . . . . . . . . .
> who are you ?
Bot: i m a serious teacher . . . . of the federation . the name of the federation . . . .
> are you my friend ?
Bot: yes . . . . . . cuba . . . . . me . mostly .
```

## Get Started

+ **Install** Python3 and PyTorch

+ **Extract dataset.** run `unzip lccc_dataset.zip` and `unzip cornell_dataset.zip` in `data` folder

+ **Modify parameters in `config.py`**:
  + `LANG` can be either `chinese` or `english`
  + `checkpoint_iter` specify the iteration to start from
  + `n_iteration` specify the target iteration numbers
  + Training iteration will start at `checkpoint_iter + 1` and end at `n_iteration`.

+ **Load pretrained model**:
  + Download [save.zip](https://pan.baidu.com/s/1t1ZIw-cXd4tzth9ei2UOLQ) (fj2u)
  + Run `unzip save.zip` in `data` folder
  + With pretrained model
    + if `LANG` is `chinese`, `checkpoint_iter` should be 230000
    + else `checkpoint_iter` should be 85000.

+ **Run the chatbot.** run `python3 main.py`

## Dataset
The datasets we used in this project are truncated ones, full dataset can be fetched from following link:

+ **Dataset for Chinese**: [A Large-scale Chinese Short-Text Conversation Dataset and Chinese pre-training dialog models](https://github.com/thu-coai/CDial-GPT)

+ **Dataset for Englist**: [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
