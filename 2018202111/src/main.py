import os
import torch
from config import *
import torch.nn as nn
from torch import optim
from model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
from model import evaluateInput
from helpers import trimRareWords, trainIters
from data_loader import load_chinese_dialog_pairs, load_english_dialog_pairs

if __name__ == "__main__":
  # Specify corpus
  # Load dialogs and generate voc
  if LANG == "chinese":
    corpus_name = "lccc"
    voc, pairs = load_chinese_dialog_pairs(corpus_name)
  else:
    corpus_name = "cornell movie-dialogs corpus"
    voc, pairs = load_english_dialog_pairs(corpus_name)

  # Trim voc and pairs
  pairs = trimRareWords(voc, pairs, MIN_COUNT)

  loadFilename = os.path.join(
      save_dir, model_name, corpus_name,
      '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
      '{}_checkpoint.tar'.format(checkpoint_iter))

  # Load model if a loadFilename is provided
  if os.path.exists(loadFilename):
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
  else:
    checkpoint = None

  print('Building encoder and decoder ...')

  # Initialize word embeddings
  embedding = nn.Embedding(voc.num_words, hidden_size)
  if os.path.exists(loadFilename):
    embedding.load_state_dict(embedding_sd)
  # Initialize encoder & decoder models
  encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
  decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size,
                                voc.num_words, decoder_n_layers, dropout)
  if os.path.exists(loadFilename):
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
  # Use appropriate device
  encoder = encoder.to(device)
  decoder = decoder.to(device)
  print('Models built and ready to go!')

  # Ensure dropout layers are in train mode
  encoder.train()
  decoder.train()

  # Initialize optimizers
  print('Building optimizers ...')
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adam(decoder.parameters(),
                                 lr=learning_rate * decoder_learning_ratio)
  if os.path.exists(loadFilename):
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

  # If you have cuda, configure cuda to call
  for state in encoder_optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        state[k] = v.cuda()

  # Run training iterations
  print("Starting Training!")
  trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
             decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
             hidden_size, checkpoint, save_dir, n_iteration, batch_size,
             print_every, save_every, clip, corpus_name, loadFilename)

  # Initialize search module
  searcher = GreedySearchDecoder(encoder, decoder)

  # Begin chatting (uncomment and run the following line to begin)
  evaluateInput(encoder, decoder, searcher, voc)
