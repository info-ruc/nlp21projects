import re
import os
import model
import torch
import data_loader
import unicodedata
import numpy as np
from config import MAX_LENGTH, LANG


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  if LANG == 'chinese':
    return s
  else:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
  if LANG == 'chinese':
    return s
  s = unicodeToAscii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  s = re.sub(r"\s+", r" ", s).strip()
  return s


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
  # Input sequences need to preserve the last word for EOS token
  return len(p[0].split(' ')) < MAX_LENGTH and len(
      p[1].split(' ')) < MAX_LENGTH


# Filter pairs using filterPair condition
def filterPairs(pairs):
  return [pair for pair in pairs if filterPair(pair)]


def trimRareWords(voc, pairs, MIN_COUNT):
  # Trim words used under the MIN_COUNT from the voc
  voc.trim(MIN_COUNT)
  # Filter out pairs with trimmed words
  keep_pairs = []
  for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True
    # Check input sentence
    for word in input_sentence.split(' '):
      if word not in voc.word2index:
        keep_input = False
        break
    # Check output sentence
    for word in output_sentence.split(' '):
      if word not in voc.word2index:
        keep_output = False
        break

    # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
    if keep_input and keep_output:
      keep_pairs.append(pair)

  print("Trimmed from {} pairs to {}, {:.4f} of total".format(
      len(pairs), len(keep_pairs),
      len(keep_pairs) / len(pairs)))
  return keep_pairs


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, encoder_n_layers,
               decoder_n_layers, hidden_size, checkpoint, save_dir,
               n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, loadFilename):
  # Number of iterations in an epoch
  # n_iteration = ceil(len(pairs / batch_size))
  num_pairs = len(pairs)

  # Initializations
  print('Initializing ...')
  start_iteration = 1
  print_loss = 0
  if os.path.exists(loadFilename):
    start_iteration = checkpoint['iteration'] + 1

  # Load batches for each iteration
  training_batches = [
      data_loader.batch2TrainData(
          voc,
          [pairs[(n * batch_size + i) % num_pairs] for i in range(batch_size)])
      for n in range(start_iteration, n_iteration + 1)
  ]

  # Training loop
  print("Training...")
  for iteration in range(start_iteration, n_iteration + 1):
    training_batch = training_batches[iteration - start_iteration]
    # Extract fields from batch
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    # Run a training iteration with batch
    loss = model.train(input_variable, lengths, target_variable, mask,
                       max_target_len, encoder, decoder, embedding,
                       encoder_optimizer, decoder_optimizer, batch_size, clip)
    print_loss += loss

    # Print progress
    if iteration % print_every == 0:
      print_loss_avg = print_loss / print_every
      print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".
            format(iteration, iteration / n_iteration * 100, print_loss_avg))
      print_loss = 0

    # Save checkpoint
    if (iteration % save_every == 0):
      directory = os.path.join(
          save_dir, model_name, corpus_name,
          '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
      if not os.path.exists(directory):
        os.makedirs(directory)
      torch.save(
          {
              'iteration': iteration,
              'en': encoder.state_dict(),
              'de': decoder.state_dict(),
              'en_opt': encoder_optimizer.state_dict(),
              'de_opt': decoder_optimizer.state_dict(),
              'loss': loss,
              'voc_dict': voc.__dict__,
              'embedding': embedding.state_dict()
          },
          os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
