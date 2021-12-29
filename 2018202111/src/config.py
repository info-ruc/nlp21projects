import os
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
save_dir = os.path.join("data", "save")

LANG = 'english'
MAX_LENGTH = 30
MIN_COUNT = 3  # Minimum word count threshold for trimming

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

checkpoint_iter = 85000 # 230000

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 85000
print_every = 500
save_every = 5000

