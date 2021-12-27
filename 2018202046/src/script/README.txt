### usage
python buid_graph.py
python finetune_bert.py
python train_bert_gcn.py

### directory structure
.
├── README.txt                      # readme
├── build_graph.py                  # buid grah
├── checkpoint
│   └── roberta-base_gcn_20ng
│       └── training.log            # training log
├── data                            # data
│   ├── 20ng.txt
│   └── corpus
│       ├── 20ng.clean.txt
│       └── 20ng.txt
├── finetune_bert.py                # finetune bert
├── model                           # model
│   ├── __init__.py
│   ├── graphconv_edge_weight.py
│   ├── models.py
│   └── torch_gcn.py
├── prepare_data.py                 # prepare data
├── train_bert_gcn.py               # train bert gcn
└── utils                           # utils
    ├── __init__.py
    └── utils.py