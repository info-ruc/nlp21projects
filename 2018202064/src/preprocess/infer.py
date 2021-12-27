'''
Author: your name
Date: 2021-12-19 11:28:42
LastEditTime: 2021-12-22 08:20:50
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/preprocess/infer.py
'''
import torch
from torch import optim
from torch._C import device
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from seq2seq.models.BertForPretrain import *
from seq2seq.dataset.dialogDatasets import *
from seq2seq.models.FinetuneModel import *
from seq2seq.models.InferModel import *

def infer_model(json_filelist, tokenizer, opt, sentence_model, sequence_model_encoder, transformer_encoder, transformer_decoder, output_mlp, logger):
    """load finetune Model and call fit() method later.
    """
    logger.info("infer: construct InferModel")
    multi_gpu = False
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
    model = InferModel(json_filelist, tokenizer, opt, sentence_model, sequence_model_encoder, transformer_encoder, transformer_decoder, output_mlp, logger)
    model = model.to(device)
    #model = torch.nn.DataParallel(model)   #? 这条语句会拖慢模型加载速度
    infer_fit(model, json_filelist, opt, logger) # 开始训练


def infer_fit(model, file_list, opt, logger):
    logger.info("infer: construct FinetuneDataset...")
    train_dataset = FinetuneDataset(file_list, logger)
    logger.info("infer: construct Dataloader...")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=opt.infer_batch_size, 
        shuffle=True,
        num_workers=8
    )
    model.eval()
    for i, training_data in enumerate(train_dataloader):
        response_list = train_step(model, training_data, opt) # 过模型, 取loss
        return          #! 能跑通了再把这个删了
    
#! pretrain_fit()函数内每一步求loss
def train_step(model, train_data, opt):
    """calculate Loss in every epoch of pretrain_fit()

    Args:
        model (Bert_model): the model which calls forward() method
        train_data (dict): trained data treated as parameters in model.forward()
        opt (options): parameters of parsers

    Returns:
        loss: loss
    """
    multi_gpu = False
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    loss = model.forward(train_data)
    return loss