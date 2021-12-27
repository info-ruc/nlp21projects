'''
Author: your name
Date: 2021-12-16 10:27:45
LastEditTime: 2021-12-20 12:15:12
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/preprocess/finetune.py
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

def finetune_model(json_filelist, tokenizer, opt, sentence_model, sequence_model_encoder, logger):
    """load BERT Model and call fit() method later.

    Args:
        json_filelist (list(str)): absolute path of dialog files with json type
        opt: parsers parameters
    """
    logger.info("finetune: loading Bert Model from opt.bert_dir...")
    multi_gpu = False
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
    model = FinetuneModel(tokenizer, sentence_model, sequence_model_encoder, logger)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("trainable parameters: {0}".format(name))
    model = model.to(device)
    #model = torch.nn.DataParallel(model)   #? 这条语句会拖慢模型加载速度
    finetune_fit(model, json_filelist, opt, logger) # 开始训练


def finetune_fit(model, file_list, opt, logger):
    logger.info("finetune: construct FinetuneDataset...")
    train_dataset = FinetuneDataset(file_list, logger)
    logger.info("finetune: construct Dataloader...")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=opt.finetune_batch_size, 
        shuffle=True,
        num_workers=8
    )
    optimizer = AdamW(model.parameters(), lr=opt.finetune_learning_rate)
    t_total = int(len(train_dataset) * opt.finetune_epochs // opt.finetune_batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total) #学习率调整策略
    for epoch in range(opt.finetune_epochs):
        print("\nEpoch ", epoch + 1, "/", opt.finetune_epochs)
        #logger.write("Epoch " + str(epoch + 1) + "/" + str(opt.pretrain_epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data, opt) # 过模型, 取loss
            loss = loss.mean()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step() # 更新模型参数
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                opt.finetune_learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=opt.finetune_learning_rate, loss=loss.item())
            avg_loss += loss.item()
        cnt = len(train_dataset) // opt.finetune_batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        if (epoch+1) % 10 == 0:
            if not os.path.exists(opt.finetune_output_modeldir):
                os.mkdir(opt.finetune_output_modeldir)
            checkpoint = {
                'epoch': epoch+1,
                'optim_state': optimizer.state_dict(),
                'sentence_model': model.sentence_model.state_dict(),
                'sequence_model_encoder': model.sequence_model_encoder.state_dict(),
                'transformer_encoder': model.transformer_encoder.state_dict(),
                'transformer_decoder': model.transformer_decoder.state_dict(),
                'output_mlp': model.output_mlp.state_dict()
            }
            torch.save(checkpoint, os.path.join(opt.finetune_output_modeldir, "checkpoint{0}.pth".format(str(epoch+1))))
            logger.info("finetune: model saved! epoch = {0}".format(str(epoch+1)))
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