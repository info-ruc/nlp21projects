'''
Author: your name
Date: 2021-11-05 11:31:55
LastEditTime: 2021-12-15 07:19:44
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/preprocess/pretrain.py
'''
import torch
from torch import optim
from torch._C import device
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from seq2seq.models.BertForPretrain import *
from seq2seq.dataset.dialogDatasets import *

#! 加载BERT模型
def pretrain_model(file_list, opt, logger):
    """load BERT Model and call fit() method later.

    Args:
        file_list (list(str)): absolute path of dialog files with json type
        opt: parsers parameters
    """
    logger.info("BERT pretrain: loading Bert Model from opt.bert_dir...")
    multi_gpu = False
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
    bert_model = BertModel.from_pretrained(opt.bert_dir)
    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    model = BertForPretrain(bert_model)
    model = model.to(device)
    #model = torch.nn.DataParallel(model)   #? 这条语句会拖慢模型加载速度
    pretrain_fit(model, file_list, tokenizer, opt, logger) # 开始训练




#! 根据model(BERT)， file_list([str]), tokenizer(BERT)训练模型，加载优化器等等
def pretrain_fit(model, file_list, tokenizer, opt, logger):
    logger.info("BERT pretrain: construct PretrainDataset...")
    train_dataset = PretrainDataset(
        file_list,      #! 读取的是json文件地址列表，此json文件已经通过读取dialog数据集从而preprocess
        512,
        tokenizer,
        logger
    )
    logger.info("BERT pretrain: construct Dataloader...")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=opt.bert_pretrain_batch_size, 
        shuffle=True,
        num_workers=8
    )
    optimizer = AdamW(model.parameters(), lr=opt.learning_rate)
    t_total = int(len(train_dataset) * opt.bert_pretrain_epochs // opt.bert_pretrain_batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total)
    for epoch in range(opt.bert_pretrain_epochs):
        print("\nEpoch ", epoch + 1, "/", opt.bert_pretrain_epochs)
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
                opt.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=opt.learning_rate, loss=loss.item())
            avg_loss += loss.item()
        cnt = len(train_dataset) // opt.bert_pretrain_batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        
    save_model(model, tokenizer, opt.save_path)     #* 将模型保存至目录

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