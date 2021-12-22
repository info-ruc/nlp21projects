'''
Author: Walden
Date: 2021-11-09 14:33:55
LastEditTime: 2021-12-16 11:30:21
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/seq2seq/models/BertForFinetune.py
'''
from transformers import BertModel
#TODO 将dialogDatasets的class 改为 DialogDatasets
from seq2seq.dataset.dialogDatasets import *
from seq2seq.utils.utils import *
import torch, os, random
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
import torch.nn.functional as F

def Loss1(pooled_output_i, pooled_output_j, unpaired_pooled_output_i):
    """ Sentence Contrastive Loss
        相似的句子：[(x1, y1), (x2, y2), ...] 
                -> pooled_output_i.shape = [batch_size, response_num, 768]
                -> pooled_output_j.shape = [batch_size, response_num, 768]
        不相似的句子：[(x1, y1'), (x2, y2'), ...] 
                -> unpaired_pooled_output_j = [batch_size, 768]
    Args:
        pooled_output_i ([batch_size, paired_size, 768]): 匹配的句子向量列表x
        pooled_output_j ([batch_size, paired_size, 768]): 匹配的句子向量列表y
        unpaired_pooled_output_j ([unpaired_size, 768]): 不匹配的句子向量列表y
    
    Returns:
        loss: -log(
            ([ exp(cosine_similarity(x1, y1)) + exp(cosine_similarity(x2, y2)) + ... ])     (正例)
            /
            ([ exp(cosine_similarity(x1, y1)) + exp(cosine_similarity(x2, y2)) + ... ] + [exp(cosine_similarity(x1, y1)) + exp(cosine_similarity(x2, y2)) + ...])       (正例+负例)
        )
    """
    #print(pooled_output_i.shape, pooled_output_j.shape, unpaired_pooled_output_i.shape)
    a = torch.sum(torch.exp(F.cosine_similarity(pooled_output_i, pooled_output_j, dim = 2)), dim=1)
    c = torch.unsqueeze(unpaired_pooled_output_i, dim=1)    #[batch_size, 1, 768]
    c = c.expand(c.shape[0], pooled_output_i.shape[1], c.shape[2])  #[batch_size, response_num, 768]
    b = a + torch.sum(torch.exp(F.cosine_similarity(pooled_output_i, c, dim = 2)), dim = 1)
    #print("a", a, "\nb", b)
    return -torch.log(a/b)
 
def Loss2(general_user_profile_3d, general_user_profile_pos, general_user_profile_neg):
    """calculate user loss2
    Args:
        general_user_profile_3d (tensor[batch_size, response_length, 768]): original profile
        general_user_profile_pos (tensor[batch_size, response_length, 768]): positive profile
        general_user_profile_neg (tensor[batch_size, response_length, 768]): negative profile

    Returns:
        tensor(batch_size): loss3
    """
    a = torch.sum(torch.exp(F.cosine_similarity(general_user_profile_3d, general_user_profile_pos, dim = 2)), dim=1)
    b = a + torch.sum(torch.exp(F.cosine_similarity(general_user_profile_3d, general_user_profile_neg, dim = 2)), dim = 1)
    return -torch.log(a/b)

def Loss3(general_user_profile, modified_pos_1, modified_pos_2, modified_neg_1, modified_neg_2):
    """contrastive learning

    Args:
        general_user_profile (batch_size, dialog_length, 768): 原序列
        modified_pos_1 (batch_size, dialog_length, 768): 内部mask的序列, label=pos
        modified_pos_2 (batch_size, dialog_length, 768): 内部replace的序列, label=pos
        modified_neg_1 (batch_size, dialog_length, 768): in-batch user的某句子加入mask的序列, label=neg
        modified_neg_2 (batch_size, dialog_length, 768): in-batch user的某句子加入mask并replace的序列, label=neg

    Returns:
        loss3(batch_size)
    """
    a = torch.sum(torch.exp(F.cosine_similarity(general_user_profile, modified_pos_1, dim = 2)), dim=1) + torch.sum(torch.exp(F.cosine_similarity(general_user_profile, modified_pos_2, dim = 2)), dim=1)
    b = a + torch.sum(torch.exp(F.cosine_similarity(general_user_profile, modified_neg_1, dim = 2)), dim = 1) + torch.sum(torch.exp(F.cosine_similarity(general_user_profile, modified_neg_2, dim = 2)), dim = 1)
    return -torch.log(a/b)


def self_contrastive_sampling(general_user_profile, general_user_profile_neg):
    """生成标签为正/负的向量

    Args:
        general_user_profile ([batch_size, dialog_length, 768]): 原始隐藏层向量，dialog_length = dialog条目总数
        每一个768维向量表示一个句子
        general_user_profile_neg = general_user_profile[derange_list]
        修改方法：
            - remove(mask): mask掉其中一个dialog（假设该dialog信息有效）
                flag = 'pos'
            - re-order: 将r_i, r_j顺序调换
                flag = 'pos' 
            - replace: 将其他user的一个dialog替换掉当前随机一个dialog
                flag = 'neg'
            - replace & reorder: 将其他user的一个dialog替换掉当前随机一个dialog，然后将其与另一个response替换
                flag = 'neg
                
    Returns:
        res_1: ([batch_size, dialog_length, 768]): remove其中一个response之后的list
        res_2: ([batch_size, dialog_length, 768]): 相似的两个response对换
        res_3: ([batch_size, dialog_length, 768]): 将in-batch user的一个response替换当前随机一个response
        res_4: ([batch_size, dialog_length, 768]): 将in-batch user的一个response替换当前随机一个response，然后其与该user中的其他一个response对换
    """
    res_1 = general_user_profile.clone().detach()  #脱离计算图，不共享内存(涉及修改)
    res_2 = general_user_profile.clone().detach()  
    res_3 = general_user_profile.clone().detach()  
    res_4 = general_user_profile.clone().detach()  
    
    
    #* res_1: ([batch_size, dialog_length, 768]): remove其中一个response之后的list
    for index, user in enumerate(res_1):
        t = random.choice(range(res_1.shape[1]))
        user[t] = torch.zeros(768)

    #* res_2: ([batch_size, dialog_length, 768]): 相似的两个response对换
    for index, user in enumerate(res_2):
        t1, t2 = random.sample(range(res_2.shape[1]), 2)
        res_t = user[t1].clone()    #[768]
        user[t1] = user[t2].clone()
        user[t2] = res_t
        
    
    #* res_3: ([batch_size, dialog_length, 768]): 将in-batch user的一个response替换当前随机一个response
    for index, user in enumerate(res_3):
        t1 = random.choice(range(res_3.shape[1]))
        t2 = random.choice(range(general_user_profile_neg.shape[1]))
        res_t = general_user_profile_neg[index][t2].clone()
        user[t1] = res_t        
    
    #* res_4: ([batch_size, dialog_length, 768]): 将in-batch user的一个response替换当前随机一个response，然后其与该user中的其他一个response对换
    for index, user in enumerate(res_4):
        t1, t3 = random.sample(range(res_4.shape[1]), 2)
        t2 = random.choice(range(general_user_profile_neg.shape[1]))
        res_t = general_user_profile_neg[index][t2].clone()
        user[t1] = res_t 
        res_m = user[t1].clone()    #[768]
        user[t1] = user[t3].clone()
        user[t2] = res_m
    return res_1, res_2, res_3, res_4


class RLForPretrain(nn.Module):
    def __init__(self, sentence_bert_model, sequence_bert_model, pretrained_dialog_dir, logger):
        super(RLForPretrain, self).__init__()
        self.logger = logger
        self.sentence_bert_model = sentence_bert_model # sentence bert encoder
        self.sequence_bert_model_encoder = sequence_bert_model.encoder # sequence bert encoder
        
    def forward(self, batch_data):
        data = {
            "response_list_input_ids"             : batch_data["response_list_input_ids"],            # [batch_size, response_length, 512]
            "response_list_attention_mask"        : batch_data["response_list_attention_mask"],       # [batch_size, response_length, 512]
            "response_list_token_type_ids"        : batch_data["response_list_token_type_ids"],       # [batch_size, response_length, 512]
            "split_size"                          : batch_data["split_size"],                         # [batch_size, 1]      
            #TODO 补上split_paired_size
            "paired_response_i_input_ids"         : batch_data["paired_response_i_input_ids"],        # [batch_size, paired_length, 512]   默认为3
            "paired_response_i_attention_mask"    : batch_data["paired_response_i_attention_mask"],   # [batch_size, paired_length, 512]   默认为3 
            "paired_response_i_token_type_ids"    : batch_data["paired_response_i_token_type_ids"],   # [batch_size, paired_length, 512]   默认为3
            "paired_response_j_input_ids"         : batch_data["paired_response_j_input_ids"],        # [batch_size, paired_length, 512]   默认为3
            "paired_response_j_attention_mask"    : batch_data["paired_response_j_attention_mask"],   # [batch_size, paired_length, 512]   默认为3
            "paired_response_j_token_type_ids"    : batch_data["paired_response_j_token_type_ids"],   # [batch_size, paired_length, 512]   默认为3
            "pos_response_list_input_ids"         : batch_data["pos_response_list_input_ids"],        # [batch_size, response_length, 512]
            "pos_response_list_attention_mask"    : batch_data["pos_response_list_attention_mask"],   # [batch_size, response_length, 512]
            "pos_response_list_token_type_ids"    : batch_data["pos_response_list_token_type_ids"],   # [batch_size, response_length, 512]
            "pos_split_size"                      : batch_data["pos_split_size"]                      # [batch_size, 1]
        }
        bs = data["response_list_input_ids"].shape[0]
        #? 生成batch大小的完全错排序列，对应in-batch选择     [0, 1, 2] -> [(0+k)%len, (1+k)%len, (2+k)%len]
        k = random.choice(range(1, bs))
        derange_list = np.array(range(0, bs))
        for i in range(len(derange_list)):
            derange_list[i] = (derange_list[i] + k) % len(derange_list)     #TODO 要判断错排对应的user是否与原先的pair user相同
        derange_list = derange_list.astype(int)
        #* 作为返回值返回，大小为batch
        loss = torch.zeros(bs).cuda()
        
        #! phase 1: 将三维张量(a, b, c)转为二维张量(a*b, c)
        user_input_ids_all = torch.reshape(data["response_list_input_ids"], (data["response_list_input_ids"].shape[0]*data["response_list_input_ids"].shape[1], data["response_list_input_ids"].shape[2]))
        user_attention_mask_all = torch.reshape(data["response_list_attention_mask"], (data["response_list_attention_mask"].shape[0]*data["response_list_attention_mask"].shape[1], data["response_list_attention_mask"].shape[2]))
        user_token_type_ids_all = torch.reshape(data["response_list_token_type_ids"], (data["response_list_token_type_ids"].shape[0]*data["response_list_token_type_ids"].shape[1], data["response_list_token_type_ids"].shape[2]))
        pos_user_input_ids_all = torch.reshape(data["pos_response_list_input_ids"], (data["pos_response_list_input_ids"].shape[0]*data["pos_response_list_input_ids"].shape[1], data["pos_response_list_input_ids"].shape[2]))
        pos_user_attention_mask_all = torch.reshape(data["pos_response_list_attention_mask"], (data["pos_response_list_attention_mask"].shape[0]*data["pos_response_list_attention_mask"].shape[1], data["pos_response_list_attention_mask"].shape[2]))
        pos_user_token_type_ids_all = torch.reshape(data["pos_response_list_token_type_ids"], (data["pos_response_list_token_type_ids"].shape[0]*data["pos_response_list_token_type_ids"].shape[1], data["pos_response_list_token_type_ids"].shape[2]))
        user_paired_response_i_input_ids = torch.reshape(data["paired_response_i_input_ids"], (data["paired_response_i_input_ids"].shape[0]*data["paired_response_i_input_ids"].shape[1], data["paired_response_i_input_ids"].shape[2]))
        user_paired_response_i_attention_mask = torch.reshape(data["paired_response_i_attention_mask"], (data["paired_response_i_attention_mask"].shape[0]*data["paired_response_i_attention_mask"].shape[1], data["paired_response_i_attention_mask"].shape[2]))
        user_paired_response_i_token_type_ids = torch.reshape(data["paired_response_i_token_type_ids"], (data["paired_response_i_token_type_ids"].shape[0]*data["paired_response_i_token_type_ids"].shape[1], data["paired_response_i_token_type_ids"].shape[2]))
        user_paired_response_j_input_ids      = torch.reshape(data["paired_response_j_input_ids"], (data["paired_response_j_input_ids"].shape[0]*data["paired_response_j_input_ids"].shape[1], data["paired_response_j_input_ids"].shape[2]))
        user_paired_response_j_attention_mask = torch.reshape(data["paired_response_j_attention_mask"], (data["paired_response_j_attention_mask"].shape[0]*data["paired_response_j_attention_mask"].shape[1], data["paired_response_j_attention_mask"].shape[2]))
        user_paired_response_j_token_type_ids = torch.reshape(data["paired_response_j_token_type_ids"], (data["paired_response_j_token_type_ids"].shape[0]*data["paired_response_j_token_type_ids"].shape[1], data["paired_response_j_token_type_ids"].shape[2]))
        
        #! phase 2-1: 该batch内的所有user的response一起过sentence_layer
        bert_inputs_all = {
            "input_ids": user_input_ids_all,
            "attention_mask": user_attention_mask_all,
            "token_type_ids": user_token_type_ids_all
        }
        #! WARNING! pooled_output_all的显存开销非常大！大约几个G
        _, pooled_output_all = self.sentence_bert_model(**bert_inputs_all)[:2]      # [batch_size * response_length, 768]
        
        #! phase 2-2: 该batch内所有user的pos_user同时过sentence_layer（不需要记录梯度）
        bert_inputs_all_pos = {
            "input_ids": pos_user_input_ids_all,
            "attention_mask": pos_user_attention_mask_all,
            "token_type_ids": pos_user_token_type_ids_all
        }
        
        
        with torch.no_grad():       #pos梯度不参与计算，节省显存
            _, pooled_output_all_pos = self.sentence_bert_model(**bert_inputs_all_pos)[:2]      # [batch_size * response_length, 768]
    
        
        #! phase 3: 该batch内所有user的paired response同时过sentence_layer，得到loss1，被比较的不需要记录梯度
        bert_inputs_i = {
            "input_ids": user_paired_response_i_input_ids,
            "attention_mask": user_paired_response_i_attention_mask,
            "token_type_ids": user_paired_response_i_token_type_ids
        }
        bert_inputs_j = {
            "input_ids": user_paired_response_j_input_ids,
            "attention_mask": user_paired_response_j_attention_mask,
            "token_type_ids": user_paired_response_j_token_type_ids
        }
        _, pooled_output_i = self.sentence_bert_model(**bert_inputs_i)[:2]      # [batch_size * paired_response_length, 768]
        with torch.no_grad():
            _, pooled_output_j = self.sentence_bert_model(**bert_inputs_j)[:2]
        #print(list([data["split_size"][_] for _ in range(len(data["split_size"]))]))
        split_paired_size = 3   #TODO 写到data字典中
        pooled_output_i_3d = pooled_output_i.reshape(int(pooled_output_i.shape[0]/split_paired_size), split_paired_size, pooled_output_i.shape[1])      # [batch_size, paired_response_length, 768]
        pooled_output_j_3d = pooled_output_j.reshape(int(pooled_output_j.shape[0]/split_paired_size), split_paired_size, pooled_output_j.shape[1])
        unpaired_output_j_3d = pooled_output_j_3d[derange_list][:,0,:]                      # [batch_size, 1, 768]
        loss1 = Loss1(pooled_output_i_3d, pooled_output_j_3d, unpaired_output_j_3d)
        self.logger.info("loss1: {0}".format(str(loss1)))
        loss += loss1
        
        #! phase 4-1: user 过sequence_layer，得到general_user_profile [batch_size, dialog_length, 768]
        pooled_output_all_3d = pooled_output_all.reshape(int(pooled_output_all.shape[0]/int(data["split_size"][0])), int(data["split_size"][0]), pooled_output_all.shape[1])           # [batch_size, response_length, 768]
        general_user_profile = self.sequence_bert_model_encoder.forward(pooled_output_all_3d).last_hidden_state            # [batch_size, dialog_length, 768] 
        #general_user_profile = self.sentence_bert_model.encoder.forward(pooled_output_all_3d).last_hidden_state            # [batch_size, dialog_length, 768] 

        
        #! phase 4-2: pos_user 过sequence_layer，得到general_user_profile_pos [batch_size, dialog_length, 768]
        # WARNING: data["split_size"][0]
        pooled_output_all_pos_3d = pooled_output_all_pos.reshape(int(pooled_output_all_pos.shape[0]/int(data["split_size"][0])), int(data["split_size"][0]), pooled_output_all_pos.shape[1])           # [batch_size, response_length, 768]
        general_user_profile_pos = self.sequence_bert_model_encoder.forward(pooled_output_all_pos_3d).last_hidden_state            # [batch_size, dialog_length, 768] 
        #general_user_profile_pos = self.sentence_bert_model.encoder.forward(pooled_output_all_pos_3d).last_hidden_state            # [batch_size, dialog_length, 768] 
        #print(general_user_profile_pos.shape)       # [batch_size, dialog_length, 768] 
        
        #! phase 4-3: 由general_user_profile生成user_inbatch的general_user_profile_neg（把梯度去掉）
        general_user_profile_neg = general_user_profile.clone().detach()[derange_list]
        # print(general_user_profile_neg.shape)      # [batch_size, dialog_length, 768] 
        
        #! phase 4-4: 由general_user_profile_3d, general_user_profile_neg, general_user_profile_pos生成loss3
        loss2 = Loss2(general_user_profile, general_user_profile_pos, general_user_profile_neg)     # [batch_size, dialog_length, 768] 
        self.logger.info("loss2: {0}".format(str(loss2)))
        loss += loss2
        
        #! phase 5: 由pooled_output_all_3d [batch_size, dialog_length, 768] 生成4种不同的序列，然后过encoder，计算loss3
        modified_pos_1, modified_pos_2, modified_neg_1, modified_neg_2 = self_contrastive_sampling(general_user_profile, general_user_profile_neg)
        modified_pos_1 = self.sentence_bert_model.encoder.forward(modified_pos_1).last_hidden_state
        modified_pos_2 = self.sentence_bert_model.encoder.forward(modified_pos_2).last_hidden_state
        modified_neg_1 = self.sentence_bert_model.encoder.forward(modified_neg_1).last_hidden_state
        modified_neg_2 = self.sentence_bert_model.encoder.forward(modified_neg_2).last_hidden_state
        loss3 = Loss3(general_user_profile, modified_pos_1, modified_pos_2, modified_neg_1, modified_neg_2)
        self.logger.info("loss3: {0}".format(str(loss3)))
        loss += loss3
        return torch.mean(loss)
            
    

def RL_pretrain_model(json_filelist, opt, sentence_bert_model, sequence_bert_model, tokenizer, logger):
    """initialize finetuning processing model

    Args:
        json_filelist (list(str)): dialog json file absolute path
        opt (argparse): argparse from shell
        sentence_bert_model (bert_model): BertModel instance
        sequence_bert_model (bert_model): BertModel instance
        tokenizer (BertTokenizer): BertTokenizer instance

    Returns:
        calling finetune_fit() method
    """
    logger.info("entering rl pretrain..")
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = RLForPretrain(sentence_bert_model, sequence_bert_model, opt.gen_userdata_outputdir, logger)
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
    model = model.to(device)
    RL_pretrain_fit(model, json_filelist, tokenizer, opt, logger)
    
#! 根据model(BERT)， file_list([str]), tokenizer(BERT)训练模型，加载优化器等等
# TODO 调用PretrainDataset, DataLoader, save_model
# TODO 比较tutorial中，Pretrain & Point dataset的区别与联系
# TODO 可能需要修改train_dataset
def RL_pretrain_fit(model, json_filelist, tokenizer, opt, logger):
    """loading dialog data from json_data, construct dialog data(PretrainDataset instance), and initialize dataloader & epoches

    Args:
        model (BertForFinetune): generated model
        json_filelist (list(str)): sampled **user file** absolute path
        tokenizer (BertTokenizer): tokenizer
        opt (argparser): argparser from shell
        sentence_bert_model
    """
    logger.info("rl pretrain: construct RL_pretrain_Dataset...")
    train_dataset = RL_pretrain_Dataset(
        json_filelist,
        tokenizer
    )
    logger.info("rl pretrain: construct Dataloader...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.rl_pretrain_batch_size, 
        shuffle=True,
        num_workers=8
    )
    optimizer = AdamW(model.parameters(), lr=opt.learning_rate)
    
    t_total = int(len(train_dataset) * opt.rl_pretrain_epochs // opt.rl_pretrain_batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total)
    
    
    for epoch in range(opt.rl_pretrain_epochs):
        print("\nEpoch ", epoch+1, "/", opt.rl_pretrain_epochs)
        #logger.write("Epoch " + str(epoch + 1) + "/" + str(opt.pretrain_epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):  #每次取出一个batch
            loss = rl_pretrain_train_step(model, training_data, opt) # 过模型, 取loss
            #if is_response == 0: continue   # post
            logger.info("loss: {0}".format(str(loss)))
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) #* 梯度裁剪只能在训练的时候使用
            optimizer.step() # 更新模型参数
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                opt.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=opt.learning_rate, loss=loss.item())
            avg_loss += loss.item()
        cnt = len(train_dataset) // opt.rl_pretrain_batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        
        #TODO 若有模型，则读取模型
        
        #optimizer, model需要保存
        if (epoch+1) % 5 == 0:
            if not os.path.exists(opt.rl_pretrain_modeldir):
                os.mkdir(opt.rl_pretrain_modeldir)
            checkpoint = {
                "epoch": epoch+1,
                "sentence_model_state": model.sentence_bert_model.state_dict(),
                "sequence_model_state": model.sequence_bert_model_encoder.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(opt.rl_pretrain_modeldir, "checkpoint{0}.pth".format(str(epoch+1))))
            logger.info("rl pretrain: model saved! epoch = {0}".format(str(epoch+1)))
def rl_pretrain_train_step(model, train_data, opt):
    """calculate Loss in every epoch of pretrain_fit()

    Args:
        model (Bert_model): the model which calls forward() method
        train_data (dict): trained data treated as parameters in model.forward()
        opt (options): parameters of parsers

    Returns:
        loss: loss
    """
    #is_response = 0
    multi_gpu = False
    device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
    #with torch.no_grad():
    for key in train_data.keys():
        #print(train_data[key])
        if type(train_data[key]) == list and len(train_data[key]) != 0: 
            #* 将[tensor(), tensor()]转为tensor()，见https://blog.csdn.net/liu16659/article/details/114752918
            train_data[key] = torch.stack(train_data[key], 1).to(device)
        train_data[key] = train_data[key].to(device)
    loss = model.forward(train_data)
    return loss#, is_response