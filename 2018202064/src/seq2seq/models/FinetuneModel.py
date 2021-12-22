'''
Author: your name
Date: 2021-12-16 11:02:49
LastEditTime: 2021-12-22 09:23:26
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/seq2seq/models/Finetune.py
'''
from transformers import BertModel
import torch, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads
class FinetuneModel(nn.Module):
    def __init__(self, tokenizer, sentence_model, sequence_model_encoder, logger, drange_size = 10, truncate_size = 30):
        super(FinetuneModel, self).__init__()
        self.sentence_model = sentence_model # bert encoder
        self.sequence_model_encoder = sequence_model_encoder
        #self.general_user_profile_mlp = nn.Linear(drange_size, 1)
        #nn.init.xavier_uniform(self.general_user_profile_mlp.weight)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True), num_layers=3)
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True), num_layers=3)
        self.output_mlp = nn.Linear(768, tokenizer.vocab_size)
        nn.init.normal_(self.output_mlp.weight)
        self.vocab_size = tokenizer.vocab_size
        self.logger = logger
        self.softmax = nn.Softmax(dim=-1)
        #self.nllloss = nn.NLLLoss(reduction = 'sum')     #? rl pretrain时，定义在类外的loss函数需要这样初始化吗？
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=0)
        #self.cls = BertPreTrainingHeads(self.bert_model.config) # 用于直接算MLM的logits, 以及CLS的hidden logits
        #self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) # 用于算loss，忽略掉之前标记的ignore_index
    def forward(self, batch_data):
        data = {
            "label": batch_data["label"],        #[batch_size, response_num, TRUNCATE_SIZE]
            "post_input_ids": batch_data["post_input_ids"],      #[batch_size, post_num, TRUNCATE_SIZE]
            "post_attention_mask": batch_data["post_attention_mask"], #[batch_size, post_num, TRUNCATE_SIZE]
            "post_token_type_ids": batch_data["post_token_type_ids"],  #[batch_size, post_num, TRUNCATE_SIZE]
            "response_input_ids": batch_data["response_input_ids"],  #[batch_size, response_num, TRUNCATE_SIZE]
            "response_attention_mask": batch_data["response_attention_mask"],    #[batch_size, response_num, TRUNCATE_SIZE]
            "response_token_type_ids": batch_data["response_token_type_ids"]     #[batch_size, response_num, TRUNCATE_SIZE]
        }
        #print("DEBUG: ", data["label"][0][0], data["response_input_ids"][0][0], sep='\n')
        
        bs = data["label"].shape[0]
        loss = torch.zeros(bs)
        #TODO// 把response_input_ids的第一个改成[SOS]，100, 20211121
        tgt_response = data["response_input_ids"].clone()
        for i in range(tgt_response.shape[0]):
            for j in range(tgt_response.shape[1]):
                tgt_response[i][j][0] = 100 #[SOS]
        
        post_embedding = self.sentence_model.embeddings.forward(data["post_input_ids"].reshape(-1, data["post_input_ids"].shape[-1]))   #[batch_size*post_num, TRUNCATE_SIZE] -> [batch_size*post_num, TRUNCATE_SIZE, 768]
        tgt = self.sentence_model.embeddings.forward(data["response_input_ids"].reshape(-1, data["response_input_ids"].shape[-1]))      #[batch_size*response_num, TRUNCATE_SIZE] -> [batch_size*response_num, TRUNCATE_SIZE, 768]
        #? modified: tgt = self.sentence_model.embeddings.forward(tgt_response.reshape(-1, tgt_response.shape[-1]))      #[batch_size*response_num, TRUNCATE_SIZE] -> [batch_size*response_num, TRUNCATE_SIZE, 768]
        #?? 不应该在tgt开头和结尾补上special token！！！重新finetune
        tgt = tgt.reshape(bs, -1, tgt.shape[1], tgt.shape[2])                 #[batch_size, response_num, TRUNCATE_SIZE, 768]
        #? 此处要3d->2d，是否需要embed位置信息？
        bert_inputs_response = {
            "input_ids": data["response_input_ids"].reshape(-1, data["response_input_ids"].shape[-1]),
            "attention_mask": data["response_attention_mask"].reshape(-1, data["response_attention_mask"].shape[-1]),
            "token_type_ids": data["response_token_type_ids"].reshape(-1, data["response_token_type_ids"].shape[-1])
        }
        _, pooled_output = self.sentence_model(**bert_inputs_response)[:2]
        #print(pooled_output.shape)      #[batch_size*response_num, 768]
        general_user_profile = self.sequence_model_encoder.forward(pooled_output.reshape(bs, -1, pooled_output.shape[-1])).last_hidden_state
        #print(general_user_profile.shape)   #[batch_size, response_num, 768]
        #* 直接取每条句子的平均值作为profile
        general_user_profile = torch.unsqueeze(torch.mean(general_user_profile, dim=1), dim=1)  #[batch_size, 1, 768]
        #print(general_user_profile.shape)
        #* 过一层mlp作为profile
        #general_user_profile = general_user_profile.transpose(1, 2)         #[batch_size, 768, response]
        #general_user_profile = self.general_user_profile_mlp(general_user_profile).transpose(1, 2) #[batch_size, 1, 768]
        #TODO// general_user_profile与post_embedding拼接
        post_embedding = post_embedding.reshape(bs, -1, post_embedding.shape[1], post_embedding.shape[2])   #[batch_size, post_num, TRUNCATE_SIZE, 768]
        #? 有没有更Pythonic的写法？
        for i in range(bs):
            for j in range(post_embedding.shape[1]):    
                post_embedding[i][j] = torch.cat((general_user_profile[i], post_embedding[i][j][1:]), dim=0)               #[1, 768] + [post_num-1, 768]
                
        #post_embedding: [batch_size, post_num, TRUNCATE_SIZE, 768]
        #tgt: [batch_size, response_num, TRUNCATE_SIZE, 768]
        post_embedding = post_embedding.reshape(-1, post_embedding.shape[2], post_embedding.shape[3])   #[batch_size*post_num, TRUNCATE_SIZE, 768]
        tgt = tgt.reshape(-1, tgt.shape[2], tgt.shape[3])                                               #[batch_size*response_num, TRUNCATE_SIZE, 768]
        memory = self.transformer_encoder(post_embedding)               #[batch_size*response_num, TRUNCATE_SIZE, 768]
        out = self.transformer_decoder(tgt=tgt, memory=memory)          #[batch_size*response_num, TRUNCATE_SIZE, 768]
        out = self.output_mlp(out)                                      #[batch_size*response_num, TRUNCATE_SIZE, 21128]
        out = out.reshape(bs, -1, out.shape[1], out.shape[2])           #[batch_size, response_num, TRUNCATE_SIZE, 21128]
        #!
        out = out[:,:,:-1,:]
        #*
        out = nn.Sigmoid()(out)
        out = self.softmax(out)
        label = data["label"]
        out = out.reshape(-1, out.shape[-1])
        #!
        label = label[:,:,1:]                   #可能需要把<EOS>后的padding去掉才行
        #*
        label = label.reshape(-1)       #[batch_size*response_num*(TRUNCATE_SIZE-1)]
        loss = self.CrossEntropyLoss(out, label)
        return loss