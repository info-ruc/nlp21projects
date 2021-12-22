'''
Author: your name
Date: 2021-12-19 12:55:55
LastEditTime: 2021-12-22 12:52:02
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/seq2seq/models/Infer.py
'''
from transformers import BertModel
import torch, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads


class InferModel(nn.Module):
    def __init__(self, json_filelist, tokenizer, opt, sentence_model, sequence_model_encoder, transformer_encoder, transformer_decoder, output_mlp, logger):
        super(InferModel, self).__init__()
        self.sentence_model = sentence_model # bert encoder
        self.sequence_model_encoder = sequence_model_encoder
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder
        self.output_mlp = output_mlp
        self.vocab_size = tokenizer.vocab_size
        self.logger = logger
        self.softmax = nn.Softmax(dim=-1)
        self.tokenizer = tokenizer
        #self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=0)
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
        bs = data["label"].shape[0]
        loss = torch.zeros(bs)
        post_embedding = self.sentence_model.embeddings.forward(data["post_input_ids"].reshape(-1, data["post_input_ids"].shape[-1]))   #[batch_size*post_num, TRUNCATE_SIZE] -> [batch_size*post_num, TRUNCATE_SIZE, 768]
        
        tgt_response = data["response_input_ids"].clone()
        for i in range(tgt_response.shape[0]):
            for j in range(tgt_response.shape[1]):
                tgt_response[i][j][0] = 100 #[SOS]
        
        tgt = self.sentence_model.embeddings.forward(data["response_input_ids"].reshape(-1, data["response_input_ids"].shape[-1]))      #[batch_size*response_num, TRUNCATE_SIZE] -> [batch_size*response_num, TRUNCATE_SIZE, 768]
        #? modified: tgt = self.sentence_model.embeddings.forward(tgt_response.reshape(-1, tgt_response.shape[-1]))      #[batch_size*response_num, TRUNCATE_SIZE] -> [batch_size*response_num, TRUNCATE_SIZE, 768]
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
        #TODO general_user_profile与post_embedding拼接
        post_embedding = post_embedding.reshape(bs, -1, post_embedding.shape[1], post_embedding.shape[2])   #[batch_size, post_num, TRUNCATE_SIZE, 768]
        #? 有没有更Pythonic的写法？
        for i in range(bs):
            for j in range(post_embedding.shape[1]):
                post_embedding[i][j] = torch.cat((general_user_profile[i], post_embedding[i][j][1:]), dim=0)               #[1, 768] + [post_num-1, 768]
        #post_embedding: [batch_size, post_num, TRUNCATE_SIZE, 768]
        #tgt: [batch_size, response_num, TRUNCATE_SIZE, 768]
        post_embedding = post_embedding.reshape(-1, post_embedding.shape[2], post_embedding.shape[3])   #[batch_size*post_num, TRUNCATE_SIZE, 768]
        tgt = tgt.reshape(-1, tgt.shape[2], tgt.shape[3])               #[batch_size*response_num, TRUNCATE_SIZE, 768]
        memory = self.transformer_encoder(post_embedding)               #[batch_size*response_num, TRUNCATE_SIZE, 768]
        #! decoder, tgt的第一个词是[SOS]
        #! for循环要求出这个词，然后shifted right,
        for index in range(tgt.shape[0]):
            # [SOS]直接取tgt[0][0]即可
            tgt_input = tgt[index][0]         #size: 768
            tgt_input = torch.unsqueeze(tgt_input, dim = 0) #size: [1, 768]
            tgt_input = torch.unsqueeze(tgt_input, dim = 0) #size: [1, 1, 768]
            response_list = [101]
            for word_index in range(tgt.shape[1]):
                if word_index == 0:
                    out = self.transformer_decoder(tgt=tgt_input, memory=memory[index:index+1])          #[1, 1, 768],      [1, TRUNCATE_SIZE, 768]用第index个post的embedding
                    out = self.output_mlp(out)      #[1, 1, 21128]
                    print("softmax前", out[:,0,:])
                    out = nn.Sigmoid()(out)
                    out = self.softmax(out)         #[1, 1, 21128]
                    print("softmax后", out[:,0,0],"最大:", torch.max(out[:,0,:], dim=-1))
                    #print(out[:,0,0], "是否index0 == 1?", out[:,0,0]==out[:,0,1])
                    out = out.detach().cpu()
                    #out = np.argmax(out, axis=-1)   #[1, 1, 1]
                    print(self.tokenizer.convert_ids_to_tokens(np.argsort(out, axis=-1)[:,:,-15:].reshape(-1)))
                    out = np.argsort(out, axis=-1)[:,:,-1]
                    word_id = out.reshape(-1)[-1]  #取最后一个
                    print("输入:", response_list, "输出:", self.tokenizer.convert_ids_to_tokens(int(word_id)))
                    response_list.append(int(word_id))
                else:
                    tgt_input = self.sentence_model.embeddings.forward(torch.tensor([response_list]).cuda())       #response_list: [1, n] -> [1, n, 768]
                    #tgt_input = torch.unsqueeze(tgt_input, dim = 0) #size: [1, n, 768]
                    #print(tgt_input.shape, memory[index:index+1].shape)
                    out = self.transformer_decoder(tgt=tgt_input, memory=memory[index:index+1])          #[1, n, 768], [1, TRUNCATE_SIZE, 768] -> [1, n, 768]
                    out = self.output_mlp(out)      #[1, n, 21128]
                    print("softmax前", out[:,0,:])
                    out = nn.Sigmoid()(out)
                    out = self.softmax(out)         #[1, n, 21128]
                    print("softmax后", out[:,0,0], "最大:", torch.max(out[:,0,:], dim=-1))
                    #print(out[:,0,0], "是否index0 == 1?", out[:,0,0]==out[:,0,1])
                    out = out[:,word_index,:]               #[1, 21128]
                    out = out.detach().cpu()
                    #out = np.argmax(out, axis=-1)   #[1, 1, 1]
                    print(self.tokenizer.convert_ids_to_tokens(np.argsort(out, axis=-1)[:,-15:].reshape(-1)))
                    out = np.argsort(out, axis=-1)[:,-1]
                    word_id = out.reshape(-1)[-1]  #取最后一个
                    print("输入:", response_list, "输出:", self.tokenizer.convert_ids_to_tokens(int(word_id)))
                    response_list.append(int(word_id))
                #response_list = np.array(self.tokenizer.convert_ids_to_tokens(out.reshape(-1))).reshape(out.shape[0], out.shape[1], out.shape[2])
            #print(response_list)
            
            print("-"*20)
            for id in response_list:
                print(self.tokenizer.convert_ids_to_tokens(id), end=' ')
            print("\n"+"-"*20)
            for id in response_list:
                print(id, end=' ')
            print("\n"+"-"*20)
            return        
            
                    
        """    #tgt_input = torch.cat(tgt_input, torch.unsqueeze(torch.unsqueeze(tgt[index][word_index], dim=0), dim=0), dim=1)    # [1, n, 768]
                    out = self.transformer_decoder(tgt=tgt_input, memory=memory[index])          
                out = out[:,-1,:]   #[1, 1, 768]
                out = self.output_mlp(out)  #[1, 1, 21128]
                out = self.softmax(out)     
                out = out.detach().cpu()
        print(response_list)
        """
                
                
            
            
        """out = self.output_mlp(out)                                      #[batch_size*response_num, TRUNCATE_SIZE, 21128]
        out = out.reshape(bs, -1, out.shape[1], out.shape[2])           #[batch_size, response_num, TRUNCATE_SIZE, 21128]
        out = self.softmax(out)
        out = out.detach().cpu()
        
        #out = np.argmax(out, axis=-1)   #[batch_size, response_num, TRUNCATE_SIZE]
        #out = np.argsort(out, axis=-1)[:,:,-1]
        response_list = np.array(self.tokenizer.convert_ids_to_tokens(out.reshape(-1))).reshape(out.shape[0], out.shape[1], out.shape[2])
        for user in response_list:
            for response in user:
                print(response)
        #? 有问题
        return response_list"""
        """#!=TODO infer与finetune在以下部分不同
        response_list = []
        tgt = tgt.reshape(bs, -1, tgt.shape[1], tgt.shape[2])
        memory = memory.reshape(bs, -1, memory.shape[1], memory.shape[2])   #[batch_size, response_num, TRUNCATE_SIZE, 768]
        tgt = torch.transpose(tgt, 0, 1)        #[response_num, batch_size, TRUNCATE_SIZE, 768]
        memory = torch.transpose(memory, 0, 1)  #[response_num, batch_size, TRUNCATE_SIZE, 768]
        for index in range(tgt.shape[0]):
            cur_tgt = tgt[:index+1] #[cur_num, batch_size, TRUNCATE_SIZE, 768]
            cur_tgt = torch.transpose(cur_tgt, 0, 1)    #[cur_num, batch_size, TRUNCATE_SIZE, 768] -> [batch_size, cur_num, TRUNCATE_SIZE, 768]
            cur_tgt = cur_tgt.reshape(cur_tgt.shape[0], -1, cur_tgt.shape[-1])#[batch_size, cur_num, TRUNCATE_SIZE, 768] -> [batch_size, cur_num*TRUNCATE_SIZE, 768]
            cur_memory = torch.transpose(memory, 0, 1)    #[res_num, batch_size, TRUNCATE_SIZE, 768] -> [batch_size, res_num, TRUNCATE_SIZE, 768]
            cur_memory = cur_memory.reshape(cur_memory.shape[0], -1, cur_memory.shape[-1])#[batch_size, res_num, TRUNCATE_SIZE, 768] -> [batch_size, res_num*TRUNCATE_SIZE, 768]
            out = self.transformer_decoder(tgt = cur_tgt, memory = cur_memory)#[batch_size, cur_num*TRUNCATE_SIZE, 768], [batch_size, res_num*TRUNCATE_SIZE, 768] -> [batch_size, cur_num*TRUNCATE_SIZE, 768]
            #out: [batch_size, cur_num*TRUNCATE_SIZE, 768]
            #TODO 好像写错了，memory应当是transformer_decoder的输出拼接才对，tgt才是当前句子？问一下
        #self.tokenizer.convert_ids_to_tokens()
        #out = self.transformer_decoder(tgt=tgt, memory=memory)          #[batch_size*response_num, TRUNCATE_SIZE, 768]
        out = self.output_mlp(out)                                      #[batch_size*response_num, TRUNCATE_SIZE, 21128]
        out = out.reshape(bs, -1, out.shape[1], out.shape[2])           #[batch_size, response_num, TRUNCATE_SIZE, 21128]
        out = self.softmax(out)
        label = data["label"]
        out = out.reshape(-1, out.shape[-1])
        label = label.reshape(-1)"""
        