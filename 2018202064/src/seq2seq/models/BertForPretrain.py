'''
Author: your name
Date: 2021-10-29 03:10:41
LastEditTime: 2021-11-16 03:51:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /SIGIR2022/seq2seq/models/BertForPretrain.py
'''
from transformers import BertModel
import torch, os, random
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads
# MLM + CLS Score BERT预训练模型
#* 传入一个bert实例，重写bert的结构（通过调用self.bert_model以获得网络输出
class BertForPretrain(nn.Module):
    def __init__(self, bert_model):
        super(BertForPretrain, self).__init__()
        self.bert_model = bert_model # bert encoder
        self.cls = BertPreTrainingHeads(self.bert_model.config) # 用于直接算MLM的logits, 以及CLS的hidden logits
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) # 用于算loss，忽略掉之前标记的ignore_index
        
    def forward(self, batch_data):
        input_ids = batch_data["input_ids"] # [batch_size, seq_length]
        attention_mask = batch_data["attention_mask"] # [batch_size, sequence_length]
        token_type_ids = batch_data["token_type_ids"] # [batch_size, sequence_length]
        mlm_labels = batch_data["mlm_labels"] # [batch_size, sequence_length]. 不预测的地方填-100, 不参与loss计算.
        #TODO// 加一个label表示它是post还是response
        bert_inputs = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'token_type_ids': token_type_ids
        }
        
        # 过bert encoder
        sequence_output, pooled_output = self.bert_model(**bert_inputs)[:2]  
        # sequence_output: [batch_size, sequence_length, hidden_size]; pooled_output: [batch_size, hidden_size]
        
        # 过MLP, 取MLM Logitsy与CLS logits
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output) # prediction_scores: [batch_size, sequence_length, vocab_size]; seq_relationship_score: [batch_size, 2] 
        # 算loss
        masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.bert_model.config.vocab_size), mlm_labels.view(-1)) # [batch_size]       
        total_loss = masked_lm_loss # [batch_size]


        return total_loss
        
#* 依据opt.save_path，将weibo dialog预训练模型参数保存在该目录下
def save_model(model, tokenizer, output_dir):
    """save model to a directory

    Args:
        model (bert_model): a generated bert model
        tokenizer (bert_tokenizer): an embedding tokenizer 
        d from bert moedl
        output_dir (str): a specified output directory absolute path
    """
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    # torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    model_to_save.bert_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
#* 设置随机种子
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True