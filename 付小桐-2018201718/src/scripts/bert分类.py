#!/usr/bin/env python
# coding: utf-8

# In[23]:


#载入相关模块
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as tfs
import warnings
import  openpyxl
import random
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler 
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import joblib
#释放显存
torch.cuda.empty_cache()


# In[24]:


#设置模型参数
SEED = 1508
BATCH_SIZE = 30
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[25]:


#读取数据
alldata = pd.read_excel('labeled评论数据.xlsx')
#提取虚假评论数据
all_false=alldata[alldata['标签']==1]
#设置index
all_false.index=range(0,len(all_false))
#提取文本内容
false_text=all_false['评价内容']
#提取真实评论数据
all_true=alldata[alldata['标签']==0]
#设置index
all_true.index=range(len(all_false),len(alldata))
#提取文本内容
true_text=all_true['评价内容']
#将真假文本内容按顺序连接
sentences = false_text.append(true_text)
#展示虚假评论数据
all_false.loc[1:5]


# In[26]:


#设定标签
false_targets = np.ones((len(false_text)))
true_targets = np.zeros((len(true_text)))
targets = np.concatenate((false_targets, true_targets), axis=0).reshape(-1, 1)   #(10000, 1)
total_targets = torch.tensor(targets)
total_targets


# In[27]:


#使用哈工大bert中文预训练模型
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir="E:/transformer_file/")


# In[28]:


print(false_text[2])
print(tokenizer.tokenize(false_text[2]))

print(tokenizer.encode(false_text[2]))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(false_text[2])))


# In[29]:


#将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size=130):
  
   tokens = tokenizer.encode(sentence[:limit_size])  #直接截断  
   if len(tokens) < limit_size + 2:                  #补齐（pad的索引号就是0）
       tokens.extend([0] * (limit_size + 2 - len(tokens)))   
   return tokens

input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]

input_tokens = torch.tensor(input_ids)
print(input_tokens.shape)                    #torch.Size([10000, 128])


# In[30]:


#建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks

atten_masks = attention_masks(input_ids)
attention_tokens = torch.tensor(atten_masks)    


# In[31]:


#####################################


# In[32]:


split_seed=156
from sklearn.model_selection import train_test_split
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens, total_targets, random_state=666, test_size=0.2)
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens, random_state=666, test_size=0.2)
print(train_inputs.shape, test_inputs.shape)      #torch.Size([8000, 128]) torch.Size([2000, 128])
print(train_masks.shape)                          #torch.Size([8000, 128])和train_inputs形状一样

print(train_inputs[0])
print(train_masks[0])


# In[33]:


train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


# In[34]:


for i, (train, mask, label) in enumerate(train_dataloader):
    print(train.shape, mask.shape, label.shape)               #torch.Size([16, 128]) torch.Size([16, 128]) torch.Size([16, 1])
    break
print('len(train_dataloader)=', len(train_dataloader))        #500


# In[35]:


##################


# In[36]:



model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels = 2)     #num_labels表示2个分类，好评和差评


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    model.to(device)


# In[37]:


#设置参数学习率
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr = LEARNING_RATE, eps = EPSILON)


# In[38]:


#设置epoch
epochs = 15
#training steps 的数量: [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * epochs

# 设计学习率scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


# In[39]:


################


# In[40]:


def binary_acc(preds, labels):      #preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()      #eq里面的两个参数的shape=torch.Size([16])    
    acc = correct.sum().item() / len(correct)
    rec=torch.lt(torch.max(preds, dim=1)[1], labels.flatten()).float()
    if labels.flatten().sum().item()!=0:
        recall=1-rec.sum().item()/labels.flatten().sum().item()
    else:
        recall=-99
    return acc, recall


# In[41]:


import time
import datetime
def format_time(elapsed):    
    elapsed_rounded = int(round((elapsed)))    
    return str(datetime.timedelta(seconds=elapsed_rounded))   #返回 hh:mm:ss 形式的时间


# In[42]:


def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc,avg_rec = [],[],[]
     
    model.train()
    for step, batch in enumerate(train_dataloader):

        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:            
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
        
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = output[0], output[1] 
    
        avg_loss.append(loss.item())
        
        acc,recall= binary_acc(logits, b_labels)
        avg_acc.append(acc)
        if recall!=-99:
            avg_rec.append(recall)
        
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)      #大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()              #更新模型参数
        scheduler.step()              #更新learning rate
        
    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    avg_rec = np.array(avg_rec).mean()
    return avg_loss, avg_acc, avg_rec


# In[43]:


def evaluate(model):    
    avg_acc,avg_rec = [],[]    
    model.eval()         #表示进入测试模式
      
    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
       
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
           
            acc,recall= binary_acc(output[0], b_labels)
            avg_acc.append(acc)
            if recall!=-99:
                avg_rec.append(recall)
    avg_acc = np.array(avg_acc).mean()
    avg_rec = np.array(avg_rec).mean()
    return avg_acc, avg_rec


# In[44]:


train_acc_arr=np.zeros(epochs)
train_recall_arr=np.zeros(epochs)
test_acc_arr=np.zeros(epochs)
test_recall_arr=np.zeros(epochs)


# In[45]:


for epoch in range(epochs):
    
    train_loss, train_acc,train_recall = train(model, optimizer)
    train_acc_arr[epoch]=train_acc
    train_recall_arr[epoch]=train_recall
    print('epoch={},训练准确率={}，损失={},recall={}'.format(epoch, train_acc, train_loss,train_recall))
    test_acc,test_recall = evaluate(model)
    test_acc_arr[epoch]=test_acc
    test_recall_arr[epoch]=test_recall
    print("epoch={},测试准确率={},recall={}".format(epoch, test_acc,test_recall))


# In[46]:


#绘制图像
x=range(1,epochs+1)
import matplotlib.pyplot as plt
from pylab import *     
plt.figure(figsize=(20,10))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
plt.ylim(0.5, 1)  # 限定纵轴的范围
plt.plot(x, train_acc_arr, marker='*',  ms=10,label=u'train accuracy')
plt.plot(x, train_recall_arr, marker='*', ms=10,label=u'train recall')
plt.plot(x, test_acc_arr, marker='o', mec='r', mfc='w',label=u'test accuracy')
plt.plot(x, test_recall_arr, marker='o', mec='r', mfc='w',label=u'test recall')

plt.legend()  # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"epoch") #X轴标签
plt.ylabel("比率") #Y轴标签
plt.title("A simple plot") #标题

plt.show()


# In[47]:


test_acc_arr


# In[48]:


test_recall_arr

