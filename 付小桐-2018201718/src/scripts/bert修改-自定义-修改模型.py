#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from transformers import get_linear_schedule_with_warmup
import joblib
import torch.nn as nn
import torch.nn.functional as F
#释放显存
torch.cuda.empty_cache()


# In[2]:


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


# In[3]:


#读取数据
alldata = pd.read_excel('虚假_真实_sampo翻后.xlsx')
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


# In[4]:


#设定标签
false_targets = np.ones((len(false_text)))
true_targets = np.zeros((len(true_text)))
targets = np.concatenate((false_targets, true_targets), axis=0).reshape(-1, 1)   #(10000, 1)
total_targets = torch.tensor(targets)
total_targets


# In[5]:


#使用哈工大bert中文预训练模型
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', cache_dir="E:/transformer_file/")


# In[6]:


#查看处理效果
print(false_text[2])
print(tokenizer.tokenize(false_text[2]))

print(tokenizer.encode(false_text[2]))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(false_text[2])))


# In[7]:


#将每一句转成数字（大于130做截断，小于130做PADDING，加上首尾两个标识，长度总共等于122）
def convert_text_to_token(tokenizer, sentence, limit_size=130):
  
   tokens = tokenizer.encode(sentence[:limit_size])  #直接截断  
   if len(tokens) < limit_size + 2:                  #补齐（pad的索引号就是0）
       tokens.extend([0] * (limit_size + 2 - len(tokens)))   
   return tokens
#应用函数
input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]
input_tokens = torch.tensor(input_ids)
#查看文本状态
print(input_tokens.shape)                  


# In[8]:


#建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks

atten_masks = attention_masks(input_ids)
attention_tokens = torch.tensor(atten_masks)    


# In[9]:


#####################################


# In[10]:


#查看encode mask后效果
split_seed=156
from sklearn.model_selection import train_test_split
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens, total_targets, random_state=666, test_size=0.2)
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens, random_state=666, test_size=0.2)
print(train_inputs.shape, test_inputs.shape)      #torch.Size([8000, 128]) torch.Size([2000, 128])
print(train_masks.shape)                          #torch.Size([8000, 128])和train_inputs形状一样

print(train_inputs[0])
print(train_masks[0])


# In[11]:


#数据读取
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


# In[12]:


#数据读取效果
for i, (train, mask, label) in enumerate(train_dataloader):
    print(train.shape, mask.shape, label.shape)              
    break
print('len(train_dataloader)=', len(train_dataloader))       


# In[13]:


##################


# In[14]:


#定义模型
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        #引入bert
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        #定义CNN参数，分别使用3、4、5提取不同粒度的特征
        self.conv1 = nn.Conv2d(1, 100, 3, padding=[2,0])
        self.conv2 = nn.Conv2d(1, 100, 4, padding=[3,0])
        self.conv3 = nn.Conv2d(1, 100, 5, padding=[4,0])
        self.dropout=nn.Dropout(p=0.5)
        self.fc = nn.Linear(100*598, 2)

    def forward(self, b_input_ids,b_input_mask):
        bert_out=self.bert(b_input_ids, attention_mask=b_input_mask)
        #拿出BERT结果
        out=bert_out['pooler_output']
        out=out.unsqueeze(1)
        out=out.unsqueeze(1)
        #进行卷积并激活
        x = [F.relu(self.conv1(out)), F.relu(self.conv2(out)), F.relu(self.conv3(out))]
        # 进行池化
        x = [F.max_pool2d(i, i.size(2)).squeeze(2) for i in x]
        x=[i.squeeze(0) for i in x]
        #连接各层
        x = torch.cat(x, 2)
        #添加dropout
        x = self.dropout(x)
        x=x.view(-1,100*598)
        #全连接输出概率
        logit = self.fc(x)
        
        return logit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)


# In[15]:


#查看模型参数
for name,param in model.named_parameters():
    print(name, param.shape)


# In[16]:


#设置参数学习率
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr = LEARNING_RATE, eps = EPSILON)


# In[17]:


#设置epoch
epochs = 15
#training steps 的数量: [number of batches] x [number of epochs]. 
total_steps = len(train_dataloader) * epochs

# 设计学习率scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


# In[18]:


################


# In[19]:


#计算准确率和召回率
def binary_acc(preds, labels):      #preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()      #eq里面的两个参数的shape=torch.Size([16])    
    acc = correct.sum().item() / len(correct)
    rec=torch.lt(torch.max(preds, dim=1)[1], labels.flatten()).float()
    if labels.flatten().sum().item()!=0:
        recall=1-rec.sum().item()/labels.flatten().sum().item()
    else:
        recall=-99
    return acc, recall


# In[20]:


import time
import datetime
def format_time(elapsed):    
    elapsed_rounded = int(round((elapsed)))    
    return str(datetime.timedelta(seconds=elapsed_rounded))   #返回 hh:mm:ss 形式的时间


# In[21]:


def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc,avg_rec = [],[],[]
    #进入训练模式
    model.train()
    for step, batch in enumerate(train_dataloader):

        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:            
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        
        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
        outputs = model(b_input_ids,b_input_mask)
        model.zero_grad()
        loss = torch.nn.functional.cross_entropy(outputs, b_labels.squeeze_())
        #计算准确率等
        avg_loss.append(loss.item())
        acc,recall= binary_acc(outputs, b_labels)
        avg_acc.append(acc)
        if recall!=-99:
            avg_rec.append(recall)
        #更新模型
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)      #大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()              #更新模型参数
        scheduler.step()              #更新learning rate
    
    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    avg_rec = np.array(avg_rec).mean()
    return avg_loss, avg_acc, avg_rec


# In[22]:


def evaluate(model):    
    avg_acc,avg_rec = [],[]    
    model.eval()         #表示进入测试模式
      
    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[2].long().to(device)
       
            outputs = model(b_input_ids,b_input_mask)
           
            acc,recall= binary_acc(outputs, b_labels)
            avg_acc.append(acc)
            if recall!=-99:
                avg_rec.append(recall)
    avg_acc = np.array(avg_acc).mean()
    avg_rec = np.array(avg_rec).mean()
    return avg_acc, avg_rec


# In[23]:


#初始化绘图变量
train_acc_arr=np.zeros(epochs)
train_recall_arr=np.zeros(epochs)
test_acc_arr=np.zeros(epochs)
test_recall_arr=np.zeros(epochs)


# In[24]:


for epoch in range(epochs):

    train_loss, train_acc,train_recall = train(model, optimizer)
    train_acc_arr[epoch]=train_acc
    train_recall_arr[epoch]=train_recall
    print('epoch={},训练准确率={}，损失={},recall={}'.format(epoch, train_acc, train_loss,train_recall))
    test_acc,test_recall = evaluate(model)
    test_acc_arr[epoch]=test_acc
    test_recall_arr[epoch]=test_recall
    print("epoch={},测试准确率={},recall={}".format(epoch, test_acc,test_recall))


# In[25]:


#绘制图像
import matplotlib.pyplot as plt
from pylab import *     
x=range(1,epochs+1)
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
plt.title("Bert-CNN") #标题

plt.show()


# In[26]:


#将BERT模型的结果复制过来绘图
test_acc_arr2=np.array([0.80568562, 0.83991081, 0.84816054, 0.80445931, 0.86867336,
       0.87591973, 0.87179487, 0.87569677, 0.88004459, 0.87714604,
       0.88149387, 0.88004459, 0.88439242, 0.88149387, 0.88149387])
test_recall_arr2=np.array([0.32169208, 0.69147374, 0.6159734 , 0.8335043 , 0.57893061,
       0.62415302, 0.76162714, 0.67013301, 0.69779001, 0.71383556,
       0.73833522, 0.75993632, 0.7653711 , 0.74889422, 0.7440633 ])


# In[28]:


#绘制对比图像
plt.figure(figsize=(20,10))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
plt.ylim(0.5, 1)  # 限定纵轴的范围
plt.plot(x, test_acc_arr, marker='*',  ms=10,label=u'BERT-CNN accuracy')
plt.plot(x, test_acc_arr2, marker='*', ms=10,label=u'BERT accuracy')
plt.plot(x, test_recall_arr, marker='o', mec='r', mfc='w',label=u'BERT-CNN recall')
plt.plot(x, test_recall_arr2, marker='o', mec='r', mfc='w',label=u'BERT recall')

plt.legend()  # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"epoch") #X轴标签
plt.ylabel("ratio") #Y轴标签
plt.title("comparison") #标题


# ## 模型应用

# In[27]:


def predict(sen):
    
   input_id = convert_text_to_token(tokenizer, sen)
   input_token =  torch.tensor(input_id).long().to(device)            #torch.Size([128])
    
   atten_mask = [float(i>0) for i in input_id]
   attention_token = torch.tensor(atten_mask).long().to(device)       #torch.Size([128])         
  
   output = model(input_token.view(1, -1), token_type_ids=None, attention_mask=attention_token.view(1, -1))     #torch.Size([128])->torch.Size([1, 128])否则会报错
   #print(output[0])
    
   return output[0]


# In[62]:


clf=model


# In[64]:


#模型应用效果
label =predict('外星人（ALIENWARE）全新x15R1笔记本电脑15.6英寸11代酷睿30显卡独显电竞吃鸡游戏 1778:11代i7/32G/3070/2K屏 官方标配。物超所值，商品设计完美，外观也很高大上，让人爱不释手。客服更是热情的没话说，这次购物很满意')
print('label:',label.item())


# In[66]:


#读取应用数据
Seven = pd.read_excel('弄完、.xlsx')
comments=Seven['评价内容']
comments


# In[67]:


labels_new=[]
for com in comments:
    temp=predict(com)
    labels_new.append(temp.item())


# In[69]:


#应用结果
Seven['labels']=labels_new
Seven


# In[70]:


pd.DataFrame(Seven).to_csv('label后模型二数据fanyi1.csv', index=False)

