'''
Author: your name
Date: 2021-11-14 05:09:06
LastEditTime: 2021-11-15 02:04:41
LastEditors: Please set LastEditors
Description: 
    user sampling，根据词频统计，找到最相近/远的两个user
    通过pickle，保存所有用户的稀疏字典dict
        dict.key: tokenizer数组中的下标
        dict.value: 此token出现的频率
FilePath: /SIGIR2022/BOW_user_dict.py
'''
#user_path = '/home/zhengyi_ma/pcb/Data/PChatbot_byuser_filter'
# 先做user层面的词袋
user_path = '/home/zhaoheng_huang/SIGIR2022/data'
output = './BOW_user_dict_small.pkl'
import os
import numpy as np
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("/home/zhaoheng_huang/SIGIR2022/pretrained_dialog_bert")
from tqdm import tqdm
from collections import defaultdict
#dict = defaultdict(lambda: 0)
dict = {}
for file in tqdm(os.listdir(user_path)):                   
    abs_user_path = os.path.join(user_path, file)
    #BOW_dict = np.zeros(tokenizer.vocab_size)
    BOW_dict = {}
    ids_list = []
    with open(abs_user_path, 'r') as fp:                    #* 打开一个文件
        for index, line in enumerate(fp):
            if index >= 100: break          #dialog超过100的截断
            line = line.split('\t')
            p, p_id, p_t, r, r_id, r_t, _, _ = line
            p_str = tokenizer(p.replace(' ', ''))["input_ids"]      #* -> list(int)
            if len(p_str) >= 2:
                p_str = p_str[1:-1]
            r_str = tokenizer(r.replace(' ', ''))["input_ids"]
            if len(r_str) >= 2:
                r_str = r_str[1:-1]
            # print(p_str, r_str)
            # ids_list.append(p_str)
            # ids_list.append(r_str)
            for id in p_str:
                if BOW_dict.get(id) == None:
                    BOW_dict[id] = 0
                else:
                    BOW_dict[id] += 1
            for id in r_str:
                if BOW_dict.get(id) == None:
                    BOW_dict[id] = 0
                else:
                    BOW_dict[id] += 1
    #print(BOW_dict)
    dict[file] = BOW_dict
import pickle
pickle.dump(dict, open(output, 'wb'))