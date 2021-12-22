'''
Author: your name
Date: 2021-10-20 02:51:45
LastEditTime: 2021-12-22 09:44:24
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /SIGIR2022/dialogDatasets.py
'''
import random
import linecache
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import json
from tqdm import tqdm
import numpy as np
import operator
import datasets
import torch
from torch.utils.data import Dataset
import numpy as np

def Generate_dialog_dataset_V2(pretrained_dialog_dir):
    """读取json_data/finetune下的dialog文件（修改后的，20211206），并生成一个字典返回

    Args:
        pretrained_dialog_dir (str): json_data/finetune
    """
    print("Generate dialog dataset...")
    res = {}
    file_list = os.listdir(pretrained_dialog_dir)
    abs_file_list = [os.path.join(pretrained_dialog_dir, file) for file in file_list]
    for filename in tqdm(abs_file_list):
        file_int = int(filename.split('/')[-1].split('.')[0])   #28211(int)
        if res.get(file_int) == None:
            res[file_int] = {
                "input_ids": [],     #其中存储字典
                "token_type_ids": [],
                "attention_mask": [],
                #"mlm_labels": [],
                #"input_terms": [],
                #"pr_label": []
            }
        with open(filename, 'r') as fp:
            for index, line in enumerate(fp):
                #* 读取一行，然后转化为json
                dict = json.loads(line)
                res[file_int]["input_ids"] = dict["input_ids"]
                res[file_int]["token_type_ids"] = dict["token_type_ids"]
                res[file_int]["attention_mask"] = dict["attention_mask"]
                #res[file_int]["mlm_labels"] = dict["mlm_labels"]
                #res[file_int]["input_terms"] = dict["input_terms"]
                #res[file_int]["pr_label"] = dict["pr_label"]
    return res



def Generate_dialog_dataset_V2_20211206(pretrained_dialog_dir):
    """读取json_data/finetune下的dialog文件（20211129），并生成一个字典返回

    Args:
        pretrained_dialog_dir (str): json_data/finetune
    """
    print("Generate dialog dataset...")
    res = {}
    file_list = os.listdir(pretrained_dialog_dir)
    abs_file_list = [os.path.join(pretrained_dialog_dir, file) for file in file_list]
    for filename in tqdm(abs_file_list):
        file_int = int(filename.split('/')[-1].split('.')[0])   #28211(int)
        if res.get(file_int) == None:
            res[file_int] = {
                "input_ids": [],     #其中存储字典
                "token_type_ids": [],
                "attention_mask": [],
                "mlm_labels": [],
                "input_terms": [],
                "pr_label": []
            }
        with open(filename, 'r') as fp:
            for index, line in enumerate(fp):
                #* 读取一行，然后转化为json
                dict = json.loads(line)
                res[file_int]["input_ids"] = dict["input_ids"]
                res[file_int]["token_type_ids"] = dict["token_type_ids"]
                res[file_int]["attention_mask"] = dict["attention_mask"]
                res[file_int]["mlm_labels"] = dict["mlm_labels"]
                res[file_int]["input_terms"] = dict["input_terms"]
                res[file_int]["pr_label"] = dict["pr_label"]
    return res
    
def Generate_dialog_dataset(pretrained_dialog_dir):
    """读取json_data/pretrain下的dialog文件，并生成一个字典返回

    Args:
        pretrained_dialog_dir (str): json_data/pretrain
    """
    print("Generate dialog dataset...")
    res = {}
    file_list = os.listdir(pretrained_dialog_dir)
    abs_file_list = [os.path.join(pretrained_dialog_dir, file) for file in file_list]
    for filename in tqdm(abs_file_list):
        file_int = int(filename.split('/')[-1].split('.')[0])   #28211(int)
        if res.get(file_int) == None:
            res[file_int] = {
                "post": [],     #其中存储字典
                "response": []
            }
        with open(filename, 'r') as fp:
            for index, line in enumerate(fp):
                #* 读取一行，然后转化为json
                dict = json.loads(line)
                if dict["pr_label"] == 0:   #post
                    res[file_int]["post"].append(
                        {
                            "input_ids": dict["input_ids"],
                            "token_type_ids": dict["token_type_ids"],
                            "attention_mask": dict["attention_mask"],
                            "mlm_labels": dict["mlm_labels"],
                            "input_terms": dict["input_terms"]
                        }
                    )
                elif dict["pr_label"] == 1: #response
                    res[file_int]["response"].append(
                        {
                            "input_ids": dict["input_ids"],
                            "token_type_ids": dict["token_type_ids"],
                            "attention_mask": dict["attention_mask"],
                            "mlm_labels": dict["mlm_labels"],
                            "input_terms": dict["input_terms"]
                        }
                    )
                else:
                    print("dialogDatasets.Generate_dialog_dataset() Wrong!!!!")
    return res


class RL_pretrain_Dataset(Dataset):
    """为finetune部分生成数据集，需要预先准备好json文件

    Args:
        Dataset (torch.utils.data.Dataset): abstract super class

    Returns:
        None
    """
    def __init__(
        self, 
        filename,           #* user json文件所在的目录下所有文件组成的列表
        tokenizer
    ): 
        self._filename = filename
        self._tokenizer = tokenizer
        #TODO// 看yutao师兄在群里的回复！
        self.userdata = []
        for file in filename:
            data = open(file, 'r').readline()
            data = json.loads(data)
            self.userdata.append(data)
        self.total_len = len(self._filename)
        print("Init finetune dataset completed! total length: ", self.total_len) 

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index):
        # 返回一个user data
        data = self.userdata[index]
        #! RuntimeError: stack expects each tensor to be equal size
        #! 返回的长度应该相等
        Size = 5
        return {
            "response_list_input_ids": np.asarray(data["response_list_input_ids"])[np.random.choice(len(data["response_list_input_ids"]), size=Size, replace=False)],
            "response_list_attention_mask": np.asarray(data["response_list_attention_mask"])[np.random.choice(len(data["response_list_attention_mask"]), size=Size, replace=False)],
            "response_list_token_type_ids": np.asarray(data["response_list_token_type_ids"])[np.random.choice(len(data["response_list_token_type_ids"]), size=Size, replace=False)],
            "split_size": np.asarray(data["split_size"]),
            "paired_response_i_input_ids": np.asarray(data["paired_response_i_input_ids"]),
            "paired_response_i_attention_mask": np.asarray(data["paired_response_i_attention_mask"]),
            "paired_response_i_token_type_ids": np.asarray(data["paired_response_i_token_type_ids"]),
            "paired_response_j_input_ids": np.asarray(data["paired_response_j_input_ids"]),
            "paired_response_j_attention_mask": np.asarray(data["paired_response_j_attention_mask"]),
            "paired_response_j_token_type_ids": np.asarray(data["paired_response_j_token_type_ids"]),
            "pos_response_list_input_ids": np.asarray(data["pos_response_list_input_ids"])[np.random.choice(len(data["pos_response_list_input_ids"]), size=Size, replace=False)],
            "pos_response_list_attention_mask": np.asarray(data["pos_response_list_attention_mask"])[np.random.choice(len(data["pos_response_list_attention_mask"]), size=Size, replace=False)],
            "pos_response_list_token_type_ids": np.asarray(data["pos_response_list_token_type_ids"])[np.random.choice(len(data["pos_response_list_token_type_ids"]), size=Size, replace=False)],
            "pos_split_size": np.asarray(data["pos_split_size"])
        }

class FinetuneDataset_20211207(Dataset):
    """为finetune部分生成数据集，需要预先准备好json文件

    Args:
        Dataset (torch.utils.data.Dataset): abstract super class

    Returns:
        None
    """
    def __init__(
        self, 
        filename,           #* user json文件所在的目录下所有文件组成的列表
        tokenizer,
        dataset_script_dir,
        dataset_cache_dir
    ): 
        self._filename = filename
        self._tokenizer = tokenizer
        #self.dialog_dataset = Generate_dialog_dataset(pretrained_dialog_dir)
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files = self._filename,
            ignore_verifications = False,
            cache_dir = dataset_cache_dir,
            features = datasets.Features({
                "filename": datasets.Value("int32"),            #* e.g. 28211
                "pos_user_jsfilename": datasets.Value("int32"),
                "paired_responses_i": [datasets.Value("int32")],    #TODO [(i1, j1), (i2, j2), ...] 多补充几个pair
                "paired_responses_j": [datasets.Value("int32")],
                "dialog_num": datasets.Value("int32")
            })
        )['train']
        self.total_len = len(self.nlp_dataset)
        print("Init finetune dataset completed! total length: ", self.total_len)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index):
        data = self.nlp_dataset[index]
        #print(np.array(data["filename"]))
        return {
            "filename": data["filename"],
            "pos_user_jsfilename":  data["pos_user_jsfilename"], 
            #"pos_user_similarity":  data["pos_user_similarity"],
            #"neg_user_jsfilename":  data["neg_user_jsfilename"],
            #"neg_user_similarity":  data["neg_user_similarity"],
            "paired_responses_i":   data["paired_responses_i"],
            "paired_responses_j":   data["paired_responses_j"],
            #"unpaired_responses_i": data["unpaired_responses_i"],
            #"unpaired_responses_j": data["unpaired_responses_j"],
            "dialog_num":           data["dialog_num"]
        }
        
        
        
        
class FinetuneDataset(Dataset):
    def __init__(
        self, 
        filename,
        logger,
        drange_size=10
    ):
        self._filename = filename
        self.userdata = []
        self.drange_size = drange_size
        for file in filename:
            data = open(file, 'r').readline()       # 每个文件只有一行
            #print(data)
            data = json.loads(data)
            self.userdata.append(data)
        self.total_len = len(self._filename)
        logger.info("finetune: Init finetune dataset completed! total length: {0}".format(self.total_len))
    def __len__(self):
        return self.total_len
    def __getitem__(self, index):
        data = self.userdata[index]
        size = np.asarray(data["label"], dtype=int).shape[0]  #该用户的语句数量
        drange = list(random.sample(range(size), self.drange_size))        #? 每个用户在finetune的时候随机选出10个句子
        return {
            "label": np.asarray(data['label'], dtype=int)[drange],
            "post_input_ids": np.asarray(data['post_input_ids'], dtype=int)[drange],
            "post_attention_mask": np.asarray(data['post_attention_mask'], dtype=int)[drange],
            "post_token_type_ids": np.asarray(data['post_token_type_ids'], dtype=int)[drange],
            "response_input_ids": np.asarray(data['response_input_ids'], dtype=int)[drange],
            "response_attention_mask": np.asarray(data['response_attention_mask'], dtype=int)[drange],
            "response_token_type_ids": np.asarray(data['response_token_type_ids'], dtype=int)[drange]
        }
        
        
        
        
        
        
        
        
        
        
        
class PretrainDataset(Dataset):
    def __init__(
        self, 
        filename, 
        max_seq_length, 
        tokenizer,
        logger
    ):
        self._filename = filename
        self._tokenizer = tokenizer
        self.userdata = []
        for file in filename:
            data = open(file, 'r').readline()
            data = json.loads(data)
            self.userdata.append(data)
        self.total_len = len(self._filename)
        logger.info("BERT pretrain: Init pretrain dataset completed! total length: ", self.total_len) 
    def __len__(self):
        return self.total_len
    def __getitem__(self, index):
        data = self.userdata[index]
        return {
            "input_ids": np.asarray(data['input_ids']),
            "token_type_ids": np.asarray(data['token_type_ids']),
            "attention_mask": np.asarray(data['attention_mask']),
            "mlm_labels": np.asarray(data['mlm_labels']),
            "pr_label": np.asarray(data['pr_label'])
        }

class PretrainDataset_20211212(Dataset):
    def __init__(
        self, 
        filename, 
        max_seq_length, 
        tokenizer, 
        dataset_script_dir, 
        dataset_cache_dir
    ):
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        #* 将json_data/下的文件转化为datasets类实例（通过脚本/dataset_script_dir/json.py完成）
        #* 即：将来读取数据时，读取的都是nlp_dataset对象。该对象需要与json_data/下的文件一一对应
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files = self._filename,        #* 待处理的json文件
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                #? json文件中每一行只有这5个键！可以适当的改造之
                'input_ids': [datasets.Value("int32")],
                'token_type_ids': [datasets.Value("int32")],
                'attention_mask': [datasets.Value("int32")],
                'mlm_labels': [datasets.Value("int32")],
                'input_terms': [datasets.Value("string")],     # 最后一个元素需要加逗号！
                'pr_label': datasets.Value("int32"),
                #?// 加一个label表示它是post还是response
            })
        )['train']
        self.total_len = len(self.nlp_dataset)
      
    def __len__(self):
        """overwrite len() method

        Returns:
            int: len(self.nlp_dataset)
        """
        return self.total_len
    
    def __getitem__(self, item):
        """overwrite [] method

        Args:
            item (int): index of self.nlp_dataset

        Returns:
            dict: a batch that includes: input_ids, token_type_ids, attention_mask, mlm_labels
        """
        data = self.nlp_dataset[item]
        return {
            "input_ids": np.array(data['input_ids']),
            "token_type_ids": np.array(data['token_type_ids']),
            "attention_mask": np.array(data['attention_mask']),
            "mlm_labels": np.array(data['mlm_labels']),
            "pr_label": np.array(data['pr_label'])
        }