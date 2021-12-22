'''
Author: your name
Date: 2021-10-28 12:17:07
LastEditTime: 2021-12-22 08:36:18
LastEditors: Please set LastEditors
Description: 
        生成json_data下的json文件
FilePath: /SIGIR2022/preprocess/generateJson.py
'''
import os, random, json
from numpy.core.defchararray import asarray
from transformers import BertTokenizer
from tqdm import tqdm
import codecs
import numpy as np
from collections import defaultdict
from seq2seq.utils.utils import *

def load_data(data_dirpath, logger):
    """根据dialog所在的目录，生成该目录下所有dialog文件的绝对路径并返回列表

    Args:
        data_dirpath (str): opt.data_dirpath

    Returns:
        abs_filepaths (list): user dialog文件的绝对路径列表
    """
    logger.info("loading user dialog data")
    filenames = sorted(os.listdir(data_dirpath))
    logger.info("loading user dialog file list complete! {0} users in total.".format(len(filenames)))
    abs_filepaths = []
    for filename in filenames:
        abs_filepaths.append(os.path.join(data_dirpath, filename))
    print("load {0} dialog file in total.".format(len(filenames)))
    print("-"*20)
    return abs_filepaths


#* 根据input_ids列表，用<MASK>(id: 103)随机替换部分元素，被替换掉的储存在labels中，
#* 否则labels储存-100, 在attention的时候，忽略掉值为-100的元素   
#* 返回input_ids, labels
def create_masks_for_sequence(input_ids, MASK_TOKEN_ID, opt):
    labels = [-100 for i in range(len(input_ids))]
    for i, input_id in enumerate(input_ids):
        if i == 0:
            continue
        if random.random() < opt.mlm_prob:
            labels[i] = input_ids[i]    #! 储存input_id，然后将原序列对应位置转化为<MASK>
            input_ids[i] = MASK_TOKEN_ID
    return input_ids, labels

#* 依据dialog文件绝对路径，生成对应json文件
#* 只存储opt.user_limit数量的用户

#TODO 疑似不需要这个函数！
def generatejsonfiles_rl_pretrain(opt, filelist, logger):
    if opt.user_limit != None:
        logger.info("rl_pretrain: generate json with user_limit: {0} (total: {1})".format(opt.user_limit, str(len(filelist))))
        filelist = filelist[:opt.user_limit]
    else:
        logger.info("rl_pretrain: generate json with all user files: {0}".format(str(len(filelist))))
        
    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")
    users_count = 0
    for curr_id, dialog_absfile in enumerate(tqdm(filelist, desc='RL pretrain generate files...')):
        user_datas = [] #* 当前用户的dialog data
        
        #* dialog_absfile: /home/zhaoheng_huang/SIGIR2022/data/28211.txt
        Filename = dialog_absfile.split('/')[-1].split('.')[0]

        #* 如果/home/zhaoheng_huang/SIGIR2022/json_data/finetune/28211.json存在，则continue
        if os.path.exists(os.path.join(opt.gen_rl_pretrain_data_outputdir, Filename+'.json')):     #TODO 修改存储路径
            continue
        
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        #mlm_labels_list = []
        #input_terms_list = []
        #pr_label_list = []
        user_datas_dict = {}
        with open(dialog_absfile, 'r') as fin:  #* 每次读一个文件，避免内存开销过大
            for i, line in enumerate(fin):
                #* 每次读一行
                cols = line.split("\t")
                if(len(cols) != 8):
                    #! 读到的行格式错误
                    continue
                p, p_id, p_t, \
                r, r_id, r_t, \
                _, _ = cols
                fulltext_p = p
                fulltext_r = r
                #* 处理post
                
                #encoded = tokenizer.encode_plus(
                #    fulltext_p, 
                #    add_special_tokens=True,
                #    truncation=True,
                #    max_length=512,
                #    padding='max_length')
                #input_ids = encoded['input_ids']
                #token_type_ids = encoded['token_type_ids']
                #attention_mask = encoded['attention_mask']
                #input_ids, mlm_labels = create_masks_for_sequence(input_ids, MASK_TOKEN_ID, opt)
                #input_ids_list.append(input_ids)
                #token_type_ids_list.append(token_type_ids)
                #attention_mask_list.append(attention_mask)
                #mlm_labels_list.append(mlm_labels)
                #input_terms_list.append(tokenizer.convert_ids_to_tokens(input_ids))
                #pr_label_list.append(0)
                
                #* 处理response
                encoded = tokenizer.encode_plus(
                    fulltext_r, 
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length')
                input_ids = encoded['input_ids']
                token_type_ids = encoded['token_type_ids']
                attention_mask = encoded['attention_mask']
                input_ids, mlm_labels = create_masks_for_sequence(input_ids, MASK_TOKEN_ID, opt)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
                #mlm_labels_list.append(mlm_labels)
                #input_terms_list.append(tokenizer.convert_ids_to_tokens(input_ids))
                #pr_label_list.append(1)
                
            
            user_datas_dict["input_ids"] = input_ids_list
            user_datas_dict["token_type_ids"] = token_type_ids_list
            user_datas_dict["attention_mask"] = attention_mask_list
            #user_datas_dict["mlm_labels"] = mlm_labels_list
            #user_datas_dict["input_terms"] = input_terms_list
            #user_datas_dict["pr_label"] = pr_label_list
            
        user_datas.append(user_datas_dict)
        if not os.path.exists(opt.gen_rl_pretrain_data_outputdir):
            os.mkdir(opt.gen_rl_pretrain_data_outputdir)
            logger.info("已创建rl pretrain json data文件夹")
        with codecs.open(os.path.join(opt.gen_rl_pretrain_data_outputdir, Filename+'.json'), 'w', encoding='utf-8') as fout:
            for d in user_datas:            #此处应该只写入一条（每个用户只有一个字典而非len(dialog)条字典
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            users_count += 1
    logger.info("已写入新rl pretrain json文件数量: {0}".format(users_count))
    print("writing new rl_pretrain files: {0}".format(users_count))


def generatejsonfiles_bert_pretrain(opt, filelist, logger):
    if opt.user_limit != None:
        logger.info("bert_pretrain: generate json with user_limit: {0} (total: {1})".format(opt.user_limit, str(len(filelist))))
        filelist = filelist[:opt.user_limit]
    else:
        logger.info("bert_pretrain: generate json with all user files: {0}".format(str(len(filelist))))
    
    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")
    users_count = 0
    for curr_id, dialog_absfile in enumerate(tqdm(filelist, desc='BERT pretrain generate files...')):
        user_datas = [] #* 当前用户的dialog data
        
        #* dialog_absfile: /home/zhaoheng_huang/SIGIR2022/data/28211.txt
        Filename = dialog_absfile.split('/')[-1].split('.')[0]

        #* 如果/home/zhaoheng_huang/SIGIR2022/json_data/28211.json存在，则continue
        if os.path.exists(os.path.join(opt.gen_data_outputdir, Filename+'.json')):
            continue
        
        with open(dialog_absfile, 'r') as fin:
            #* 每次读一个文件，避免内存开销过大
            for i, line in enumerate(fin):
                #* 每次读一行
                cols = line.split("\t")
                if(len(cols) != 8):
                    #! 读到的行格式错误
                    continue
                p, p_id, p_t, \
                r, r_id, r_t, \
                _, _ = cols
                fulltext_p = p
                fulltext_r = r
                #* 处理post
                #TODO pretrain阶段加入post训练bert，finetune阶段不引入post，因此需要在generate dialog bert之后把此段注释掉
                
                encoded = tokenizer.encode_plus(
                    fulltext_p, 
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length')
                input_ids = encoded['input_ids']
                token_type_ids = encoded['token_type_ids']
                attention_mask = encoded['attention_mask']
                input_ids, mlm_labels = create_masks_for_sequence(input_ids, MASK_TOKEN_ID, opt)
                data = {
                    #TODO// 加一个label表明它是post或者是response
                    "input_ids": input_ids, 
                    "token_type_ids": token_type_ids, 
                    "attention_mask": attention_mask, 
                    "mlm_labels": mlm_labels, 
                    "input_terms": tokenizer.convert_ids_to_tokens(input_ids),
                    "pr_label": 0               #* pr_label = 0 ==> post
                    #TODO 添加当前dialog在文件中的位置下标
                }
                user_datas += [data]
                
                #* 处理response
                encoded = tokenizer.encode_plus(
                    fulltext_r, 
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length')
                input_ids = encoded['input_ids']
                token_type_ids = encoded['token_type_ids']
                attention_mask = encoded['attention_mask']
                input_ids, mlm_labels = create_masks_for_sequence(input_ids, MASK_TOKEN_ID, opt)
                data2 = {
                    "input_ids": input_ids, 
                    "token_type_ids": token_type_ids, 
                    "attention_mask": attention_mask, 
                    "mlm_labels":mlm_labels, 
                    "input_terms": tokenizer.convert_ids_to_tokens(input_ids),
                    "pr_label": 1               #* pr_label = 1 ==> response
                    #TODO 添加当前dialog在文件中的位置下标
                }
                user_datas += [data2]
        if not os.path.exists(opt.gen_data_outputdir):
            os.mkdir(opt.gen_data_outputdir)
            logger.info("已创建bert pretrain json data文件夹")
        with codecs.open(os.path.join(opt.gen_data_outputdir, Filename+'.json'), 'w', encoding='utf-8') as fout:
            for d in user_datas:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            users_count += 1
    logger.info("已写入新bert_pretrain json文件数量: {0}".format(users_count))
    print("writing new bert_pretrain files: ", users_count)
    print("-"*20)


def get_sim(s1, s2):
    """计算两个布尔向量的相似度(同时出现【1】时才计算)
    e.g.
    s1 = [0, 0, 1, 1]
    s2 = [1, 1, 1, 0]
    res = 1/4
    Args:
        s1 ([vocab_size]): 第1个句子
        s2 ([vocab_size]): 第2个句子
    """
    length = len(s1)
    if len(s1) != len(s2):
        raise ValueError("Length error!")
    count = 0
    for i in range(length):
        if s1[i] == 1 and s2[i] == 1:
            count += 1
    return count/length

def User_sentence_handler(User_sentence_list, num=3, truncate=15):
    """生成匹配的sentence pair

    Args:
        User_sentence_list ([response_dialog, vocab_size]): 单个用户对应的每条response的vocab list，目标计算response之间的相关性，从而返回
        num (int): 默认返回正pair的长度为3
        truncate (int): 每个用户仅仅随机抽取最多15条dialog计算相似度

    Returns:
        res_i_list[list] 
        res_j_list[list]
    """
    res_i_list = []
    res_j_list = []
    if len(User_sentence_list) < truncate:  #不需要截断数据
        picked_size = len(User_sentence_list)
        matrix = np.zeros((len(User_sentence_list), len(User_sentence_list)))
        res = []
        for i in range(len(User_sentence_list)):
            for j in range(i, len(User_sentence_list)):
                if i == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = get_sim(User_sentence_list[i], User_sentence_list[j])
                    res.append([i, j, matrix[i][j]])
                    matrix[j][i] = matrix[i][j]
        res = sorted(res, key=lambda x: -x[2])
        res = res[:num]
        for i, j, value in res:
            res_i_list.append(i)
            res_j_list.append(j)
        return res_i_list, res_j_list
        
    else:           # 需要截断一部分数据，随机选择truncate条
        picked_size = truncate
        derange_list = sorted(random.sample(range(len(User_sentence_list)), truncate))
        #[1, 3, 5, 6, ...]
        matrix = np.zeros((picked_size, picked_size))
        res = []
        for index_i, i in enumerate(derange_list):
            for index_j, j in enumerate(derange_list):
                if index_i > index_j: continue
                if index_i == index_j:
                    matrix[index_i][index_j] = 0
                else:
                    matrix[index_i][index_j] = get_sim(User_sentence_list[i], User_sentence_list[j])
                    res.append([i, j, matrix[index_i][index_j]])
                    matrix[index_j][index_i] = matrix[index_i][index_j]
        res = sorted(res, key=lambda x: -x[2])
        res = res[:num]
        for i, j, value in res:
            res_i_list.append(i)
            res_j_list.append(j)
        return res_i_list, res_j_list
    

def generatejsonfiles_users_rl_pretrain(
    opt, logger, filelist,
    output_dir = '/home/zhaoheng_huang/SIGIR2022/json_data/RLpretrain', 
    tokenizer_path = '/home/zhaoheng_huang/SIGIR2022/pretrained_dialog_bert'):
    # user_path = '/home/zhengyi_ma/pcb/Data/PChatbot_byuser_filter'
    # user_path = '/home/zhaoheng_huang/SIGIR2022/data'
    """
    **每一个用户**维护以下字典，后续__getitem__返回前要将数据切割好
    "response_list_input_ids": tensor(total_len_u1, 512)                   #total_len_u1指的是该用户所有response的数量(先不考虑post)
    "response_list_attention_mask": tensor(total_len_u1, 512)
    "response_list_token_type_ids": tensor(total_len_u1, 512)
    "split_size": total_len_u1         
    "paired_response_i_input_ids": tensor(paired_length_u1, 512)                          #匹配的句子对数量
    "paired_response_i_attention_mask": tensor(paired_length_u1, 512)                          #匹配的句子对数量
    "paired_response_i_token_type_ids": tensor(paired_length_u1, 512)                          #匹配的句子对数量
    "paired_response_j_input_ids": tensor(paired_length_u1, 512)                          #匹配的句子对数量
    "paired_response_j_attention_mask": tensor(paired_length_u1, 512)                          #匹配的句子对数量
    "paired_response_j_token_type_ids": tensor(paired_length_u1, 512)                          #匹配的句子对数量
    
    "pos_response_list_input_ids": tensor(total_len_u2, 512)                   #total_len_u2指的是该用户所有response的数量(先不考虑post)
    "pos_response_list_attention_mask": tensor(total_len_u2, 512)
    "pos_response_list_token_type_ids": tensor(total_len_u2, 512)
    "pos_split_size": total_len_u2                    
    """
    logger.info("entering phase: rl pretrain")
    if opt.user_limit != None:
        logger.info("rl_pretrain: generate json with user_limit: {0} (total: {1})".format(opt.user_limit, str(len(filelist))))
        filelist = filelist[:opt.user_limit]
    else:
        logger.info("rl_pretrain: generate json with all user files: {0}".format(str(len(filelist))))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    file2cbow = {}          # 文件名 -> [vocab], 词袋向量
    file2lines = {}         # 文件名 -> 该文件的dialog数目
    file2boolmatrix = {}    # 文件名 -> [response, vocab]，每一行的词出现的布尔向量
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")
    for abs_user_path in tqdm(filelist, desc='RL pretrain generate files'):         
        BOW_vector = np.zeros(tokenizer.vocab_size)             # 初始化user词袋，定义在for line前面
        ids_list = []
        Filename = abs_user_path.split('/')[-1].split('.')[0]
        if os.path.exists(os.path.join(output_dir, Filename+'.json')):
            continue        # json文件已存在
        with open(abs_user_path, 'r') as fp:                    # 打开一个文件
            lines = 0
            total_sentence_bag = []
            for line in fp:                                     # 读文件的每一行
                sentence_single_bag = np.zeros(tokenizer.vocab_size)    #初始化sentence词袋，定义在for line后面
                lines += 1
                line = line.split('\t')
                p, p_id, p_t, r, r_id, r_t, _, _ = line
                
                
                p_str = tokenizer(p.replace(' ', ''))["input_ids"]      #* -> list(int)
                if len(p_str) >= 2:
                    p_str = p_str[1:-1]                                 #* 统计词频时，去掉首尾被tokenizer带上的特殊标记
                r_str = tokenizer(r.replace(' ', ''))["input_ids"]
                if len(r_str) >= 2:
                    r_str = r_str[1:-1]
                for id in p_str:
                    BOW_vector[id] += 1
                for id in r_str:
                    BOW_vector[id] += 1             # user词袋
                    sentence_single_bag[id] = 1     # sentence词袋，只计算该词是否出现
                total_sentence_bag.append(sentence_single_bag)
        file2cbow[Filename+".txt"] = BOW_vector
        file2lines[Filename+".txt"] = lines
        file2boolmatrix[Filename+".txt"] = total_sentence_bag        # 记录统计file内每一行的词出现布尔向量
        
    User_list = []
    for filename in file2cbow:   #* filename: xxx.txt
        User_list.append([filename, file2cbow[filename], file2boolmatrix[filename]])    #user词袋和sentence词袋都加入进来
        
    m = np.zeros((len(User_list), len(User_list)))
    #TODO 在这里完成user词袋，【此处比较慢】(O(N^2))，建议修改
    logger.debug("calculate user similarity...WARNING: it might take a lot of time")                
    for i in tqdm(range(len(User_list)-1), desc="RL pretrain calculate user similarity"):
        m[i][i] = 100.0
        for j in range(i+1, len(User_list)):
            m[i][j] = cosine_similarity(User_list[i][1], User_list[j][1])
            m[j][i] = cosine_similarity(User_list[i][1], User_list[j][1])
    logger.debug("generate user json files...WARNING: it might take a lot of time to generate paired sentences")
    for i in tqdm(range(len(User_list) - 1), desc="RL pretrain generate sentence similarity"):
        #TODO 在这里完成sentence词袋，【此处比较慢】，可以设置截断前10条dialog等等
        #* filename: User_list[i][0], 28211.txt
        _m = m[i].copy()
        _m[i] = 0           # 不能与自己最相似
        Max = np.argmax(_m)
        """        
        print(User_list[i][0],  '==> Max: ', User_list[Max][0], ": ", m[i][Max],
                                '==> Min: ', User_list[Min][0], ": ", m[i][Min]
        )
        """
        split_size = 0
        response_list_input_ids = []
        response_list_attention_mask = []
        response_list_token_type_ids = []
        with open(os.path.join(opt.data_dirpath, User_list[i][0]), 'r') as fp:
            index = 0
            for line in fp:
                index += 1
                line = line.split('\t')
                p, p_id, p_t, r, r_id, r_t, _, _ = line
                #TODO 这里可以加入对post的处理
                encoded = tokenizer.encode_plus(
                    r, 
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                )
                input_ids_sentence = encoded['input_ids']   #[512]
                attention_mask_sentence = encoded['attention_mask']
                token_type_ids_sentence = encoded['token_type_ids']
                input_ids_sentence, mlm_labels = create_masks_for_sequence(input_ids_sentence, MASK_TOKEN_ID, opt)
                response_list_input_ids.append(input_ids_sentence)
                response_list_attention_mask.append(attention_mask_sentence)
                response_list_token_type_ids.append(token_type_ids_sentence)
            #! RuntimeError: stack expects each tensor to be equal size
            if index >= 5:
                split_size = 5
            else:
                split_size = index
            
        pos_split_size = 0
        pos_response_list_input_ids = []
        pos_response_list_attention_mask = []
        pos_response_list_token_type_ids = []
        with open(os.path.join(opt.data_dirpath, User_list[Max][0]), 'r') as fp:    #pos user
            index = 0
            for line in fp:
                index += 1
                line = line.split('\t')
                p, p_id, p_t, r, r_id, r_t, _, _ = line
                #TODO 这里可以加入对post的处理
                encoded = tokenizer.encode_plus(
                    r, 
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                )
                pos_input_ids_sentence = encoded['input_ids']   #[512]
                pos_token_type_ids_sentence = encoded['token_type_ids']
                pos_attention_mask_sentence = encoded['attention_mask']
                pos_input_ids_sentence, mlm_labels = create_masks_for_sequence(pos_input_ids_sentence, MASK_TOKEN_ID, opt)
                pos_response_list_input_ids.append(pos_input_ids_sentence)
                pos_response_list_attention_mask.append(pos_attention_mask_sentence)
                pos_response_list_token_type_ids.append(pos_token_type_ids_sentence)
            #! RuntimeError: stack expects each tensor to be equal size
            if index >= 5:
                pos_split_size = 5
            else:
                pos_split_size = index
        
        res_i_list, res_j_list = User_sentence_handler(User_list[i][2])
        paired_response_i_input_ids = np.array(response_list_input_ids)[res_i_list]
        paired_response_i_attention_mask = np.array(response_list_attention_mask)[res_i_list]
        paired_response_i_token_type_ids = np.array(response_list_token_type_ids)[res_i_list]
        paired_response_j_input_ids = np.array(response_list_input_ids)[res_j_list]
        paired_response_j_attention_mask = np.array(response_list_attention_mask)[res_j_list]
        paired_response_j_token_type_ids = np.array(response_list_token_type_ids)[res_j_list]
        with codecs.open(os.path.join(output_dir, User_list[i][0].split('.')[0]+'.json'), 'w', encoding='utf-8') as fout:
            # for d in user_datas:
            #     fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            # users_count += 1
            user_json_data = {
                "response_list_input_ids": response_list_input_ids,
                "response_list_attention_mask": response_list_attention_mask,
                "response_list_token_type_ids": response_list_token_type_ids,
                "split_size": [split_size],
                #TODO 传入split_paired_size
                "paired_response_i_input_ids": paired_response_i_input_ids.tolist(),
                "paired_response_i_attention_mask": paired_response_i_attention_mask.tolist(),
                "paired_response_i_token_type_ids": paired_response_i_token_type_ids.tolist(),
                "paired_response_j_input_ids": paired_response_j_input_ids.tolist(),
                "paired_response_j_attention_mask": paired_response_j_attention_mask.tolist(),
                "paired_response_j_token_type_ids": paired_response_j_token_type_ids.tolist(),
                "pos_response_list_input_ids": pos_response_list_input_ids,
                "pos_response_list_attention_mask": pos_response_list_attention_mask,
                "pos_response_list_token_type_ids": pos_response_list_token_type_ids,
                "pos_split_size": [pos_split_size]
            }
            fout.write(json.dumps(user_json_data, ensure_ascii=False) + "\n")
            
            
            
            
def generatejsonfiles_finetune(
        opt, logger, filelist,
        output_dir, tokenizer_path
    ):
    """在finetune内，post与response均需要生成对应的input_ids, token_type_ids, attention_mask
    """
    TRUNCATE_SENTENCE_SIZE = 30         #每个句子最大长度
    
    logger.info("entering phase: finetune")
    if opt.user_limit != None:
        logger.info("finetune: generate json with user_limit: {0} (total: {1})".format(opt.user_limit, str(len(filelist))))
        filelist = filelist[:opt.user_limit]
    else:
        logger.info("finetune: generate json with all user files: {0}".format(str(len(filelist))))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    user_count = 0
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")
    for abs_user_path in tqdm(filelist, desc='finetune generate files'):
        Filename = abs_user_path.split('/')[-1].split('.')[0]       #28211
        if os.path.exists(os.path.join(output_dir, Filename+'.json')):
            continue        # 已经生成过该文件对应的json文件
        user_count += 1
        with open(abs_user_path, 'r') as fp:        # 打开一个文件
            lines = 0
            label_list = []       #(response_num, sentence_size, 21128)
            post_input_ids_list = []
            post_attention_mask_list = []
            post_token_type_ids_list = []
            response_input_ids_list = []
            response_attention_mask_list = []
            response_token_type_ids_list = []
            for line in fp:
                lines += 1
                line = line.split("\t")
                p, p_id, p_t, r, r_id, r_t, _, _ = line
                r_str = tokenizer(r.replace(' ', ''))["input_ids"]
                #TODO 记录r_str序列长度
                r_str = r_str[:TRUNCATE_SENTENCE_SIZE]
                #label = np.zeros((len(r_str), tokenizer.vocab_size))  
                #label = np.zeros((TRUNCATE_SENTENCE_SIZE, tokenizer.vocab_size))
                #for index, input_id in enumerate(r_str):
                #    label[index][input_id] = 1
                #label_list.append(label.tolist())
                label_list.append(r_str[:] + [0]*(TRUNCATE_SENTENCE_SIZE - len(r_str)))
                #post
                encoded_post = tokenizer.encode_plus(
                    p,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=TRUNCATE_SENTENCE_SIZE,
                    padding='max_length'
                )
                post_input_ids = encoded_post['input_ids']   #[truncate_size]
                post_token_type_ids = encoded_post['token_type_ids'] #[truncate_size]  句子编号
                post_attention_mask = encoded_post['attention_mask'] #[truncate_size]
                #post_input_ids, mlm_labels = create_masks_for_sequence(post_input_ids, MASK_TOKEN_ID, opt)
                post_input_ids_list.append(post_input_ids)
                post_token_type_ids_list.append(post_token_type_ids)
                post_attention_mask_list.append(post_attention_mask)
                
                #TODO
                encoded_response = tokenizer.encode_plus(
                    r,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=TRUNCATE_SENTENCE_SIZE,
                    padding='max_length'
                )
                response_input_ids = encoded_response['input_ids']  #[truncate_size]
                response_token_type_ids = encoded_response['token_type_ids']    #[truncate_size]
                response_attention_mask = encoded_response['attention_mask']    #[truncate_size]
                #response_input_ids, mlm_labels = create_masks_for_sequence(response_input_ids, MASK_TOKEN_ID, opt)
                response_input_ids_list.append(response_input_ids)
                response_token_type_ids_list.append(response_token_type_ids)
                response_attention_mask_list.append(response_attention_mask)
                
            with codecs.open(os.path.join(output_dir, Filename+'.json'), 'w', encoding='utf-8') as fout:
                # for d in user_datas:
                #     fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                # users_count += 1
                user_json_data = {
                    "label": label_list,        #[response_num, TRUNCATE_SIZE]
                    "post_input_ids": post_input_ids_list,      #[post_num, TRUNCATE_SIZE]
                    "post_attention_mask": post_attention_mask_list, #[post_num, TRUNCATE_SIZE]
                    "post_token_type_ids": post_token_type_ids_list,  #[post_num, TRUNCATE_SIZE]
                    "response_input_ids": response_input_ids_list,  #[response_num, TRUNCATE_SIZE]
                    "response_attention_mask": response_attention_mask_list,    #[response_num, TRUNCATE_SIZE]
                    "response_token_type_ids": response_token_type_ids_list     #[response_num, TRUNCATE_SIZE]
                }
                #for key in user_json_data:
                #    print(key, '==>', np.array(user_json_data[key]).shape)
                fout.write(json.dumps(user_json_data, ensure_ascii=False) + "\n")
    logger.info("已写入新finetune json文件数量: {0}".format(user_count))
    print("writing new finetune files: ", user_count)
    print("-"*20)