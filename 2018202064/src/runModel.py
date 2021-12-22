'''
Author: zhaoheng_huang
Date: 2021-10-17 06:35:33
LastEditTime: 2021-12-22 09:41:51
LastEditors: Please set LastEditors
Description: runModel
FilePath: /zhaoheng_huang/SIGIR2022/runModel.py
'''
import argparse

parser = argparse.ArgumentParser()  #生成ArgumentParser实例: parser
#! 数据集目录路径
default_dataset_dirpath = './data'
#default_dataset_dirpath = '/home/zhengyi_ma/pcb/Data/PChatbot_byuser_filter'
#! 结果保存文件名
default_results_filepath = './results/res.txt'
#! preprocess
parser.add_argument('--data_dirpath',                   action='store', dest='data_dirpath', help='user dataset的目录路径', default=default_dataset_dirpath)
parser.add_argument('--gen_userdata_outputdir',         action='store', dest='gen_userdata_outputdir', type=str, default="/home/zhaoheng_huang/SIGIR2022/json_data/RLpretrain", help="生成采样users的json文件")
parser.add_argument('--user_limit',                     action='store', dest='user_limit', type=int, default='3000000', help="读取data/的用户总数不能超过user_limit")
parser.add_argument('--gen_data_outputdir',             action='store', dest='gen_data_outputdir', type=str, default="/home/zhaoheng_huang/SIGIR2022/json_data/BERTpretrain", help="生成BERTpretrain json数据之后存储的目录")
parser.add_argument('--gen_finetunedata_outputdir',     action='store', dest='gen_finetunedata_outputdir', type=str, default='/home/zhaoheng_huang/SIGIR2022/json_data/finetune', help='生成finetune json文件的目录')
#parser.add_argument('--gen_rl_pretrain_data_outputdir',    action='store', dest='gen_rl_pretrain_data_outputdir', type=str, default="/home/zhaoheng_huang/SIGIR2022/json_data/RLpretrain", help="生成rl pretrain的数据保存目录")
#! BERT pretrain
parser.add_argument('--bert_dir',                       action='store', dest='bert_dir', type=str, help='加载BERT预训练模型的目录')
parser.add_argument('--bert_pretrain_epochs',           action='store', dest='bert_pretrain_epochs', type=int, default=4, help="训练轮次")
parser.add_argument('--bert_pretrain_batch_size',       action='store', dest='bert_pretrain_batch_size', type=int, default=5, help='bert pretrain的batchsize')
parser.add_argument('--save_path',                      action='store', dest='save_path', type=str, default='/home/zhaoheng_huang/data/pretrained_dialog_bert', help="pretrain后的BERT模型保存目录")
parser.add_argument('--mlm_prob',                       action='store', dest='mlm_prob', type=float, default=0.15 ,help="pretrain bert生成数据MASK的概率")
#! 对比学习pretrain
parser.add_argument('--rl_pretrain_epochs',             action='store', dest='rl_pretrain_epochs', type=int, default=2, help="rl pretrain训练轮次")
parser.add_argument('--rl_pretrain_batch_size',         action='store', dest='rl_pretrain_batch_size', type=int, default=3, help="rl pretrain训练的batch size")
parser.add_argument('--rl_pretrain_modeldir',           action='store', dest='rl_pretrain_modeldir', type=str, default='/home/zhaoheng_huang/SIGIR2022/rl_pretrainmodel')
#! finetune
parser.add_argument('--finetune_batch_size',            action='store', dest='finetune_batch_size', type=int, default=3, help="finetune训练batch size")
parser.add_argument('--finetune_epochs',                action='store', dest='finetune_epochs', type=int, default=20, help='finetune训练轮次')
parser.add_argument('--finetune_output_modeldir',       action='store', dest='finetune_output_modeldir', type=str, default='/home/zhaoheng_huang/SIGIR2022/finetunemodel')
parser.add_argument('--finetune_learning_rate',         action='store', dest='finetune_learning_rate', type=float, default=1e-2)
#! infer
parser.add_argument('--infer_outputdir',                action='store', dest='infer_outputdir', type=str, default='/home/zhaoheng_huang/SIGIR2022/infer_output')
parser.add_argument('--infer_batch_size',               action='store', dest='infer_batch_size', type=int, default=15)
#! 其他模型参数
#parser.add_argument('--learning_rate',     action='store', dest='learning_rate', type=float, default=5e-5, help="学习率")
parser.add_argument('--learning_rate',                  action='store', dest='learning_rate', type=float, default=5e-3, help="学习率")
parser.add_argument('--random_seed',                    action='store', dest='random_seed', default=None, help='Random seed随机种子', type=int)
parser.add_argument('--device',                         action='store', dest='device', default=None, help='GPU device编号')
parser.add_argument('--result_filepath',                action='store', dest='result_filepath', help='result file结果保存文件路径', default=default_results_filepath)
parser.add_argument('--log_dirpath',                    action='store', dest='log_dirpath', type=str, default='/home/zhaoheng_huang/SIGIR2022/log')
parser.add_argument('--phase',                          action='store', dest='phase', default='train', help='预训练[bert_pretrain/rl_pretrain]或精调[finetune]或直接生成response[infer]')
opt = parser.parse_args()

#TODO 5. BERT+微博语料完形填空，下游finetune，跑跨用户，loss4

#* 创建log文件夹，设置日志参数
import time, os, logging
nowtime_log = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())+".log"
if not os.path.exists(opt.log_dirpath):
    os.mkdir(opt.log_dirpath)
logging.basicConfig(
    level=logging.DEBUG,
    filename=os.path.join(opt.log_dirpath, nowtime_log),
    filemode='w',
    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger_pretrain = logging.getLogger("pretrain")
logger_finetune = logging.getLogger("finetune")

import torch, json, pickle, sys
from numpy.lib.utils import source
from tqdm import tqdm

multi_gpu = False
device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
if opt.random_seed is not None: torch.cuda.manual_seed_all(opt.random_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


#* 数据预处理：将tsv文件转化为能过bert pretrain的json文件
from preprocess.generateJson import *
from preprocess.pretrain import *
from preprocess.finetune import *
from preprocess.infer import *


#! load seq2seq, transformers model
from transformers import AdamW, get_linear_schedule_with_warmup
from seq2seq.loss import Perplexity #! 继承了nn.NLLLoss
from seq2seq.dataset.dialogDatasets import *
#from seq2seq.dataset.BertForPretrain import *
from seq2seq.models.BertForPretrain import *
from seq2seq.models.RLForPretrain import *
from seq2seq.models.InferModel import *

from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel



if __name__ == "__main__":
    print("-"*20)
    for k in list(vars(opt).keys()):
        print('| %s: %s' % (k, vars(opt)[k]))
    print("-"*20)
    logging.info("print argparser")
    logging.info("into preprocessing...")
    if opt.phase != 'finetune' and opt.phase != 'infer':
        file_list = load_data(opt.data_dirpath, logging)             # 加载dialog文件
        generatejsonfiles_bert_pretrain(opt, file_list, logging)          # 生成pretrain json文件
    #疑似不需用此文件generatejsonfiles_rl_pretrain(opt, file_list, logging)       # 生成rl pretrain json文件
    
    if opt.phase == 'bert_pretrain':
        logger_pretrain.info("entering phase: bert pretrain")
        if os.path.exists(opt.save_path):                   # 已有bert模型
            logger_pretrain.info("find pretrained weibo model at: {0}".format(opt.save_path))
        else:   # 未有bert模型，则BERT pretrain, 并保存模型
            json_filelist = [os.path.join(opt.gen_data_outputdir, file) for file in os.listdir(opt.gen_data_outputdir)]
            pretrain_model(json_filelist, opt, logger_pretrain)                          
    elif opt.phase == 'rl_pretrain':
        generatejsonfiles_users_rl_pretrain(opt, logging, file_list, output_dir = opt.gen_userdata_outputdir, tokenizer_path = opt.save_path)     # 生成users文件以便rl pretrain
        #* user sampling，json结果文件保存在json_data/users目录下
        sentence_bert_model = BertModel.from_pretrained(opt.save_path)
        sentence_bert_model.encoder.layer = sentence_bert_model.encoder.layer[9:]   #取最后三层
        sequence_bert_model = BertModel.from_pretrained(opt.save_path)
        sequence_bert_model.encoder.layer = sequence_bert_model.encoder.layer[9:]   #取最后三层
        #* 加载sampled user json结果文件
        json_filelist = load_data(opt.gen_userdata_outputdir, logger_pretrain)
        tokenizer = BertTokenizer.from_pretrained(opt.save_path)
        RL_pretrain_model(json_filelist, opt, sentence_bert_model, sequence_bert_model, tokenizer, logger_pretrain)
    elif opt.phase == 'finetune':
        #* finetune
        checkpoint = "checkpoint5.pth"
        tokenizer = BertTokenizer.from_pretrained(opt.save_path)
        #json_filelist = load_data(opt.gen_finetunedata_outputdir, logger_finetune)
        json_filelist = load_data(opt.data_dirpath, logger_finetune)
        generatejsonfiles_finetune(opt, logger_finetune, json_filelist, opt.gen_finetunedata_outputdir, opt.save_path)
        #TODO 待优化：为避免出现model.parameters为空的情况出现，加载pretrain模型，然后取出其中几块，最后再load_state_dict加载参数
        sentence_bert_model = BertModel.from_pretrained(opt.save_path)  #先用Pretrained Bert初始化
        sentence_bert_model.encoder.layer = sentence_bert_model.encoder.layer[9:]
        sentence_bert_model.load_state_dict(torch.load(os.path.join(opt.rl_pretrain_modeldir, checkpoint))["sentence_model_state"])
        #sentence_bert_model = torch.load(os.path.join(opt.rl_pretrain_modeldir, checkpoint))["sentence_model_state"]
        sequence_bert_model_encoder = BertModel.from_pretrained(opt.save_path)  #先用Pretrained Bert初始化
        sequence_bert_model_encoder.encoder.layer = sequence_bert_model_encoder.encoder.layer[9:]
        sequence_bert_model_encoder = sequence_bert_model_encoder.encoder
        sequence_bert_model_encoder.load_state_dict(torch.load(os.path.join(opt.rl_pretrain_modeldir, checkpoint))["sequence_model_state"])
        #sequence_bert_model_encoder = torch.load(os.path.join(opt.rl_pretrain_modeldir, checkpoint))["sequence_model_state"]
        
        json_filelist = load_data(opt.gen_finetunedata_outputdir, logger_finetune)
        finetune_model(json_filelist, tokenizer, opt, sentence_bert_model, sequence_bert_model_encoder, logger_finetune)
    elif opt.phase == 'infer':
        checkpoint = 'checkpoint10.pth'
        json_filelist = load_data(opt.gen_finetunedata_outputdir, logger_finetune)
        generatejsonfiles_finetune(opt, logger_finetune, json_filelist, opt.gen_finetunedata_outputdir, opt.save_path)
        
        tokenizer = BertTokenizer.from_pretrained(opt.save_path)
        #finetune_output_modeldir
        sentence_bert_model = BertModel.from_pretrained(opt.save_path)  #先用Pretrained Bert初始化
        sentence_bert_model.encoder.layer = sentence_bert_model.encoder.layer[9:]
        sentence_bert_model.load_state_dict(torch.load(os.path.join(opt.finetune_output_modeldir, checkpoint))["sentence_model"])
        sequence_bert_model_encoder = BertModel.from_pretrained(opt.save_path)  #先用Pretrained Bert初始化
        sequence_bert_model_encoder.encoder.layer = sequence_bert_model_encoder.encoder.layer[9:]
        sequence_bert_model_encoder = sequence_bert_model_encoder.encoder
        sequence_bert_model_encoder.load_state_dict(torch.load(os.path.join(opt.finetune_output_modeldir, checkpoint))["sequence_model_encoder"])
        transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True), num_layers=3)
        transformer_encoder.load_state_dict(torch.load(os.path.join(opt.finetune_output_modeldir, checkpoint))["transformer_encoder"])
        transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True), num_layers=3)
        transformer_decoder.load_state_dict(torch.load(os.path.join(opt.finetune_output_modeldir, checkpoint))["transformer_decoder"])
        output_mlp = nn.Linear(768, tokenizer.vocab_size)
        output_mlp.load_state_dict(torch.load(os.path.join(opt.finetune_output_modeldir, checkpoint))["output_mlp"])
        infer_model(json_filelist, tokenizer, opt, sentence_bert_model, sequence_bert_model_encoder, transformer_encoder, transformer_decoder, output_mlp, logger_finetune)
    else:
        raise TypeError("Wrong phase (expected 'bert_pretrain', 'rl_pretrain', 'finetune' or 'infer')")