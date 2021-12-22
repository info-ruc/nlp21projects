###
 # @Author: your name
 # @Date: 2021-10-18 06:19:38
 # @LastEditTime: 2021-12-22 10:29:05
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /zhaoheng_huang/SIGIR2022/scripts/runModel.sh
###
#! Working directory: SIGIR2022/ 
#! To run model, execute following commands:

#! conda activate py38
#! bash ./scripts/runModel.sh

DEVICES='0'     #GPU卡编号
#DEVICES=None

#! 非常小的一个数据集，调试时用
SMALL_DATA_DIR='/home/zhaoheng_huang/SIGIR2022/data'                                         # 19 Users
#! PChatbot数据集（特别大），评测时用
BIG_DATA_DIR='/home/zhengyi_ma/pcb/Data/PChatbot_byuser_filter' # 1301274 Users

#! 输出结果
RESULT_FILE='/home/zhaoheng_huang/results/res_runModel.txt'

#! BERT模型的目录，其他不需修改的设置
BERT_DIR='/home/zhaoheng_huang/data/bert-base-chinese'

# bert_pretrain, rl_pretrain, finetune, infer
PHASE='infer'

USER_LIMIT=10000
PRETRAIN_BS=3
PRETRAIN_EPOCHS=3
RL_PRETRAIN_BS=3
RL_PRETRAIN_EPOCHS=5
FINETUNE_EPOCHS=10
FINETUNE_BS=20
# -O取消debug模式
python      \
    runModel.py  \
    --device $DEVICES \
    --random_seed 2022 \
    --data_dirpath $BIG_DATA_DIR \
    --result_filepath $RESULT_FILE \
    --user_limit $USER_LIMIT \
    --bert_pretrain_batch_size $PRETRAIN_BS \
    --bert_dir $BERT_DIR \
    --rl_pretrain_batch_size $RL_PRETRAIN_BS \
    --rl_pretrain_epochs $RL_PRETRAIN_EPOCHS \
    --finetune_epochs $FINETUNE_EPOCHS \
    --finetune_batch_size $FINETUNE_BS \
    --phase $PHASE
#    > res.log 2>&1 &        #* 放到后台
#echo $!             #* 打印进程号
#tail -f res.log