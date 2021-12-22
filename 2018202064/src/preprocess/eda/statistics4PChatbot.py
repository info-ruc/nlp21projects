'''
Author: your name
Date: 2021-11-13 02:53:46
LastEditTime: 2021-11-25 02:49:30
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/statistics4PChatbot.py
'''
#* 统计用户dialog长度的分布特征
path = '/home/zhengyi_ma/pcb/Data/PChatbot_byuser_filter'
#path = '/home/zhaoheng_huang/SIGIR2022/data'
import os
from tqdm import tqdm
import pickle
from pprint import pprint
from collections import defaultdict
#dict = defaultdict(lambda: 0)
dict = {}
for file in tqdm(os.listdir(path)):
    abs_path = os.path.join(path, file)
    count = 0
    with open(abs_path, 'r') as fp:
        for line in fp:
            count += 1
    if dict.get(count) == None:
        dict[count] = 1
    else:
        dict[count] += 1
pprint(dict.items())
with open("./statistics.pkl", 'wb') as f:
    pickle.dump(dict, f)