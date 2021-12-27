'''
Author: your name
Date: 2021-11-14 07:53:16
LastEditTime: 2021-11-22 02:42:29
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /SIGIR2022/seq2seq/utils/utils.py
'''
import torch
def cosine_similarity(vector1, vector2):
    """Compute cosine similarity of two vectors which have the same dimension N

    Args:
        vector1 (list(int(N))): the first vector
        vector2 (list(int(N))): the second vector

    Returns:
        int: the cosine value (-1, 1)
    """
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)
    #    return torch.round(dot_product / ((normA ** 0.5) * (normB ** 0.5)) * 100)

