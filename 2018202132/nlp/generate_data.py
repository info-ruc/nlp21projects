import spacy
import re
import os


nlp = spacy.load("en_core_web_sm")
filenames = os.listdir(r"D:/自然语言处理/Arxiv6K")


def isNoise(token):
    is_noise = False
    if token.pos_ in noisy_pos_tags:
        is_noise = True
    elif token.is_stop:
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    return is_noise


def cleanup(token, lower=True):
    if lower:
        token = token.lower()
    return token.strip()


#with open("label_entity.txt", "w") as f:
#   for filename in filenames:
#        path = "./Arxiv6K/" + filename + "/main.tex"
#        document = open(path).read()

        # 数据处理
#        document = re.sub(r'\{[^\}]+\}', "", document)  # 删除{}的所有内容
#        document = re.sub(r'\[[^\}]+\]', "", document)  # 删除[]的所有内容
#        document = re.sub(r'\\[^\s\n]+[\s\n]', "", document)  # 删除所有以/开头的标志
#        document = re.sub(r'[\$\%\&]', "", document)  # 删除所有$、%、&字符
#        document = nlp(document)

        # 得到所有的词性标注
#        all_tags = {w.pos: w.pos_ for w in document}

        # 去除噪声词
#        noisy_pos_tags = ["PROP"]
#        min_token_length = 2

        # 实体检测
#        labels = set([w.label_ for w in document.ents])
#        for label in labels:
#            entities = [cleanup(e.string, lower=False) for e in document.ents if label==e.label_]
#            entities = list(set(entities))
#            f.writelines(label + str(entities))


path = "./document.txt"
document = open(path, errors='ignore').read()
with open("label_entity.txt", "w") as f:
    document = nlp(document)

    # 得到所有的词性标注
    all_tags = {w.pos: w.pos_ for w in document}

    # 去除噪声词
    noisy_pos_tags = ["PROP"]
    min_token_length = 2

    # 实体检测
    labels = set([w.label_ for w in document.ents])
    for label in labels:
        entities = [cleanup(e.string, lower=False) for e in document.ents if label==e.label_]
        entities = list(set(entities))
        f.write(label + str(entities))


