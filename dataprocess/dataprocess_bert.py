########################################################################
# 数据预处理
########################################################################
from torch.utils.data import Dataset
import random
from gensim.models import KeyedVectors
import os
import numpy as np
from nltk.tokenize import word_tokenize
import re


# 数据预处理类,用于把数据处理成dataset形式，然后在主函数入口处使用DataLoader加载
class Dataprocess(Dataset):
    def __init__(self, args, state='train', index=0):
        super(Dataprocess, self).__init__()
        self.data_path = args.data_path        
        self.dataset = args.dataset
        self.seed = args.seed
        with open(args.data_path, "r") as f:
            datas = f.read().splitlines()        

        texts, labels = self.random_all(datas)
        features = self.get_features(args, texts, labels)

        ########################################################################
        # 划分训练集，验证集和测试集, 交叉验证方式
        ########################################################################
        if state=="train": # 生成训练集
            datas_feature = features[:int(index * len(features) / args.k)] + features[int((index + 1) * len(features) / args.k):]
            self.features = datas_feature[0:int((args.k-1)/args.k*len(datas_feature))]

        elif state == "dev": # 生成验证集
            datas_feature = features[:int(index * len(features) / args.k)] + features[int((index + 1) * len(features) / args.k):]
            self.features = datas_feature[int((args.k-1)/args.k * len(datas_feature)):]

        elif state == "test": # 生成测试集
            self.features = features[int(index * len(features) / args.k):int((index + 1) * len(features) / args.k)]
    
    def __getitem__(self, index):   #返回第i个数据样本
        return self.features[index]
    
    def __len__(self):   #返回数据集的大小
        return len(self.features)
    

    

    ########################################################################
    # 随机打乱，然后文本转成了[['increased','from'],['study','three']....]，关键词['increased','','elbow painful joint pain'.......]，标签[1,0,1,0...]
    ########################################################################
    def random_all(self, datas):        
        random.seed(self.seed)
        random.shuffle(datas)

        text, labels = [], [] # 定义三个空列表，分别是文本、关键词和对应的标签
        for line in datas:
            parts = line.split("\t")  # 以tab键（4个空格）分割字符串line，最后有一个\n符号，所以我们以-1截止，不包含最后一个字符\n
            sentence, label = parts[0], parts[1]  # 取出文本和标签分别存入对应的列表中
            words = []  # 再把字符串改成列表存储
            for word in sentence.strip().split():  # strip去掉字符串前后空格
                words.append(word)                     
            text.append(words)            
            labels.append(int(label))
        return text, labels

    

    ########################################################################
    # 处理出需要的数据input_ids，input_mask，label_ids
    ########################################################################
    def get_features(self, args, texts, labels):
        features = []       
        
        for id, sent in enumerate(texts):
            # 此处，句子实际长度为sent+keyword[i],加上truncation=True不会出现警告
            encoded_sent = args.tokenizer.encode_plus(sent, max_length=args.max_seq_len, pad_to_max_length=True, truncation=True)
            input_ids = encoded_sent.get('input_ids')            
            attention_masks = encoded_sent.get('attention_mask')
            
           
            label_ids = int(labels[id])
            text = ' '.join(sent)        

            feature = {
                        'texts': text, 'input_ids': input_ids, 'attention_masks': attention_masks, 'label_ids': label_ids
                    }
            features.append(feature)
        return features

