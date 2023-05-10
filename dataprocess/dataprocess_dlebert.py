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
        self.stop_path = args.stop_path
        self.meddra_path = args.meddra_path
        self.synonyms_num = args.synonyms_num
        self.dataset = args.dataset
        with open(args.data_path, "r") as f:
            datas = f.read().splitlines()        

        keywords = self.get_keywords()   
        
        texts, labels, keywords = self.random_all(args, datas, keywords)

        if not os.path.exists("./word2vec/word2vec_embedding_"+self.dataset+".npy") and ("./word2vec/word_ids_list_"+self.dataset+".npz"): 
            word_vectors = KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

            word_ids_list = self.similar(texts, word_vectors, keywords)

            corpus = self.vocab_list(word_ids_list)

            self.word_vocab = Vocabulary(corpus, vocab_type='word')     # 产生list的idx2token 以及dict的tiken2idx

            self.word_embedding = self.get_word_embedding(word_vectors, corpus)

            features = self.get_features(args, texts, labels, keywords, word_ids_list)
        else:
            word_ids_list_data = np.load("./word2vec/word_ids_list_"+self.dataset+".npz", allow_pickle=True) # 载入生成的list
            word_ids_list = list(word_ids_list_data['word_ids_list_name'])
            corpus = self.vocab_list(word_ids_list)
            self.word_vocab = Vocabulary(corpus, vocab_type='word')     # 产生list的idx2token 以及dict的tiken2idx
            self.word_embedding = np.load("./word2vec/word2vec_embedding_"+self.dataset+".npy") # 载入生成的词向量
            features = self.get_features(args, texts, labels, keywords, word_ids_list)


        ########################################################################
        # 划分训练集，验证集和测试集，训练集和验证集之和：测试集=7:3，训练集和测试集采用5折交叉验证
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
    # 根据meddra.tsv得到数据集中每个文本对应的关键词
    ########################################################################
    def get_keywords(self):
        # 加载症状词txt文件，有75377个        
        with open(self.meddra_path,"r") as f:
            meddra = f.read().splitlines()    # 此时的content是一个list。里面是很多的75377个str
        
        # 加载停用词，有891个        
        with open(self.stop_path,"r") as f:
            stop = f.read().splitlines()    # 此时的content是一个list。里面是很多的891个str

        # 加载需要过滤的标点符号、数字、数学符号
        remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

        words_list = []  # 定义一个空列表
        for line in meddra:
            line = line.replace("\n", "")
            line = line.split('\t')  # \t就没有了,此时line就是一个包含4个字符串的list
            line = line[3]  #取出第四列元素,此时就是一个str"spontaneous ejaculation"
            line = re.sub(remove_chars, '', line)  # 去掉标点符号、数字、数学符号, 同上是一个大str
            list_str = word_tokenize(line)  # 按空格分词，变成一个list里包含多个str单词
            words = [w for w in list_str if not w in stop]  # 去掉停用词了
            # 去掉单词长度小于3的词
            for word in words:
                if len(word) > 3:
                    word = word.lower()     #把所有的大写字符转成小写字符
                    words_list.append(word)     # 一个包含多个str单词的list
                # 变成一个大list里面包含很多小list，每个小list里面包含很多str单词

        # 去掉words_list里面的重复单词
        result = set(words_list)
        result = list(result)
        
        # 构建字典树
        trie_obj = Trie()
        for word in result:
            trie_obj.insert(word)

        # 加载数据集        
        with open(self.data_path, "r") as f:            
            data = f.read().splitlines()  # 一个list包含4467个str,例如’increased from num int - num_int mg.\t0‘由文本、制表符、标签构成
        
        # 提取出text,用于后续的关键词提取
        texts = []
        for line in data:
            parts = line[:-1].split("\t")  # 以tab键（4个空格）分割字符串line，最后有一个\n符号，所以我们以-1截止，不包含最后一个字符\n
            sentence = parts[0]  # 取出文本和标签分别存入对应的列表中
            texts.append(sentence)

        keywords = []
        for text in texts:
            line = re.sub(remove_chars, '', text)  # 去掉标点符号、数字、数学符号
            list_words = word_tokenize(line)  # 分词，变成一个list里包含多个str单词
            filtered_words = [w for w in list_words if not w in stop]  # 文本中去掉停用词1了
            keyword = []


            for word in filtered_words:
                if len(word) > 3:  # 少于3的词不用进行匹配
                    search = trie_obj.search(word)   # 精准匹配
                    if search == True:
                        keyword.append(word)
            lines = ' '.join(keyword)
            keywords.append(lines)

        return(keywords)


    ########################################################################
    # 随机打乱，然后文本转成了[['increased','from'],['study','three']....]，关键词['increased','','elbow painful joint pain'.......]，标签[1,0,1,0...]
    ########################################################################
    def random_all(self, args, datas, keywords):
        data_all = []
        for i, data in enumerate(datas):
            parts = data.split("\t")  # 以tab键（4个空格）分割字符串line
            sentence, label = parts[0], parts[1]
            text = sentence+'\t'+keywords[i]+'\t'+label
            data_all.append(text)
        random.seed(args.seed)
        random.shuffle(data_all)

        text, key_values, labels = [], [], []  # 定义三个空列表，分别是文本、关键词和对应的标签
        for line in data_all:
            parts = line.split("\t")  # 以tab键（4个空格）分割字符串line，最后有一个\n符号，所以我们以-1截止，不包含最后一个字符\n
            sentence, keyword, label = parts[0], parts[1], parts[2]  # 取出文本和标签分别存入对应的列表中
            words = []  # 再把字符串改成列表存储
            for word in sentence.strip().split():  # strip去掉字符串前后空格
                words.append(word)                     
            text.append(words)
            key_values.append(keyword)
            labels.append(int(label))
        return text, labels, key_values

    ########################################################################
    # 构建匹配词语表示word_ids_list处理为[[['decreased','increasing','increase'],[]....],[[],[]....]]....]  没有关键词的就用空list表示
    ########################################################################
    def similar(self, texts, word_vectors, keywords):        
        word_ids_list = []
        for i, line in enumerate(texts):   # line= list['incresed', 'from', 'int']
            g = []
            for word in line:          # word = str  'incresed'
                if word in keywords[i] and word_vectors.index_to_key:             
                    try:
                        m = word_vectors.most_similar(positive = [word], topn = 5)     # [(),(),()]
                        datas2 = []
                        for k in m:
                            n = k[0]   # 取元祖的第一个元素
                            datas2.append(n)
                        g.append(datas2)
                    except:
                        g.append([])
                else:
                    g.append([])
            word_ids_list.append(g)
        np.savez("./word2vec/word_ids_list_"+self.dataset+".npz", word_ids_list_name=word_ids_list)    # word_ids_list_name的名字是固定的   
        return word_ids_list
 

    ########################################################################
    # 构建词库，不重复的相似词集合，例如['incresed','increase','decreased']
    ########################################################################
    def vocab_list(self, word_ids_list):
        corpus = []
        for i in word_ids_list:
            for j in i:
                for k in j:
                    corpus.append(k)
        corpus = set(corpus)
        corpus = list(corpus) 
        return corpus

    ########################################################################
    # 形成word_embedding，数组，维度为[word_vocab.size, embed_dim]
    ########################################################################
    def get_word_embedding(self, word_vectors, corpus):
                  
        embed_dim = 300   # 词向量维度
        word_embedding = np.empty([self.word_vocab.size, embed_dim])
        scale = np.sqrt(3.0 / embed_dim)    # 开平方

        for idx, word in enumerate(self.word_vocab.idx2token):
            if word in word_vectors.index_to_key:
                word_embedding[idx, :] = word_vectors.get_vector(word)
            else:
                word_embedding[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
        np.save("./word2vec/word2vec_embedding_"+self.dataset+".npy", word_embedding) # 保存生成的词向量        
        return word_embedding

    ########################################################################
    # 处理出需要的数据input_ids，input_mask，token_type_ids， word_ids，word_mask
    ########################################################################
    def get_features(self, args, texts, labels, keywords, word_ids_list):
        features = []        
        word_pad_id = self.word_vocab.convert_token_to_id('[PAD]')    #0
        for id, sent in enumerate(texts):
            # 此处，句子实际长度为sent+keyword[i],加上truncation=True不会出现警告
            encoded_sent = args.tokenizer.encode_plus(sent, keywords[id], max_length=args.max_seq_len, pad_to_max_length=True, truncation=True)
            input_ids = encoded_sent.get('input_ids')
            token_type_ids = encoded_sent.get('token_type_ids')
            attention_masks = encoded_sent.get('attention_mask')
            word_ids = []
            for j in word_ids_list[id]:        # j=['increased','incres','decrease']
                word_i = self.word_vocab.convert_tokens_to_ids(j)    # 把对应的单词转成数字  ['increased','incres','decrease']转化为[1,354,45]
                word_pad_num = 5 - len(j)   # 把不足三个单词的位置数量计算出来
                word_i = word_i + [word_pad_id] * word_pad_num   # [word_pad_id]先转成了list类型  ,会转成一个list。包含三个int [2134,0,0]
                word_ids.append(word_i)

            word_ids = [[word_pad_id]*5] + word_ids + [[word_pad_id]*5]     # 开头和结尾进行补0进行padding   ,因为进入bert会有cls和sep符号
            if len(word_ids) > args.max_seq_len:                
                word_ids = word_ids[: args.max_seq_len]       # 长度为实际15
            
            # 进行padding补齐操作
            padding_length2 = args.max_seq_len - len(word_ids)
            word_ids += [[word_pad_id]*5] * padding_length2     # 变为70

            word_masks = []            
            for i in word_ids:
                j_result = []
                for j in i:
                    if j == 0:
                        i_rt = False
                    else:
                        i_rt = True
                    j_result.append(i_rt)
                word_masks.append(j_result)
            label_ids = int(labels[id])
            text = ' '.join(sent)
            

            feature = {
                        'texts': text, 'input_ids': input_ids, 'attention_masks': attention_masks, 'token_type_ids': token_type_ids,
                        'word_ids': word_ids, 'word_masks': word_masks, 'label_ids': label_ids
                    }
            features.append(feature)
        return features


class Vocabulary(object):
    """
    构建词表
    """
    def __init__(self, tokens, vocab_type=''):
        """
        :param tokens:
        :param vocab_type:
        """
        assert vocab_type in ['label', 'word', '']
        self.token2idx = {}
        self.idx2token = []
        self.size = 0
        if vocab_type == 'word':
            tokens = ['[PAD]', '[UNK]'] + tokens
        elif vocab_type == 'label':
            tokens = ['[PAD]'] + tokens
        # self.tokens = tokens

        for token in tokens:
            self.token2idx[token] = self.size
            self.idx2token.append(token)
            self.size += 1

    def get_size(self):
        return self.size

    def convert_token_to_id(self, token):
        if token in self.token2idx:
            return self.token2idx[token]
        else:
            return self.token2idx['[UNK]']

    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, idx):
        return self.idx2token[idx]

    def convert_ids_to_tokens(self, ids):
        return [self.convert_id_to_token(ids) for ids in ids]


# 字典树的实现，可以进行精确匹配和模糊匹配
class TrieNode:
    def __init__(self):
        self.children = dict()   
        self.isEnd = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.isEnd = True

    def search(self, prefix):  # 通过判断结尾的字符是不是end,这个函数时精确匹配
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.isEnd

    def startsWith(self, prefix):  # 通过判断结尾的字符是不是end,这个函数是前缀匹配，只要是
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
