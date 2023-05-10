########################################################################
# 数据预处理
########################################################################
from torch.utils.data import Dataset
import random
from nltk.tokenize import word_tokenize
import re


# 数据预处理类,用于把数据处理成dataset形式，然后在主函数入口处使用DataLoader加载
class Dataprocess(Dataset):
    def __init__(self, args, state='train', index=0):
        super(Dataprocess, self).__init__()
        self.data_path = args.data_path
        self.stop_path = args.stop_path
        self.meddra_path = args.meddra_path
        self.dataset = args.dataset
        with open(args.data_path, "r") as f:
            datas = f.read().splitlines()        

        keywords = self.get_keywords()   
        
        texts, labels, keywords = self.random_all(args, datas, keywords)

        features = self.get_features(args, texts, labels, keywords)



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
    # 处理出需要的数据input_ids，input_mask，token_type_ids， word_ids，word_mask
    ########################################################################
    def get_features(self, args, texts, labels, keywords):
        features = []        
        
        for id, sent in enumerate(texts):
            # 此处，句子实际长度为sent+keyword[i],加上truncation=True不会出现警告
            encoded_sent = args.tokenizer.encode_plus(sent, keywords[id], max_length=args.max_seq_len, pad_to_max_length=True, truncation=True)
            input_ids = encoded_sent.get('input_ids')
            token_type_ids = encoded_sent.get('token_type_ids')
            attention_masks = encoded_sent.get('attention_mask')
            
            label_ids = int(labels[id])
            text = ' '.join(sent)

            feature = {
                        'texts': text, 'input_ids': input_ids, 'attention_masks': attention_masks, 'token_type_ids': token_type_ids,
                        'label_ids': label_ids
                    }
            features.append(feature)
        return features




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
