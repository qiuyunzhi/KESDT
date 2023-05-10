########################################################################
# 主函数，包含各种神经网络模型
########################################################################
import torch
import numpy as np
import torch.optim as optim
import random
import argparse
import os
from transformers import BertTokenizer, BertConfig
import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed):
    random.seed(seed)  # random的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)   # 必须加上，为了禁止hash随机化
    np.random.seed(seed)  # numpy的随机种子
    torch.manual_seed(seed)  # 为CPU运算设置随机种子
    torch.cuda.manual_seed(seed)   # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU计算设置随机种子
    torch.backends.cudnn.benchmark = False   # 可以设置，或不设置，会牺牲计算效率换取
    torch.backends.cudnn.deterministic = True  # 固定网络


def set_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--dataset", type=str, default="cadec")  
    parser.add_argument("--model", type=str, default='dlebert')    
    parser.add_argument('--device', type=str, default="cuda:3", help='e.g. cuda:0')

    parser.add_argument("--meddra_path", type=str, default='./datasets/meddra.tsv', help='数据路径')
    parser.add_argument("--stop_path", type=str, default='./datasets/stop_words.txt', help='数据路径')
    parser.add_argument('--bert_path', type=str, default='./pretrain_model/bert-base-uncased/', help="数据路径")

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--k", type=int, default=5, help="交叉验证次数")
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help='Bert的学习率')

    parser.add_argument('--num_labels', default=2, type=int, help="分成几类")
    parser.add_argument('--synonyms_num', default=3, type=int, help="表示同义词的个数")
    parser.add_argument('--add_layer', default=1, type=int, help="表示在bert的第几层嵌入外部知识")

    args = parser.parse_args()
    return args

if __name__ == '__main__':    
    args = set_args()
    setup_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)  
    
    # 每个数据集处理的句子长度是不一样的，因此用下面的选择语句
    if args.dataset == "cadec":
        args.max_seq_len = 70
        args.data_path = './datasets/cadec.txt'
        #args.learning_rate = 5e-5
    elif args.dataset == "twitter":
        args.max_seq_len = 46
        args.data_path = './datasets/twitter.txt'
        #args.learning_rate = 5e-5
    elif args.dataset == "twimed":
        args.max_seq_len = 65
        args.data_path = './datasets/twimed.txt'
        #args.learning_rate = 5e-5
    else:
        pass
    #分词器
    args.tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
      

    accuracy, precision, recall, f1 = 0, 0, 0, 0
    for i in range(0, args.k):  # i从0开始到config.k-1结束
        print("第%d次交叉验证开始了" %(i+1))       
        
        # 根据模型的类型导入不同的函数
        if args.model == "bert":   
            from dataprocess.dataprocess_bert import Dataprocess           
            from trainer.trainer_bert import train, evaluate, dataLoader        
            from models.bert import bert
            train_loader = dataLoader(args, state="train", index=i)
            dev_loader = dataLoader(args, state="dev", index=i)
            test_loader = dataLoader(args, state="test", index=i)  
            args.num_labels = 2 
            model = bert(args).to(args.device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        elif args.model == "kbert":
            from dataprocess.dataprocess_kbert import Dataprocess                
            from trainer.trainer_kbert import train, evaluate, dataLoader        
            from models.kbert import kbert
            train_loader = dataLoader(args, state="train", index=i)
            dev_loader = dataLoader(args, state="dev", index=i)
            test_loader = dataLoader(args, state="test", index=i) 
            args.num_labels = 2 
            model = kbert(args).to(args.device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        elif args.model == "sbert": 
            from dataprocess.dataprocess_sbert import Dataprocess             
            from trainer.trainer_sbert import train, evaluate, dataLoader        
            from models.sbert import sbert
            train_loader = dataLoader(args, state="train", index=i)
            dev_loader = dataLoader(args, state="dev", index=i)
            test_loader = dataLoader(args, state="test", index=i) 
            # 初始化模型配置（BertConfig类）
            config = BertConfig.from_pretrained(args.bert_path)
            config.num_labels = args.num_labels   
            config.add_layer = args.add_layer       # 在bert的第几层后面融入词汇信息
            
            # 需要Dataprocess类传递一些参数过来，词向量维度，以及生成的词向量
            dataset = Dataprocess(args)
            word_embedding = dataset.word_embedding
            config.word_vocab_size = word_embedding.shape[0]                       # 词表大小，根据我生成的词向量获得       
            config.word_embed_dim = word_embedding.shape[1]  
            config.dropout = args.dropout
            # 初始化模型
            model = sbert.from_pretrained(args.bert_path, config=config).to(args.device) 
            # 初始化模型的词向量
            model.word_embeddings.weight.data.copy_(torch.from_numpy(word_embedding))
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        elif args.model == "dlebert":    
            from dataprocess.dataprocess_dlebert import Dataprocess            
            from trainer.trainer_dlebert import train, evaluate, dataLoader        
            from models.dlebert import dlebert

            train_loader = dataLoader(args, state="train", index=i)
            dev_loader = dataLoader(args, state="dev", index=i)
            test_loader = dataLoader(args, state="test", index=i) 

            # 初始化模型配置（BertConfig类）
            config = BertConfig.from_pretrained(args.bert_path)
            config.num_labels = args.num_labels   
            config.add_layer = args.add_layer       # 在bert的第几层后面融入词汇信息
            
            # 需要Dataprocess类传递一些参数过来，词向量维度，以及生成的词向量
            dataset = Dataprocess(args)
            word_embedding = dataset.word_embedding
            config.word_vocab_size = word_embedding.shape[0]                       # 词表大小，根据我生成的词向量获得       
            config.word_embed_dim = word_embedding.shape[1]  
            config.dropout = args.dropout
            # 初始化模型
            model = dlebert.from_pretrained(args.bert_path, config=config).to(args.device) 
            # 初始化模型的词向量
            model.word_embeddings.weight.data.copy_(torch.from_numpy(word_embedding))
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        else:
            pass  



        train(args, model, train_loader, dev_loader, optimizer)
        #model.load_state_dict(torch.load("./best_bert_model.pth"))
        test_acc, test_pre, test_rec, test_f1 = evaluate(args, model, test_loader)
        print("test_acc: {:.4f}, test_pre: {:.4f}, test_rec: {:.4f}, test_f1: {:.4f}".format(test_acc, test_pre, test_rec, test_f1))
        accuracy += test_acc / args.k
        precision += test_pre / args.k
        recall += test_rec / args.k
        f1 += test_f1 / args.k
    print(args.dataset+"_"+args.model+"交叉验证结果：accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}".format(accuracy, precision, recall, f1))