
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.ERROR)

class bert(nn.Module):
    def __init__(self, config):
        super(bert, self).__init__()
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.fc = nn.Linear(self.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)        
        #out = self.fc(outputs[1])  
        out = self.fc(outputs[0][:,0,:]) 
        out = self.dropout(out) 

        out = F.softmax(out, dim=1)
        return out
