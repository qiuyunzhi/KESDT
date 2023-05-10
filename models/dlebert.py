
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertLayer, BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING, _TOKENIZER_FOR_DOC, _CONFIG_FOR_DOC
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings_to_callable


class dlebert(BertPreTrainedModel):
    def __init__(self, config):
        super(dlebert, self).__init__(config)
        self.word_embeddings = nn.Embedding(config.word_vocab_size, config.word_embed_dim)
        self.bert = LEBertModel(config)        
        self.fc = nn.Linear(config.hidden_size, config.num_labels)  
        self.dropout = nn.Dropout(config.dropout)       
        self.init_weights()     # 将所有子模型的linear参数赋值为1

    def forward(self, input_ids, attention_mask, token_type_ids, word_ids, word_mask):
        word_embeddings = self.word_embeddings(word_ids)
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            word_embeddings=word_embeddings, word_mask=word_mask
        )      
        #out = self.fc(outputs[1])
        out = self.fc(outputs[0][:,0,:])  
        out = self.dropout(out)
                
        out = F.softmax(out, dim=1)
        return out  


# LEBertModel类调用BertEncoder类， BertEncoder类调用WordEmbeddingAdapter类
class LEBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)      # BertEncoder 改了
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        word_embeddings=None,   #新加的
        word_mask=None,    # 新加的
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            word_embeddings=word_embeddings,    # 新加的
            word_mask=word_mask,   #新加的
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.word_embedding_adapter = WordEmbeddingAdapter(config)     # 新加的，传入之前处理过的WordEmbeddingAdapter内容

    def forward(
        self,
        hidden_states,
        word_embeddings,     # 新加的
        word_mask,   # 新加的
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # 在第i层之后，进行融合  新加的  config.add_layer表示第几层transformer后面融入词性息
            if i == self.config.add_layer:
                hidden_states = self.word_embedding_adapter(hidden_states, word_embeddings, word_mask)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class WordEmbeddingAdapter(nn.Module):
    def __init__(self, config):
        super(WordEmbeddingAdapter, self).__init__()
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

        self.linear1 = nn.Linear(config.word_embed_dim, config.hidden_size)   # (200, 768)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)   # (768,768)
       
        
        #self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)   #按照正太分布初始化att_W
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)   #归一化  (768,) 就是只有一行向量，共768个值
        self.wq = nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))  # (768,768)
        self.wk = nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))  # (768,768)
        self.wv = nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))  # (768,768)


    def forward(self, layer_output, word_embeddings, word_mask):
        """
        :param layer_output: bert layer的输出,[b_size, len_input, d_model]=[4,150,768]
        :param word_embeddings:每个汉字对应的词向量集合,[b_size, len_input, num_word, d_word]= [4,150,3,200]
        :param word_mask:每个汉字对应的词向量集合的attention mask, [b_size, len_input, num_word]=[4,150,3],,,其值为true或False 表示该字符有没有匹配到词组

        该类的作用在于计算bert传入的字符向量与本文定义的词向量进行融合
        """

        # transform
        # 将词向量，与字符向量进行维度对齐

        word_outputs = self.linear1(word_embeddings)  # 将词向量200变为768维
        #word_outputs = self.tanh(word_outputs)  #
        #word_outputs = self.linear2(word_outputs)  # 
        word_outputs = self.dropout(word_outputs)   # word_outputs：[b_size, len_input, num_word, d_model]

        # 计算每个字符向量，与其对应的所有词向量的注意力权重，点积的计算方式
        Q = torch.matmul(layer_output, self.wq)   # [b_size, len_input, d_model]
        K = torch.matmul(word_outputs, self.wk)   #[b_size, len_input, num_word, d_model]
        V = torch.matmul(word_outputs, self.wv) 
        QK = torch.matmul(Q.unsqueeze(2), torch.transpose(K, 2, 3))  # [b_size, len_input, 1, num_word]
        socres = QK.squeeze(2)
        socres.masked_fill_(word_mask, -1e9)  # 将pad的注意力设为很小的数  ,某些行定义为1或0
        socres = F.softmax(socres, dim=-1)  # [b_size, len_input, num_word]
        socres = socres.unsqueeze(-1)  # [b_size, len_input, num_word, 1]
        socres = V * socres             # [b_size, len_input, num_word, d_model]

        #加权求和和特征融合
        weighted_word_embedding = torch.sum(socres, dim=2)  # 加权求和，得到每个汉字对应的词向量集合的表示
        layer_output = layer_output + weighted_word_embedding  # [4,150,768]

        #
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output)

        return layer_output


# 原始的词嵌入方式
'''
class WordEmbeddingAdapter(nn.Module):
    def __init__(self, config):
        super(WordEmbeddingAdapter, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tanh = nn.Tanh()

        self.linear1 = nn.Linear(config.word_embed_dim, config.hidden_size)   # (200, 768)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)   # (768,768)
       
        self.attn_W = nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))  # (768,768)
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)   #按照正太分布初始化att_W
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)   #归一化  (768,) 就是只有一行向量，共768个值

    def forward(self, layer_output, word_embeddings, word_mask):
        """
        :param layer_output: bert layer的输出,[b_size, len_input, d_model]=[4,150,768]
        :param word_embeddings:每个汉字对应的词向量集合,[b_size, len_input, num_word, d_word]= [4,150,3,200]
        :param word_mask:每个汉字对应的词向量集合的attention mask, [b_size, len_input, num_word]=[4,150,3],,,其值为true或False 表示该字符有没有匹配到词组

        该类的作用在于计算bert传入的字符向量与本文定义的词向量进行融合
        """

        # transform
        # 将词向量，与字符向量进行维度对齐

        word_outputs = self.linear1(word_embeddings)  # 将词向量200变为768维[4,150,3,768]
        #word_outputs = self.tanh(word_outputs)  #
        #word_outputs = self.linear2(word_outputs)  # [4,150,3,768]
        word_outputs = self.dropout(word_outputs)   # word_outputs：[b_size, len_input, num_word, d_model]

        # 计算每个字符向量，与其对应的所有词向量的注意力权重，然后加权求和。采用双线性映射计算注意力权重
        socres = torch.matmul(layer_output.unsqueeze(2), self.attn_W)  # [b_size, len_input, 1, d_model] = [64,46,1,768]   matmul矩阵乘法函数   unsqueeze 升维
        socres = torch.matmul(socres, torch.transpose(word_outputs, 2, 3))  # [b_size, len_input, 1, num_word]= [4,150,1,3]
        socres = socres.squeeze(2)  # [b_size, len_input, num_word] = [4,150,3]
        socres.masked_fill_(word_mask, -1e9)  # 将pad的注意力设为很小的数  ,某些行定义为1或0，构成双线性注意力权重
        socres = F.softmax(socres, dim=-1)  # [b_size, len_input, num_word]=[4,150,3]
        attn = socres.unsqueeze(-1)  # [b_size, len_input, num_word, 1]=[4,150,3，1]

        #加权求和和特征融合
        weighted_word_embedding = torch.sum(word_outputs * attn, dim=2)  # [N, L, D]=[4,150,768]   # 加权求和，得到每个汉字对应的词向量集合的表示
        layer_output = layer_output + weighted_word_embedding  # [4,150,768]

        #
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output)

        return layer_output
'''