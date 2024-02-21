import string
from itertools import zip_longest
from wobert import WoBertTokenizer
import jieba
import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer, AutoModel


class MyDataset(Dataset):
    def __init__(self, path):
        self.data = self.__load_data__(path)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __load_data__(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            label, text = int(line[1]), line[0]
            data.append({'label': label, 'text': text})
        return data

    def process(self):
        pass


class MyCollate():
    def __init__(self, config):
        self.config = config
        if self.config.model == 'CNN' or self.config.model == 'BiLSTM' or self.config.model == 'CNN_BiLSTM' or self.config.model == 'Mvatt_LSTM' or self.config.model == 'CAT_BiGRU' or self.config.model == 'Transformer_BiLSTM':
            self.token = KeyedVectors.load_word2vec_format(config.pretrained, binary=False)
        elif self.config.model == 'BERT' :
            self.token = AutoTokenizer.from_pretrained(config.pretrained)
        elif self.config.model == 'RoBERTa':
            self.token = BertTokenizer.from_pretrained(config.pretrained)
        elif self.config.model[:5] == 'IDIHR':
            self.rtoken = BertTokenizer.from_pretrained(config.pretrained)
            self.wtoken = WoBertTokenizer.from_pretrained('junnyu/wobert_chinese_plus_base')

    def get_token(self, model, texts, max_length):
        if model == 'CNN' or model == 'BiLSTM' or model == 'CNN_BiLSTM' or model == 'Mvatt_LSTM' or model == 'CAT_BiGRU' or model == 'Transformer_BiLSTM':
            tf = []
            for text in texts:
                text = [word for word in jieba.lcut(text) if word in self.token.key_to_index]
                tf.append(self.pad_list(self.token[text], max_length, np.zeros(300)))
            return torch.FloatTensor(tf)
        elif model == 'BERT' or model == 'RoBERTa':
            return self.token.batch_encode_plus(batch_text_or_text_pairs=texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        elif model[:5] == 'IDIHR':
            tf = {'roberta': [], 'wobert': [], 'mask': []}
            tf['roberta'] = self.rtoken.batch_encode_plus(batch_text_or_text_pairs=texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            tf['wobert'] = self.wtoken.batch_encode_plus(batch_text_or_text_pairs=texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            tf['mask'] = self.get_mask(texts, max_length)
        return tf


    def pad_list(self, lst, target_length, fillvalue):
        padded_list = list(zip_longest(lst, range(target_length), fillvalue=fillvalue))[:target_length]
        return [item[0] for item in padded_list]

    def collate_fn(self, data):
        labels = [i['label'] for i in data]
        texts = [i['text'] for i in data]
        labels = torch.LongTensor(labels)
        texts = self.get_token(self.config.model, texts, self.config.max_length)
        return labels, texts

    def get_mask(self, texts, max_length):
        ll=torch.empty(self.config.batch_size,max_length)
        for i,text in enumerate(texts):
            ll[i]=self.one_hot_encoding(text, max_length)
        return torch.Tensor(ll)
    def one_hot_encoding(self,sentence, max_length):
        one_hot = torch.zeros(max_length)
        pun = string.punctuation + '，。！？【】（）《》“”‘’；：——……、·'
        for i, char in enumerate(sentence):
            if char in pun:
                one_hot[i] = 1
            if i == max_length - 1:
                break
        return one_hot

"""
class config():
    def __init__(self):
        self.batch_size = 16
        self.max_length = 100
        self.model = 'IDIHR'
        self.pretrained = 'pretrain/roberta'


if __name__ == '__main__':
    data = MyDataset('datasets/new/train.tsv')
    roberta = AutoModel.from_pretrained('pretrain/roberta')
    co = MyCollate(config())
    out = co.collate_fn(data[:16])
    print(out[1])
    print(roberta.config)"""
# 保存结果到文件
