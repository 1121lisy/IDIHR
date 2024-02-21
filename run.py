import argparse
import json
import os
import time
import warnings
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from myDataset import MyDataset, MyCollate
from util import init_network, set_seed, test, eval

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Hyperbolic Representation')
parser.add_argument('--model', default='CNN', type=str, help='')
parser.add_argument('--dataset', default='new', type=str)
parser.add_argument('--max_length', default=100, type=int, help='max seq length')
parser.add_argument('--num_labels', default=2, type=int, help='num labels')
parser.add_argument('--pretrained', default='pretrain/word2vec/merge_sgns_bigram_char300.txt', type=str,
                    help='pretrained model path')
parser.add_argument('--hidden_channels', default=256, type=int, help='hidden channels')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout')
parser.add_argument('--init', default='random', type=str, help='xavier,kaiming,normal,random init weight')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--patience', default=10, type=int, help='patience')
parser.add_argument('--seed', default=114514, type=int, help='seed')
parser.add_argument('--device', default='None', type=str, help='device')
args = parser.parse_args()


class Config():
    def __init__(self):
        self.bert_h = 768
        self.hidden_size = args.hidden_channels
        self.model = args.model
        self.num_labels = args.num_labels
        self.pretrained = args.pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dropout = args.dropout
        self.EPOCH = args.epochs
        self.max_length = args.max_length
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.seed = args.seed
        self.init = args.init
        self.dataset = args.dataset
        self.patience = args.patience
        self.batch_size = args.batch_size

    def log(self):
        return {'model': self.model,
                'bert_h': self.bert_h,
                'hidden_size': self.hidden_size,
                'num_labels': self.num_labels,
                'pretrained': self.pretrained,
                'device': self.device,
                'dropout': self.dropout,
                'max_length': self.max_length,
                'EPOCH': self.EPOCH,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'init': self.init,
                'seed': self.seed,
                'dataset': self.dataset,
                'patience': self.patience,
                'batch_size': self.batch_size
                }


config = Config()


def get_model(model):
    if model == 'CNN':
        from model.CNN import TextCNN
        return TextCNN(embed_dim=300, num_classes=config.num_labels)
    elif model == 'BiLSTM':
        from model.BiLSTM import BiLSTM
        return BiLSTM(embedding_dim=300, hidden_dim=config.hidden_size, dropout=config.dropout,
                      output_dim=config.num_labels)
    elif model == 'CNN_BiLSTM':
        from model.CNN_BiLSTM import CNN_BiLSTM
        return CNN_BiLSTM(embed_dim=300, num_classes=config.num_labels, hidden_dim=config.hidden_size)
    elif model == 'Mvatt_LSTM':
        from model.Mvatt_LSTM import BiLSTM_MultiheadAttention
        return BiLSTM_MultiheadAttention(embed_dim=300, num_classes=config.num_labels, hidden_dim=config.hidden_size)
    elif model == 'CAT_BiGRU':
        from model.CAT_BiGRU import BiGRU_Attention
        return BiGRU_Attention(embed_dim=300, num_classes=config.num_labels, hidden_dim=config.hidden_size)
    elif model == 'Transformer_BiLSTM':
        from model.Transformer_BiLSTM import Transformer_BiLSTM
        return Transformer_BiLSTM(embed_dim=300, num_classes=config.num_labels, hidden_dim=config.hidden_size)
    elif model == 'BERT':
        config.pretrained = 'pretrain/bert'
        from model.BERT import BertClassifier
        return BertClassifier(bert_model_name=config.pretrained, num_labels=config.num_labels, dropout=config.dropout)
    elif model == 'RoBERTa':
        config.pretrained = 'pretrain/roberta'
        from model.RoBERTa import RoBertaClassifier
        return RoBertaClassifier(bert_model_name=config.pretrained, num_labels=config.num_labels,
                                 dropout=config.dropout)
    elif model[:5] == 'IDIHR':
        config.pretrained = 'pretrain/roberta'
        if model[-2:] == 'sp':
            from model.IDIHR_sp import IDIHR_sp
            return IDIHR_sp(bert_model_name=config.pretrained,
                            num_labels=config.num_labels,
                            dropout=config.dropout)
        elif model[-2:] == 'ra':
            from model.IDIHR_ra import IDIHR_ra
            return IDIHR_ra( max_len=config.max_length,
                            num_labels=config.num_labels,
                            dropout=config.dropout)
        elif model[-2:] == 'hr':
            from model.IDIHR_hr import IDIHR_hr
            return IDIHR_hr(bert_model_name=config.pretrained,
                            max_len=config.max_length,
                            num_labels=config.num_labels,
                            dropout=config.dropout)
        elif model[-2:] == 'at':
            from model.IDIHR_at import IDIHR_at
            return IDIHR_at(bert_model_name=config.pretrained,
                            max_len=config.max_length,
                            num_labels=config.num_labels,
                            dropout=config.dropout)
        else:
            from model.IDIHR import IDIHR
            return IDIHR(bert_model_name=config.pretrained, max_len=config.max_length, num_labels=config.num_labels,
                     dropout=config.dropout)


if __name__ == '__main__':

    start_time = time.time()  # 起始时间

    set_seed(config.seed)  # 固定随机种子

    save_dir = 'log/' + config.dataset + '/' + config.model

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, 'best.pth')
    model = get_model(config.model)
    # 数据集加载
    train_dataset = MyDataset('datasets/' + config.dataset + '/train.tsv')
    dev_dataset = MyDataset('datasets/' + config.dataset + '/dev.tsv')
    test_dataset = MyDataset('datasets/' + config.dataset + '/test.tsv')
    collate = MyCollate(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate.collate_fn,
                                  drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False,
                                collate_fn=collate.collate_fn,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                 collate_fn=collate.collate_fn,
                                 drop_last=True)

    print(config.device)

    # 模型加载
    model = get_model(config.model)
    model.to(config.device)

    # 权重初始化
    init_network(model=model, method=config.init)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion.to(config.device)
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best = 0
    count = 0
    end_epoch = config.EPOCH
    for epoch in range(config.EPOCH):
        eval(Epoch=epoch, EPOCHS=config.EPOCH, model=model, test=False, data_loader=train_dataloader,
             device=config.device,
             criterion=criterion, optimizer=optimizer)

        acc, f1, r, p, dev_ls = eval(Epoch=epoch, EPOCHS=config.EPOCH, model=model, data_loader=dev_dataloader,
                                     device=config.device, criterion=criterion, optimizer=optimizer, test=True)
        tqdm.write('dev acc:{},f1:{},r:{},p:{},loss:{}'.format(acc, f1, r, p, dev_ls))
        if acc > best:
            count = 0
            best = acc
            torch.save(model.state_dict(), model_save_path)
        elif count < config.patience:
            count += 1
        else:
            end_epoch = epoch + 1
            break
    model.load_state_dict(torch.load(model_save_path))
    acc, f1, r, p = test(model=model, data_loader=dev_dataloader, device=config.device)
    print('test acc:{},f1:{},r:{},p:{}'.format(acc, f1, r, p))

    end_time = time.time()
    run_time = round(end_time - start_time)
    # 计算时分秒
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    # 输出
    print('运行时间：{}时{}分{}秒'.format(hour, minute, second))
    with open(save_dir + '/{}.json'.format(end_time), 'w', encoding='utf-8') as f:
        json.dump({'performance': {'acc': acc, 'f1': f1, 'r': r, 'p': p},
                   'end_epoch': end_epoch,
                   'log': config.log()
                   }, f,
                  ensure_ascii=False,
                  indent=4)
    save_path = save_dir + '/0best.json'
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            best_p = json.load(f)
    else:
        best_p = {'performance': {'acc': 0, 'f1': 0, 'r': 0, 'p': 0}, 'config': config.log()}
    if acc > best_p['performance']['acc']:
        best_p = {'performance': {'acc': acc, 'f1': f1, 'r': r, 'p': p}, 'config': config.log()}
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(best_p, f, ensure_ascii=False, indent=4)
