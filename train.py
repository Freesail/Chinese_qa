from collections import Counter
import copy
from torch import optim, nn
import torch
import json
from DataPipeline import DataPipeline
from model import BiDAF
from ema import EMA


def f1_score(pred, gt):
    pred = [str(x) for x in range(pred[0], pred[1] + 1)]
    gt = [str(x) for x in range(gt[0], gt[1] + 1)]

    common = Counter(pred) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def r_score(pred, gt):
    pred = [str(x) for x in range(pred[0], pred[1] + 1)]
    gt = [str(x) for x in range(gt[0], gt[1] + 1)]

    common = Counter(pred) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(gt)
    beta = precision / recall
    rouge = ((1 + beta ** 2) * precision * recall) / ((beta ** 2) * precision + recall)
    return rouge


def exact_match_score(pred, gt):
    return pred == gt


def train_val_model(pipeline_cfg, model_cfg, train_cfg):
    data_pipeline = DataPipeline(
        **pipeline_cfg
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_cfg['cxt_emb_pretrained'] is not None:
        model_cfg['cxt_emb_pretrained'] = torch.load(model_cfg['cxt_emb_pretrained'])
    bidaf = BiDAF(word_emb=data_pipeline.word_type.vocab.vectors, **model_cfg)
    ema = EMA(train_cfg['exp_decay_rate'])
    for name, param in bidaf.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, bidaf.parameters())
    optimizer = optim.Adadelta(parameters, lr=train_cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    result = {
        'best_f1': 0.0,
        'best_model': None
    }

    num_epochs = train_cfg['num_epochs']
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            val_answers = dict()
            val_f1 = 0
            val_em = 0
            val_cnt = 0
            val_r = 0

            if phase == 'train':
                bidaf.train()
            else:
                bidaf.eval()
                backup_params = EMA(0)
                for name, param in bidaf.named_parameters():
                    if param.requires_grad:
                        backup_params.register(name, param.data)
                        param.data.copy_(ema.get(name))

            with torch.set_grad_enabled(phase == 'train'):
                for batch_num, batch in enumerate(data_pipeline.data_iterators[phase]):
                    optimizer.zero_grad()
                    p1, p2 = bidaf(batch)
                    loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        for name, param in bidaf.named_parameters():
                            if param.requires_grad:
                                ema.update(name, param.data)
                        if batch_num % train_cfg['batch_per_disp'] == 0:
                            batch_loss = loss.item()
                            print('batch %d: loss %.3f' % (batch_num, batch_loss))

                    if phase == 'val':
                        batch_size, c_len = p1.size()
                        val_cnt += batch_size
                        ls = nn.LogSoftmax(dim=1)
                        mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1). \
                            unsqueeze(0).expand(batch_size, -1, -1)
                        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
                        score, s_idx = score.max(dim=1)
                        score, e_idx = score.max(dim=1)
                        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

                        for i in range(batch_size):
                            answer = (s_idx[i], e_idx[i])
                            gt = (batch.s_idx[i], batch.e_idx[i])
                            val_f1 += f1_score(answer, gt)
                            val_em += exact_match_score(answer, gt)
                            val_r += r_score(answer, gt)

            if phase == 'val':
                val_f1 = val_f1 * 100 / val_cnt
                val_em = val_em * 100 / val_cnt
                val_r = val_r * 100 / val_cnt
                print('Epoch %d: %s f1 %.3f | %s em %.3f |  %s rouge %.3f'
                      % (epoch, phase, val_f1, phase, val_em, phase, val_r))
                if val_f1 > result['best_f1']:
                    result['best_f1'] = val_f1
                    result['best_em'] = val_em
                    result['best_model'] = copy.deepcopy(bidaf.state_dict())
                    torch.save(result, train_cfg['ckpoint_file'])
                    # with open(train_cfg['val_answers'], 'w', encoding='utf-8') as f:
                    #     print(json.dumps(val_answers), file=f)
                for name, param in bidaf.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(backup_params.get(name))
