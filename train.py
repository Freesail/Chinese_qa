from torch import optim, nn
import torch

from DataPipeline import DataPipeline
from BiDAF import BiDAF


def train_val_model(pipeline_cfg, model_cfg, train_cfg):
    data_pipeline = DataPipeline(
        **pipeline_cfg
    )

    model = BiDAF(word_emb=data_pipeline.word_type.vocab.vectors, **model_cfg)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=train_cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    num_epochs = train_cfg['num_epochs']
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for i, batch in enumerate(data_pipeline.data_iterators[phase]):
                optimizer.zero_grad()
                with torch.set_grad_enable(phase == 'train'):
                    p1, p2 = model(batch)
                    loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if i % 100 == 0:
                            batch_loss = loss.item()
                            print('batch %d: loss %.3f' % (i, batch_loss))
