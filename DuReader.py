from torchtext import data
import os
import torch
import json
import pkuseg
import jieba
import re


# seg = pkuseg.pkuseg()
# def chinese_tokenize(tokens):
#     return [token.replace("''", '"').replace("``", '"') for token in seg.cut(tokens)]

def regex_change(line):
    # URL，为了防止对中文的过滤，所以使用[a-zA-Z0-9]而不是\w
    url_regex = re.compile(r"""
        (https?://)?
        ([a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)*
        (/[a-zA-Z0-9]+)*
    """, re.VERBOSE | re.IGNORECASE)
    # 剔除日期
    data_regex = re.compile(u"""        #utf-8编码
        年 |
        月 |
        日 |
        (周一) |
        (周二) | 
        (周三) | 
        (周四) | 
        (周五) | 
        (周六)
    """, re.VERBOSE)
    # 剔除所有数字
    decimal_regex = re.compile(r"[\d+\.\d*]]")
    # 剔除空格
    space_regex = re.compile(r"\s+")

    # eng
    eng_regex = re.compile(r'[a-zA-Z]')

    # punc
    punc_regex = re.compile(r'[\W]')  # ("[" + re.escape(string.punctuation) + "]")

    line = url_regex.sub(r"", line)
    line = data_regex.sub(r"", line)
    line = space_regex.sub(r"", line)
    line = decimal_regex.sub(r"", line)
    line = eng_regex.sub(r"", line)
    line = punc_regex.sub(r"", line)

    return line


def chinese_tokenize(tokens):
    return list(jieba.cut(tokens, cut_all=False))


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))
        if len(results) > 0:
            return results[0]


class DuReader:
    def __init__(self, raw_examples, processed_examples, saved_datasets):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processed_examples = processed_examples
        if not os.path.exists(processed_examples):
            print('preprocess raw examples')
            self.preprocess(raw_examples, processed_examples)
        self.id_type = data.RawField(is_target=False)
        self.word_type = data.Field(batch_first=True, include_lengths=True, tokenize=chinese_tokenize)
        self.idx_type = data.Field(sequential=False, use_vocab=False, unk_token=None)

        self.fields = {
            'id': ('id', self.id_type),
            'context': ('context', self.word_type),
            'question': ('question', self.word_type),
            's_idx': ('s_idx', self.idx_type),
            'e_idx': ('e_idx', self.idx_type),
        }

        if os.path.exists(saved_datasets):
            print('load datasets')
            self.datasets = self.load_datasets(saved_datasets)
        else:
            print('generate and save datasets')
            datasets = data.TabularDataset(
                path=self.processed_examples,
                format='json',
                fields=self.fields,
            ).split(split_ratio=0.7)

            self.datasets = {
                'train': datasets[0],
                'val': datasets[1]
            }
            self.save_datasets(saved_datasets)

        print('generate_vocab')
        self.word_type.build_vocab(*self.datasets.values())

        print('generate iterator')
        self.data_iterators = dict()
        self.data_iterators['train'] = data.BucketIterator(
            self.datasets['train'],
            batch_size=1,
            sort_key=lambda x: len(x.context),
            train=True,
            repeat=True,
            shuffle=True,
            sort=True,
            device=self.device,
        )
        self.data_iterators['val'] = data.BucketIterator(
            self.datasets['val'],
            batch_size=1,
            sort_key=lambda x: len(x.context),
            repeat=True,
            sort=True,
            device=self.device,
        )

    def save_datasets(self, saved_datasets):
        examples = {
            x: self.datasets[x].examples
            for x in ['train', 'val']
        }
        torch.save(examples, saved_datasets)

    def load_datasets(self, saved_datasets):
        examples = torch.load(saved_datasets)
        datasets = {
            x: data.Dataset(examples=examples[x], fields=list(self.fields.values()))
            for x in ['train', 'val']
        }
        return datasets

    @staticmethod
    def preprocess(raw_examples, processed_examples):
        dump = []

        with open(raw_examples, 'r') as f:  # , encoding='utf-8'
            raw_data = json.load(f)
            raw_data = raw_data['data']
            for article in raw_data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    context = regex_change(context)
                    con_tokens = chinese_tokenize(context)
                    if len(con_tokens) > 3:
                        for qa in paragraph['qas']:
                            example_id = qa['id']
                            question = qa['question']
                            for ans in qa['answers']:
                                answer = ans['text']
                                answer = regex_change(answer)
                                ans_tokens = chinese_tokenize(answer)
                                try:
                                    (s_idx, e_idx) = find_sub_list(ans_tokens, con_tokens)
                                    dump.append(dict([('id', example_id),
                                                      ('context', context),
                                                      ('question', question),
                                                      ('answer', answer),
                                                      ('s_idx', s_idx),
                                                      ('e_idx', e_idx)]))
                                except (TypeError, IndexError) as error:
                                    pass

        with open(processed_examples, 'w') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)


if __name__ == '__main__':
    du_reader = DuReader('raw_examples.json', 'processed_examples.json', 'train_val_dataset.pt')
