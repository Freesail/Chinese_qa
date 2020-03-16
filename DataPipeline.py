from torchtext import data
import os
import torch
from torchtext import vocab
from torchtext.vocab import GloVe
import nltk
import json
import jieba
import re
import dill


class DataPipeline:
    def __init__(self, raw_folder,
                 train_file,
                 processed_folder,
                 saved_datasets,
                 saved_field,
                 val_file=None,
                 language='English',
                 context_threshold=-1,
                 batch_size=64):

        self.raw_fold = raw_folder
        self.train_file = train_file
        self.processed_folder = processed_folder
        self.saved_datasets = saved_datasets
        self.val_file = val_file
        self.language = language
        self.context_threshold = context_threshold
        self.batch_size = batch_size
        self.saved_field = saved_field
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.language == 'English':
            self.tokenize = lambda x: \
                [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(x)]
        else:
            self.tokenize = lambda x: list(jieba.cut(x, cut_all=False))

        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)
            print('preprocess raw examples')
            for examples in os.listdir(raw_folder):
                raw_examples = os.path.join(raw_folder, examples)
                processed_examples = os.path.join(processed_folder, examples)
                if self.language == 'English':
                    self.process_eng(raw_examples, processed_examples)
                else:
                    self.process_ch(raw_examples, processed_examples)

        self.id_type = data.RawField(is_target=False)

        if os.path.exists(saved_field):
            # self.word_type = torch.load(saved_field)
            with open(saved_field, 'rb') as f:
                self.word_type = dill.load(f)
        else:
            if self.language == 'English':
                word_vector = GloVe(name='6B', dim=100)
                self.word_type = data.Field(batch_first=True, include_lengths=True,
                                            tokenize=self.tokenize, lower=True)
            else:
                word_vector = vocab.Vectors('Tencent_AILab_ChineseEmbedding.txt')
                self.word_type = data.Field(batch_first=True, include_lengths=True,
                                            tokenize=self.tokenize)
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
            # if self.language == 'English':
            datasets = data.TabularDataset.splits(
                path=processed_folder,
                train=train_file,
                validation=val_file,
                format='json',
                fields=self.fields)
            # else:
            #     datasets = data.TabularDataset(
            #         path=os.path.join(processed_folder, train_file),
            #         format='json',
            #         fields=self.fields,
            #     ).split(split_ratio=0.7)

            if self.context_threshold > 0:
                datasets[0].examples = \
                    [e for e in datasets[0].examples if len(e.context) <= self.context_threshold]

            self.datasets = {
                'train': datasets[0],
                'val': datasets[1]
            }
            self.save_datasets(saved_datasets)

        if os.path.exists(saved_field):
            pass
        else:
            print('generate_vocab')
            self.word_type.build_vocab(*self.datasets.values(), vectors=word_vector)
            with open(saved_field, 'wb') as f:
                dill.dump(self.word_type, f)
            # torch.save(self.word_type, saved_field)

        print('generate iterator')
        self.data_iterators = dict()
        self.data_iterators['train'] = data.BucketIterator(
            self.datasets['train'],
            batch_size=batch_size,
            sort_key=lambda x: len(x.context),
            train=True,
            shuffle=True,
            sort=True,
            device=self.device,
        )
        self.data_iterators['val'] = data.BucketIterator(
            self.datasets['val'],
            batch_size=batch_size,
            sort_key=lambda x: len(x.context),
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

    def process_eng(self, raw_examples, processed_examples):
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(raw_examples, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = self.tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))

        with open(processed_examples, 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)

    def process_ch(self, raw_examples, processed_examples):

        def regex_change(line):
            eng_regex = re.compile(r'[a-zA-Z]')
            punc_regex = re.compile(r'[\W]')  # ("[" + re.escape(string.punctuation) + "]")
            decimal_regex = re.compile(r'[\d]')

            line = punc_regex.sub(r"", line)
            line = decimal_regex.sub(r"", line)
            line = eng_regex.sub(r"", line)

            return line

        def find_sub_list(sl, l):
            sll = len(sl)
            i = len(l)
            for idx, ele in enumerate(l):
                if ele == sl[0]:
                    if l[idx:idx + sll] == sl:
                        i = idx
                        break
            l = l[:i + sll] + '\u5929\u5b89\u95e8 ' + l[i + sll:]
            l = l[:i] + '\u9955\u992e' + l[i:]

            l = self.tokenize(l)
            end = l.index('\u5929\u5b89\u95e8')
            start = l.index('\u9955\u992e')
            return start, end-2

        dump = []
        with open(raw_examples, 'r') as f:  # , encoding='utf-8'
            raw_data = json.load(f)
            raw_data = raw_data['data']
            for article in raw_data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    context = regex_change(context)
                    con_tokens = self.tokenize(context)
                    if len(con_tokens) > 3:
                        for qa in paragraph['qas']:
                            example_id = qa['id']
                            question = qa['question']
                            for ans in qa['answers']:
                                answer = ans['text']
                                answer = regex_change(answer)
                                try:
                                    (s_idx, e_idx) = find_sub_list(answer, context)
                                    if e_idx < len(con_tokens):
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
