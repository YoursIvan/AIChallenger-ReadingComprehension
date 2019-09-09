
import os
import torch
import glob
import json
import torch
import sys
import io

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')
#sys.stdin= io.TextIOWrapper(sys.stdin.buffer, encoding='gb18030')

def prep_glove():
    vocab = {}
    ivocab = []
    tensors = []
    with open('data/embedding/merge_sgns_bigram_char300.txt', 'r',encoding= 'utf-8') as f:
        line_count = 0
        for line in f:
            if line_count != 0:
                vals = line.rstrip().split(' ')
                print(line_count)
                if len(vals) != 301:
                    print(line)
                    continue
                assert(len(vals) == 301)

                word = vals[0]
                vec = torch.FloatTensor([ float(v) for v in vals[1:] ])
                vocab[word] = len(ivocab)
                ivocab.append(word)
                tensors.append(vec)
                assert (vec.size(0) == 300)
            line_count += 1

    assert len(tensors) == len(ivocab)
    tensors = torch.cat(tensors).view(len(ivocab), 300)
    with open('data/embedding/words_emb.pt', 'wb') as fpw:
        torch.save([tensors, vocab, ivocab], fpw)


class Dictionary(object):
    def __init__(self, task):
        self.task = task
        filename = os.path.join('data', self.task, 'word2idx.pt')

        if os.path.exists(filename):
            self.word2idx = torch.load(os.path.join('data', self.task, 'word2idx.pt'))
            self.idx2word = torch.load(os.path.join('data', self.task, 'idx2word.pt'))
            self.word2idx_count = torch.load(os.path.join('data', self.task, 'word2idx_count.pt'))

            self.char2idx = torch.load(os.path.join('data', self.task, 'char2idx.pt'))
            self.idx2char = torch.load(os.path.join('data', self.task, 'idx2char.pt'))
            self.char2idx_count = torch.load(os.path.join('data', self.task, 'char2idx_count.pt'))

        else:
            self.word2idx = {'<<padding>>':0, '<<unk>>':1}
            self.word2idx_count = {'<<padding>>':0, '<<unk>>':0}
            self.idx2word = ['<<padding>>', '<<unk>>']

            self.char2idx = {'<<padding>>': 0, '<<unk>>': 1}
            self.char2idx_count = {'<<padding>>': 0, '<<unk>>': 0}
            self.idx2char = ['<<padding>>', '<<unk>>']

            self.build_dict('train')
            self.build_dict('valid')
            if self.task != 'squad':
                self.build_dict('test')

            torch.save(self.word2idx, os.path.join('data', self.task, 'word2idx.pt'))
            torch.save(self.idx2word, os.path.join('data', self.task, 'idx2word.pt'))
            torch.save(self.word2idx_count, os.path.join('data', self.task, 'word2idx_count.pt'))

            torch.save(self.char2idx, os.path.join('data', self.task, 'char2idx.pt'))
            torch.save(self.idx2char, os.path.join('data', self.task, 'idx2char.pt'))
            torch.save(self.char2idx_count, os.path.join('data', self.task, 'char2idx_count.pt'))

        filename_emb = os.path.join('data', task, 'embeddings.pt')
        if os.path.exists(filename_emb):
            self.embs = torch.load(filename_emb)
        else:
            self.embs = self.build_emb()

        print ("vacabulary size: " + str(len(self.idx2word)))

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.word2idx_count[word] = 1
        else:
            self.word2idx_count[word] += 1

        return self.word2idx[word]

    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = len(self.idx2char)
            self.idx2char.append(char)
            self.char2idx_count[char] = 1
        else:
            self.char2idx_count[char] += 1

        return self.char2idx[char]

    def build_dict(self, dataset):
        filename = os.path.join('data', self.task, 'sequenceAfterWash', dataset+'.json')

        if self.task == 'RC':
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_all = json.load(fpr)
                for instance in data_all:
                    print(dataset + '......')

                    words = instance['question']
                    for option in instance['options']: words += option
                    for sent in instance['article']: words += sent
                    for word in words: self.add_word(word)

                    chars = instance['question_c']
                    for option_c in instance['options_c']: chars += option_c
                    for sent_c in instance['article_c']: chars += sent_c
                    for char in chars: self.add_char(char)
        else:
            assert False, 'the task ' + self.task + ' is not supported!'

    def build_emb(self, all_vacob=False, filter=False, threshold=10):
        word2idx = torch.load(os.path.join('data', self.task, 'word2idx.pt'))
        idx2word = torch.load(os.path.join('data', self.task, 'idx2word.pt'))

        emb = torch.FloatTensor(len(idx2word), 300).zero_()
        print ("Loading Embedding ...")
        print ("Raw vacabulary size: " + str(len(idx2word)) )

        if not os.path.exists('data/embedding/words_emb.pt'): prep_glove()
        glove_tensors, glove_vocab, glove_ivocab = torch.load('data/embedding/words_emb.pt')

        if not all_vacob:
            self.word2idx = {'<<padding>>':0, '<<unk>>':1}
            self.idx2word = ['<<padding>>', '<<unk>>']
        count = 0
        for w_id, word in enumerate(idx2word):
            if word in glove_vocab:
                id = self.add_word(word)
                emb[id] = glove_tensors[glove_vocab[word]]
                count += 1
        emb = emb[:len(self.idx2word)]

        print("Number of words not appear in glove: " + str(len(idx2word)-count) )
        print ("Vacabulary size: " + str(len(self.idx2word)))
        torch.save(emb, os.path.join('data', self.task, 'embeddings.pt'))
        torch.save(self.word2idx, os.path.join('data', self.task, 'word2idx.pt'))
        torch.save(self.idx2word, os.path.join('data', self.task, 'idx2word.pt'))
        return emb

    def filter(self, threshold=10):
        for word, count in self.word2idx_count.items():
            if count > threshold and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, task):
        self.task = task
        self.dictionary = Dictionary(task)
        self.data_all, self.start_id, self.indices = {}, {}, {}
        setnames = ['train', 'valid','test']
        for setname in setnames:
            self.data_all[setname] = self.load_data(os.path.join('data', self.task, 'sequenceAfterWash', setname) + '.json')
            print(setname, len(self.data_all[setname]))
            self.start_id[setname] = 0
            self.indices[setname] = torch.randperm(len(self.data_all[setname])) if setname == 'train' else torch.range(0, len(self.data_all[setname])-1)

    def seq2tensor(self, words):
        seq_tensor = torch.LongTensor(len(words))
        for id, word in enumerate(words):
            seq_tensor[id] = self.dictionary.word2idx[word] if word in self.dictionary.word2idx else 1
        return seq_tensor

    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as fpr:
            data = json.load(fpr)
        return data

    def get_batch(self, batch_size, setname):
        if self.start_id[setname] >= len(self.data_all[setname]):
            self.start_id[setname] = 0
            if setname == 'train': self.indices[setname] = torch.randperm(len(self.data_all[setname]))

        end_id = self.start_id[setname] + batch_size if self.start_id[setname] + batch_size < len(self.data_all[setname]) else len(self.data_all[setname])
        documents, questions, options, labels = [], [], [], []
        documents_c,questions_c,options_c = [],[],[]
        for i in range(self.start_id[setname], end_id):
            instance_id = int(self.indices[setname][i])

            instance = self.data_all[setname][instance_id]

            questions.append(instance['question'])
            options.append(instance['options'])
            documents.append(instance['article'])

            questions_c.append(instance['question_c'])
            options_c.append(instance['options_c'])
            documents_c.append(instance['article_c'])

            if setname!='test':
                labels.append(instance['ground_truth'])

        self.start_id[setname] += batch_size

        questions = self.seq2tensor(questions)
        documents = self.seq2Htensor(documents)
        options = self.seq2Htensor(options)

        questions_c = self.seq2tensor_c(questions_c)
        documents_c = self.seq2Htensor_c(documents_c)
        options_c = self.seq2Htensor_c(options_c)

        labels = torch.LongTensor(labels)
        return [documents, questions, options,documents_c,questions_c,options_c,labels]




    def seq2tensor(self, sents, sent_len_bound=50):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, sent_len_bound)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                sent_tensor[s_id][w_id] = self.dictionary.word2idx.get(word, 1)
        return [sent_tensor, sent_len]

    def seq2Htensor(self, docs, sent_num_bound=50, sent_len_bound=50):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, sent_num_bound)
        sent_len_max = max([len(w) for s in docs for w in s ])
        sent_len_max = min(sent_len_max, sent_len_bound)

        sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.dictionary.word2idx.get(word, 1)
        return [sent_tensor, doc_len, sent_len]

    def seq2tensor_c(self, sents, sent_len_bound=50, word_len_bound=10):
        sent_len_max = max([len(w) for w in sents])
        sent_len_max = min(sent_len_max, sent_len_bound)

        word_len_max = max([len(c) for w in sents for c in w])
        word_len_max = min(word_len_max,word_len_bound)

        char_tensor = torch.LongTensor(len(sents), sent_len_max, word_len_max).zero_()
        word_len = torch.LongTensor(len(sents), sent_len_max).zero_()
        sent_len = torch.LongTensor(len()).zero_()

        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                word_len[s_id][w_id] = len(word)
                for c_id, char in enumerate(word):
                    if c_id >= word_len_max: break
                    char_tensor[s_id][w_id][c_id] = self.dictionary.char2idx.get(char, 1)
        return [char_tensor, sent_len, word_len]

    def seq2Htensor_c(self, docs, sent_num_bound=50, sent_len_bound=50, word_len_bound=10):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, sent_num_bound)
        sent_len_max = max([len(w) for s in docs for w in s ])
        sent_len_max = min(sent_len_max, sent_len_bound)
        word_len_max = max([len(c) for s in docs for w in s for c in w])
        word_len_max = min(word_len_max, word_len_bound)

        char_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max, word_len_max).zero_()
        word_len = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    word_len[d_id][s_id][w_id] = len(word)
                    for c_id, char in enumerate(word):
                        if c_id>= word_len_max: break
                        char_tensor[d_id][s_id][w_id][c_id] = self.dictionary.char2idx.get(char, 1)

        return [char_tensor, doc_len, sent_len, word_len]
