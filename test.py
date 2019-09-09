'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import torch
from utils.corpus import Corpus
from race.comatch import CoMatch
import argparse
import torch
from utils.corpus import Corpus
import json
import warnings

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Multiple Choice Reading Comprehension')
parser.add_argument('--task', type=str, default='RC',
                    help='task name')
parser.add_argument('--model', type=str, default='CoMatch',
                    help='model name')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--mem_dim', type=int, default=150,
                    help='hidden memory size')
parser.add_argument('--lr', type=float, default=0.002,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                    help='batch size')
parser.add_argument('--dropoutP', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--interval', type=int, default=5000, metavar='N',
                    help='report interval')
parser.add_argument('--exp_idx', type=str,  default='1',
                    help='experiment index')
parser.add_argument('--log', type=str,  default='nothing',
                    help='take note')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

def accuracy(ground_truth, prediction):
    assert(len(ground_truth) == len(prediction))
    accuracy = float( (ground_truth==prediction).float().mean(0) )
    return accuracy

def evaluation(model, optimizer, criterion, corpus, cuda, batch_size, dataset='valid'):
    model.eval()

    pred_all = []
    total_loss = 0
    count = 0
    num = 0
    while True:
        data = corpus.get_batch(batch_size, dataset)
        output = model(data)
        _, pred = output.max(1)
        pred_all.append(pred.cpu())
        if corpus.start_id[dataset] >= len(corpus.data_all[dataset]): break

    id = 280001
    with open ('data/' + args.task + '/test.json', 'r', encoding='utf-8') as fp:
        line = fp.readline()
        with open('trainedmodel/' + args.task + '_result.txt', 'w', encoding='utf-8') as fpw:
            for pred_batch in pred_all:
                for pred in pred_batch.numpy():
                    a = json.loads(line)
                    if pred == 0:
                        fpw.write(str(id))
                        fpw.write('\t')
                        fpw.write(a['alternatives'].split('|')[0])
                    elif pred == 1:
                        fpw.write(str(id))
                        fpw.write('\t')
                        fpw.write(a['alternatives'].split('|')[1])
                    else:
                        fpw.write(str(id))
                        fpw.write('\t')
                        fpw.write(a['alternatives'].split('|')[2])
                    fpw.write('\n')
                    id+=1
                    line = fp.readline()


corpus = Corpus('RC')
model = eval(args.model)(corpus, args)
model.train()
parameters = filter(lambda p: p.requires_grad, model.parameters())

model, optimizer, criterion = torch.load('trainedmodel/RC_save_best.pt')
evaluation(model, optimizer, criterion, corpus, args.cuda, args.batch_size, dataset='test')






