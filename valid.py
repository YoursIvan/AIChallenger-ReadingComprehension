import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utils.corpus import Corpus
from race.comatch import CoMatch
from race.evaluate import evaluation
import warnings


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
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--dropoutP', type=float, default=0.2,
                    help='dropout ratio')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--interval', type=int, default=200, metavar='N',
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

corpus = Corpus('RC')
model, optimizer, criterion = torch.load('trainedmodel/RC_save_best.pt')
score = evaluation(model, optimizer, criterion, corpus, args.cuda, args.batch_size,'valid')
#score1 = evaluation(model, optimizer, criterion, corpus, args.cuda, args.batch_size,'valid1')
#score2 = evaluation(model, optimizer, criterion, corpus, args.cuda, args.batch_size,'valid2')
print('DEV accuracy: ' + str(score))
#print('DEV1 accuracy: ' + str(score))
#print('DEV2 accuracy: ' + str(score))