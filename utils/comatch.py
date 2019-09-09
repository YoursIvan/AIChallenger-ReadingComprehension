'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i,:,:seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)

    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class SelfMatchNet(nn.Module):
    def __init__(self,mem_dim,attn_size,):
        self.params = {
            "W_h_P": Variable(torch.zeros(2*mem_dim,mem_dim)),
            "W_v_Phat": Variable(torch.zeros(2*mem_dim,mem_dim)),
            "W_h_a": Variable(torch.zeros(2*mem_dim,mem_dim)),
            "v": Variable(torch.zeros(mem_dim,1))}
        self.gated_attention_module =


    def forward(self, inputs):
        passage,passage_w_len = inputs
        passage_score = self.gated_attention_module(passage,passage_w_len)

class MatchNet(nn.Module):
    def __init__(self, mem_dim, dropoutP):
        super(MatchNet, self).__init__()
        self.map_linear = nn.Linear(2*mem_dim, 2*mem_dim)
        self.trans_linear = nn.Linear(mem_dim, mem_dim)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm( torch.transpose(proj_q, 1, 2) )
        att_norm = masked_softmax(att_weights, seq_len)
        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min,elem_mul], 2)

        output = nn.ReLU()(self.map_linear(all_con))

        return output

class MaskLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.3):
        super(MaskLSTM, self).__init__()
        self.lstm_module = nn.LSTM(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional, dropout=dropoutP)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i,:seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.drop_module(input*mask_in)

        H, _ = self.lstm_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i,:seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask
        return output

class MaskLSTM_c(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP = 0.3):
        super(MaskLSTM_c, self).__init__()
        self.lstm_module = nn.LSTM(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional, dropout=dropoutP)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        input, seq_lens = inputs
        input_drop = self.drop_module(input)
        H, [h0,c0] = self.lstm_module(input_drop)

        return torch.squeeze(h0[0])


class CoMatch(nn.Module):
    def __init__(self, corpus, args):
        super(CoMatch, self).__init__()
        self.emb_dim = 300
        self.att_size = 300
        self.mem_dim = args.mem_dim
        self.dropoutP = args.dropoutP
        self.cuda_bool = args.cuda

        self.wordembs = nn.Embedding(len(corpus.dictionary), self.emb_dim)
        self.wordembs.weight.data.copy_(corpus.dictionary.embs)
        self.wordembs.weight.requires_grad = False

        self.charembs = nn.Embedding(len(corpus.dictionary.idx2char), self.emb_dim)
        self.charembs.weight.requires_grad = True

        self.encoder = MaskLSTM(self.emb_dim * 2, self.mem_dim, dropoutP=self.dropoutP)
        self.c_encoder = MaskLSTM_c(self.emb_dim, self.emb_dim, dropoutP=self.dropoutP)
        self.l_encoder = MaskLSTM(self.mem_dim * 8, self.mem_dim, dropoutP=self.dropoutP)
        self.h_encoder = MaskLSTM(self.mem_dim * 2, self.mem_dim, dropoutP=0)

        self.self_match_module = SelfMatchNet(self.mem_dim*2,self.att_size)

        self.match_module = MatchNet(self.mem_dim * 2, self.dropoutP)
        self.rank_module = nn.Linear(self.mem_dim * 2, 1)

        self.drop_module = nn.Dropout(self.dropoutP)
        self.initiation()

    def initiation(self):
        initrange = 0.1
        nn.init.uniform_(self.charembs.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        documents, questions, options, documents_c, questions_c, options_c, _ = inputs  # Batch = 5
        d_word, d_h_len, d_l_len = documents  # [5,31,46] #31 #46
        o_word, o_h_len, o_l_len = options
        q_word, q_len = questions  # [5,14]

        d_char, _, _, d_w_len = documents_c
        o_char, _, _, o_w_len = options_c
        q_char, _, q_w_len = questions_c

        if self.cuda_bool:
            d_word, d_h_len, d_l_len, o_word, o_h_len, o_l_len, q_word, q_len, d_w_len, o_w_len, q_w_len, d_char, o_char, q_char = d_word.cuda(), d_h_len.cuda(), d_l_len.cuda(), o_word.cuda(), o_h_len.cuda(), o_l_len.cuda(), q_word.cuda(), q_len.cuda(), d_w_len.cuda(), o_w_len.cuda(), q_w_len.cuda(), d_char.cuda(), o_char.cuda(), q_char.cuda()

        d_char_embs = self.drop_module(Variable(self.charembs(d_char), requires_grad=True))  # [5,31,46,n,300]
        o_char_embs = self.drop_module(Variable(self.charembs(o_char), requires_grad=True))  # {5,4,10,n,300]
        q_char_embs = self.drop_module(Variable(self.charembs(q_char), requires_grad=True))  # [5,14,n,300]

        d_char_hidden = self.c_encoder([d_char_embs.view(
            d_char_embs.size(0) * d_char_embs.size(1) * d_char_embs.size(2), d_char_embs.size(3), self.emb_dim),
                                        d_w_len.view(-1)])  # [m,n,300]
        o_char_hidden = self.c_encoder([o_char_embs.view(
            o_char_embs.size(0) * o_char_embs.size(1) * o_char_embs.size(2), o_char_embs.size(3), self.emb_dim),
                                        o_w_len.view(-1)])
        q_char_hidden = self.c_encoder(
            [q_char_embs.view(q_char_embs.size(0) * q_char_embs.size(1), q_char_embs.size(2), self.emb_dim), q_w_len])

        d_char_hidden_3d = d_char_hidden.view(d_char_embs.size(0), d_char_embs.size(1) * d_char_embs.size(2),
                                              self.emb_dim)
        o_char_hidden_3d = o_char_hidden.view(o_char_embs.size(0), o_char_embs.size(1) * o_char_embs.size(2),
                                              self.emb_dim)
        q_char_hidden_3d = q_char_hidden.view(q_char_embs.size(0), q_char_embs.size(1), self.emb_dim)

        d_char_hidden_4d = d_char_hidden_3d.view(d_char_embs.size(0), d_char_embs.size(1), d_char_embs.size(2),
                                                 self.emb_dim)
        o_char_hidden_4d = o_char_hidden_3d.view(o_char_embs.size(0), o_char_embs.size(1), o_char_embs.size(2),
                                                 self.emb_dim)

        d_embs = self.drop_module(Variable(self.wordembs(d_word), requires_grad=False))  # [5,31,46,300]
        o_embs = self.drop_module(Variable(self.wordembs(o_word), requires_grad=False))  # {5,4,10,300]
        q_embs = self.drop_module(Variable(self.wordembs(q_word), requires_grad=False))  # [5,14,300]

        d_embs = torch.cat([d_embs, d_char_hidden_4d], -1)
        o_embs = torch.cat([o_embs, o_char_hidden_4d], -1)
        q_embs = torch.cat([q_embs, q_char_hidden_3d], -1)

        d_hidden = self.encoder([d_embs.view(d_embs.size(0) * d_embs.size(1), d_embs.size(2), self.emb_dim * 2),
                                 d_l_len.view(-1)])  # [155,46,300]
        o_hidden = self.encoder([o_embs.view(o_embs.size(0) * o_embs.size(1), o_embs.size(2), self.emb_dim * 2),
                                 o_l_len.view(-1)])  # {20,10,300]
        q_hidden = self.encoder([q_embs, q_len])  # [5,14,300]

        # self-match
        passage_hidden = d_hidden.view(d_embs.size(0),d_embs.size(1)*d_embs.size(2),d_hidden.size(-1))
        passage_hidden = self.self_match_module(passage_hidden)

        d_self_match = passage_hidden.view(d_embs.size(0) * d_embs.size(1), d_embs.size(2), self.emb_dim * 2)


        d_hidden_3d = d_self_match.view(d_embs.size(0), d_embs.size(1) * d_embs.size(2), d_hidden.size(-1))  # [5,1426,300]
        d_hidden_3d_repeat = d_self_match.repeat(1, o_embs.size(1), 1).view(d_hidden_3d.size(0) * o_embs.size(1),
                                                                           d_hidden_3d.size(1),
                                                                           d_hidden_3d.size(2))  # [20,1426,300]

        do_match = self.match_module([d_hidden_3d_repeat, o_hidden, o_l_len.view(-1)])  # [20,1426,600]
        dq_match = self.match_module([d_hidden_3d, q_hidden, q_len])  # [5,1426,600]

        dq_match_repeat = dq_match.repeat(1, o_embs.size(1), 1).view(dq_match.size(0) * o_embs.size(1),
                                                                     dq_match.size(1),
                                                                     dq_match.size(2))  # [20,1426,600]

        co_match = torch.cat([do_match, dq_match_repeat], -1)  # [20,1426,1200]
        co_match_hier = co_match.view(d_embs.size(0) * o_embs.size(1) * d_embs.size(1), d_embs.size(2),
                                      -1)  # [620,46,1200]

        l_hidden = self.l_encoder([co_match_hier, d_l_len.repeat(1, o_embs.size(1)).view(-1)])  # [620,46,300]
        l_hidden_pool, _ = l_hidden.max(1)  # [620,300]

        h_hidden = self.h_encoder([l_hidden_pool.view(d_embs.size(0) * o_embs.size(1), d_embs.size(1), -1),
                                   d_h_len.view(-1, 1).repeat(1, o_embs.size(1)).view(-1)])  # [20,31,300]
        h_hidden_pool, _ = h_hidden.max(1)  # [20,300]

        o_rep = h_hidden_pool.view(d_embs.size(0), o_embs.size(1), -1)  # [5,4,300]

        output = torch.nn.functional.log_softmax(self.rank_module(o_rep).squeeze(2))  # [5,4]

        return output
