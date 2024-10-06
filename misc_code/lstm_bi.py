#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:25 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np

class LSTM_Bi(nn.Module):
    def __init__(self, in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len, max_len):
        super(LSTM_Bi, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.word_embeddings = nn.Embedding(in_dim, embedding_dim, padding_idx=0)
        self.lstm_f = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_b = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * max_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Sigmoid()
        self.fixed_len = fixed_len
        self.max_len = max_len

    def forward(self, Xs):
        # torch.save(Xs.to("cpu"), 'tensor.pt')
        Xs_f = Xs
        Xs_b = torch.flip(Xs, dims=[-1])
        Xs_b = torch.tensor(Xs_b, device=self.device)
        Xs_f = torch.tensor(Xs_f, device=self.device)
        bs = Xs_b.shape[0]

        Xs_len = [Xs_b.shape[1]] * bs
        
        # embedding
        Xs_f = self.word_embeddings(Xs_f)
        Xs_b = self.word_embeddings(Xs_b)
        
        # packing the padded sequences
        Xs_f = pack_padded_sequence(Xs_f, Xs_len, batch_first=True, enforce_sorted=False)
        Xs_b = pack_padded_sequence(Xs_b, Xs_len, batch_first=True, enforce_sorted=False)
        
        # feed the lstm by the packed input
        ini_hc_state_f = (torch.zeros(1, bs, self.hidden_dim).to(self.device),
                          torch.zeros(1, bs, self.hidden_dim).to(self.device))
        ini_hc_state_b = (torch.zeros(1, bs, self.hidden_dim).to(self.device),
                          torch.zeros(1, bs, self.hidden_dim).to(self.device))

        lstm_out_f, _ = self.lstm_f(Xs_f, ini_hc_state_f)
        lstm_out_b, _ = self.lstm_b(Xs_b, ini_hc_state_b)
        
        # unpack outputs
        lstm_out_f, lstm_out_len = pad_packed_sequence(lstm_out_f, batch_first=True)
        lstm_out_b, _            = pad_packed_sequence(lstm_out_b, batch_first=True)
        lstm_out_f = lstm_out_f.flatten(1,2)
        lstm_out_b = lstm_out_b.flatten(1,2)
        out = lstm_out_f + lstm_out_b
        # lstm hidden state to output space
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        # compute scores
        # scores = F.log_softmax(out, dim=1)
        out = self.softmax(out)
        
        return out  
    
    def set_param(self, param_dict):
        try:
            for pn, _ in self.named_parameters():
                exec('self.%s.data = torch.tensor(param_dict[pn])' % pn)
            self.hidden_dim = param_dict['hidden_dim']
            self.fixed_len = param_dict['fixed_len']
            self.forward = self.forward_flen if self.fixed_len else self.forward_vlen
            self.to(self.device)
        except:
            print('Unmatched parameter names or shapes.')      
    
    def get_param(self):
        param_dict = {}
        for pn, pv in self.named_parameters():
            param_dict[pn] = pv.data.cpu().numpy()
        param_dict['hidden_dim'] = self.hidden_dim
        param_dict['fixed_len'] = self.fixed_len
        return param_dict
        