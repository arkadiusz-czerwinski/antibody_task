#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:04:06 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

from lstm_bi import LSTM_Bi
from utils_data import ProteinSeqDataset, aa2id_i, aa2id_o, collate_fn
from tqdm import tqdm
import numpy as np
import torch
import sys

class ModelLSTM:
    def __init__(self, in_dim=40, embedding_dim=64, hidden_dim=64, device='cpu', gapped=True, fixed_len=True, out_dim=None, max_len=131):
        self.gapped = gapped
        out_dim if out_dim is not None else len(aa2id_o[gapped])
        self.nn = LSTM_Bi(in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len, max_len)
        self.to(device)
        
    def fit(self, dataset, n_epoch=10, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None):
        # loss function and optimization algorithm
        loss_fn = torch.nn.BCELoss()  # binary cross entropy
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        
        # to track minimum validation loss
        min_loss = np.inf
        
        batch_size = trn_batch_size
        train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size)
        val_dataloader = torch.utils.data.DataLoader(dataset['val'], batch_size=batch_size)
        for epoch in range(n_epoch):
            # training
            self.nn.train()
            loss_avg, acc_avg, cnt = 0, 0, 0
            with tqdm(total=len(dataset['train']), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for batch in train_dataloader:
                    X = batch['input_ids']
                    y = batch['label']
                    y = torch.tensor(y, device=self.nn.device, dtype=torch.float)
                    X = torch.tensor(X, device=self.nn.device)
                    # targets
                    
                    # forward and backward routine
                    self.nn.zero_grad()
                    scores = self.nn(X, aa2id_i[self.gapped]).squeeze()
                    loss = loss_fn(scores, y)
                    loss.backward()
                    op.step()
                    
                    # compute statistics
                    L = X.shape[0]
                    predicted = torch.tensor(scores > 0.5, dtype=torch.long).flatten()
                    loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                    corr = (predicted == y.flatten()).data.cpu().numpy()
                    acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
                    cnt += L
                    
                    # update progress bar
                    pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
                    pbar.update(X.shape[0])
            
            # validation
            self.nn.eval()
            loss_avg, acc_avg, cnt = 0, 0, 0
            with torch.set_grad_enabled(False):
                with tqdm(total=len(dataset['val']), desc='          (VLD)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                    for batch in val_dataloader:
                        X = batch['input_ids']
                        y = batch['label']
                        y = torch.tensor(y, device=self.nn.device, dtype=torch.float)
                        X = torch.tensor(X, device=self.nn.device)
                        # targets
                        
                        # forward routine
                        scores = self.nn(X, aa2id_i[self.gapped]).flatten()
                        loss = loss_fn(scores, y)
                        
                        # compute statistics
                        L = X.shape[0]
                        predicted = torch.tensor(scores > 0.5, dtype=torch.long).flatten()
                        loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                        corr = (predicted == y.flatten()).data.cpu().numpy()
                        acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
                        cnt += L
                        
                        # update progress bar
                        pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
                        pbar.update(X.shape[0])
            
            # save model
            if loss_avg < min_loss and save_fp:
                min_loss = loss_avg
                self.save('{}/lstm_{:.6f}.npy'.format(save_fp, loss_avg))
    
    def eval(self, dataset, batch_size=512):        
        # dataset and dataset loader
        test_dataset = torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size)

        self.nn.eval()
        scores = []
        sys.stdout.flush()
        with torch.set_grad_enabled(False):
            with tqdm(total=len(dataset['test']), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for n, (batch) in enumerate(test_dataset):
                    X = batch['input_ids']
                    y = batch['label']
                    y = torch.tensor(y, device=self.nn.device, dtype=torch.float)
                    X = torch.tensor(X, device=self.nn.device)
                    out = self.nn(X, aa2id_i[self.gapped]).squeeze()
                    predicted = torch.tensor(out > 0.5, dtype=torch.long).flatten()
                    preds = predicted == y
                    for pred in preds:
                        scores.append(pred.cpu().numpy())
                    pbar.update(X.shape[0])
        print(sum(scores)/len(scores))
        return scores
    
    def save(self, fn):
        param_dict = self.nn.get_param()
        param_dict['gapped'] = self.gapped
        np.save(fn, param_dict)
    
    def load(self, fn):
        param_dict = np.load(fn, allow_pickle=True).item()
        self.gapped = param_dict['gapped']
        self.nn.set_param(param_dict)

    def to(self, device):
        self.nn.to(device)
        self.nn.device = device
        
    def summary(self):
        for n, w in self.nn.named_parameters():
            print('{}:\t{}'.format(n, w.shape))
#        print('LSTM: \t{}'.format(self.nn.lstm_f.all_weights))
        print('Fixed Length:\t{}'.format(self.nn.fixed_len) )
        print('Gapped:\t{}'.format(self.gapped))
        print('Device:\t{}'.format(self.nn.device))
            
