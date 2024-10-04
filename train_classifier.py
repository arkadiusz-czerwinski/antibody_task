from ablstm import ModelLSTM
import torch
# initialize model
# change device to 'cpu' if CUDA is not working properly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ModelLSTM(embedding_dim=64, hidden_dim=64, device=device, gapped=True, fixed_len=True)
print('Model initialized.')


# data files
trn_fn = './data/sample/human_train.txt'
vld_fn = './data/sample/human_val.txt'

# fit model w/o save
model.fit(trn_fn=trn_fn, vld_fn=vld_fn, n_epoch=1, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None)
# # fit model w/ save
# model.fit(trn_fn=trn_fn, vld_fn=vld_fn, n_epoch=1, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp='./saved_models/tmp')
print('Done.')