"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#from model.gcn import GCNClassifier
from model.gcn_lstm_elmo_cdr import GCNClassifier
from utils import constant, torch_utils

class Trainer(object):
    def __init__(self, opt,knowledge_emb=None,word_emb=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    """
    20
    tokens_elmoid, masks, pos, head,
    subj_mask, obj_mask, dis1, dis2,
    all_two_mesh_index, token_id, subj_positions, obj_positions,
    rels, orig_idx, batch[0], batch[7],
    batch[8], batch[14], batch[15],batch[16]
    """
    labels = batch[16]
    if cuda:
        inputs = [b.cuda() for b in batch[:16]]
        labels = labels.cuda()
    else:
        inputs = [b for b in batch[:16]]
        labels = labels
    return inputs, labels

class GCNTrainer(Trainer):
    def __init__(self, opt,knowledge_emb=None,word_emb=None):
        self.opt = opt
        self.knowledge_emb = knowledge_emb
        self.word_emb=word_emb
        self.model = GCNClassifier(opt,knowledge_emb=knowledge_emb,word_emb=word_emb)
        #print(self.model)
        self.criterion = nn.BCEWithLogitsLoss()

        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels = unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits= self.model(inputs)
        #loss = self.criterion(logits, labels )
        loss = self.criterion(logits.squeeze(dim=-1), labels)
        #if loss>10.0:
            #print(loss,logits,labels)
            #print(inputs)
        # l2 decay on all conv layers
        #print(self.opt.get('conv_l2', 0) )
        '''
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        '''
        loss_val = loss.item()
        # backward
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch,unsort=True):
        inputs, labels= unpack_batch(batch, self.opt['cuda'])
        #orig_idx = batch[8]
        # forward
        self.model.eval()

        logits = self.model(inputs)

        loss = self.criterion(logits.squeeze(dim=-1), labels)
        #print(loss.item())
        #probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        probs = torch.sigmoid(logits).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        '''
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        '''
        return predictions, probs, loss.item()
