"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils
from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertModel, RobertaModel
from model import multihead_att

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt,knowledge_emb=None,word_emb=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt,knowledge_emb=knowledge_emb,word_emb=word_emb)
        in_dim=opt['hidden_dim']  #这个是MLP层输出的维度
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs= self.gcn_model(inputs)
        #print(outputs)
        logits = self.classifier(outputs)
        #print(logits)
        return logits

class GCNRelationModel(nn.Module):
    def __init__(self, opt,knowledge_emb=None,word_emb=None):
        super().__init__()
        self.opt = opt
        self.knowledge_emb = knowledge_emb
        self.word_emb = word_emb
        # create embedding layers
        self.mesh_emb = nn.Embedding(opt['meshembedding_size'], opt['meshembedding_dim'])
        self.word_emb_model = nn.Embedding(opt['pubembedding_size'], opt['wordembedding_dim'])

        #print(self.mesh_emb)
        if self.knowledge_emb is None:
            self.mesh_emb.weight.data[:, :].uniform_(-1.0, 1.0)
        else:
            self.knowledge_emb = torch.from_numpy(self.knowledge_emb)
            self.mesh_emb.weight.data.copy_(self.knowledge_emb)

        if self.word_emb is None:
            # 随机初始化 词向量矩阵 可以用[-0.25, 0.25]
            self.word_emb_model.weight.data[:, :].uniform_(-1.0, 1.0)
        else:
            self.word_emb = torch.from_numpy(self.word_emb)

            self.word_emb_model.weight.data.copy_(self.word_emb)



        #print(self.mesh_emb)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.dis_emb = nn.Embedding(opt['dis_range']*2+1, opt['dis_dim']) if opt['dis_dim'] > 0 else None
        embeddings = ( self.mesh_emb,self.word_emb_model,self.pos_emb, self.dis_emb)
        #self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        #消融实验，对比实验修改的地方

        in_dim = opt['rnn_hidden'] * 2 + opt['hidden_dim'] * 3
        #in_dim = opt['rnn_hidden'] * 2 + opt['hidden_dim'] * 3
        #in_dim = 500 + opt['hidden_dim'] * 3
        # in_dim = 500

        # 消融实验，去掉多层感知机：设置mlp_layers = 1即可
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        self.mlp_drop = nn.Dropout(opt['mlp_dropout'])

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        tokens_elmoid, masks, pos, head, subj_mask, obj_mask, dis1, dis2, all_two_mesh_index, token_id, subj_positions, obj_positions, ctd_label, input_ids, attention_mask, token_type_ids= inputs
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, token_id, l, prune, subj_positions, obj_positions):
            trees = [head_to_tree(head[i], token_id[i], l[i], prune, subj_positions[i], obj_positions[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in
                   trees]
            new_adj=[]
            for instance in adj:

                target_zero_array = [np.zeros((len(instance[0][0]), len(instance[0][0])), dtype=np.float32)]
                if (instance==target_zero_array).all():
                    new_adj.append([np.eye(len(instance[0][0]), k=1, dtype=np.float32)+np.eye(len(instance[0][0]), k=-1, dtype=np.float32)])
                else:
                    new_adj.append(instance)
            adj = np.concatenate(new_adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda() if self.opt['cuda'] else adj

        adj = inputs_to_tree_reps(head.data, token_id.data, l, self.opt['prune_k'], subj_positions.data, obj_positions.data)

        h_rnn = self.gcn(adj, inputs)
        
        # pooling

        h_rnn=self.out_mlp(h_rnn)

        h_rnn=self.mlp_drop(h_rnn)
        #return outputs, h_out
        return h_rnn

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.heads = opt['heads']
        self.d_model = opt['rnn_hidden'] * 2
        self.d_k = int(self.d_model / self.heads)
        # 做输入特征（ELMo（word2vec) + POS + position）对比实验时需要修改
        #self.in_dim = opt['wordembedding_dim']
        #self.in_dim = opt['wordembedding_dim'] + opt['pos_dim']
        #self.in_dim = opt['wordembedding_dim'] + opt['pos_dim'] + opt['dis_dim'] * 2
        #self.in_dim = opt['elmo_dim']
        #self.in_dim = opt['elmo_dim'] + opt['pos_dim']
        #self.in_dim = opt['elmo_dim'] + opt['pos_dim'] + opt['dis_dim'] * 2
        if opt['use_bert'] and opt['pos_dim'] > 0 and opt['dis_dim'] > 0:
            self.bert_in_dim = opt['bert_dim'] + opt['pos_dim'] + opt['dis_dim'] * 2 + opt['meshembedding_dim']
        elif opt['use_bert'] and opt['pos_dim'] > 0 and opt['dis_dim'] < 0:
            self.bert_in_dim = opt['bert_dim'] + opt['pos_dim'] + opt['meshembedding_dim']
        elif opt['use_bert'] and opt['pos_dim'] < 0 and opt['dis_dim'] > 0:
            self.bert_in_dim = opt['bert_dim'] + opt['dis_dim'] * 2 + opt['meshembedding_dim']
        elif opt['use_bert'] and opt['pos_dim'] < 0 and opt['dis_dim'] < 0:
            self.bert_in_dim = opt['bert_dim'] + opt['meshembedding_dim']
        if opt['use_elmo'] and opt['pos_dim'] > 0 and opt['dis_dim'] > 0:
            self.elmo_in_dim = opt['elmo_dim'] + opt['pos_dim'] + opt['dis_dim'] * 2 + opt['meshembedding_dim']
        elif opt['use_elmo'] and opt['pos_dim'] > 0 and opt['dis_dim'] < 0:
            self.elmo_in_dim = opt['elmo_dim'] + opt['pos_dim'] + opt['meshembedding_dim']
        elif opt['use_elmo'] and opt['pos_dim'] < 0 and opt['dis_dim'] > 0:
            self.elmo_in_dim = opt['elmo_dim'] + opt['dis_dim'] * 2 + opt['meshembedding_dim']
        elif opt['use_elmo'] and opt['pos_dim'] < 0 and opt['dis_dim'] < 0:
            self.elmo_in_dim = opt['elmo_dim'] + opt['meshembedding_dim']
        if not opt['use_elmo'] and not opt['use_bert'] and opt['pos_dim'] > 0 and opt['dis_dim'] > 0:
            self.woed_in_dim = opt['wordembedding_dim'] + opt['pos_dim'] + opt['dis_dim'] * 2
        elif not opt['use_elmo'] and not opt['use_bert'] and opt['pos_dim'] > 0 and opt['dis_dim'] < 0:
            self.word_in_dim = opt['wordembedding_dim'] + opt['pos_dim']
        elif not opt['use_elmo'] and not opt['use_bert'] and opt['pos_dim'] < 0 and opt['dis_dim'] > 0:
            self.word_in_dim = opt['wordembedding_dim'] + opt['dis_dim'] * 2
        elif not opt['use_elmo'] and not opt['use_bert'] and opt['pos_dim'] < 0 and opt['dis_dim'] < 0:
            self.word_in_dim = opt['wordembedding_dim']

        self.elmo = Elmo(opt['elmooptionsfile'], opt['elmoweightfile'], 1)

        self.bert = RobertaModel.from_pretrained('./bert_model/bert_fine')

        self.slf_rnn_attn = multihead_att.MultiHeadAttention(
            self.heads, self.d_model, self.d_k, self.d_k, dropout=0.2)

        
        self.slf_gcn_attn = multihead_att.MultiHeadAttention(
            5, 500, 100, 100, dropout=0.2)

        self.slf_rel_attn = multihead_att.MultiHeadAttention(
            1, 500, 500, 500, dropout=0.2)

        


        self.mesh_emb, self.word_emb_model, self.pos_emb, self.dis_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            bert_input_size = self.bert_in_dim
            elmo_input_size = self.elmo_in_dim

            self.bert_rnn = nn.LSTM(bert_input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.elmo_rnn = nn.LSTM(elmo_input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            '''
            self.rnn = nn.GRU(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                               dropout=opt['rnn_dropout'], bidirectional=True)
            '''
            self.in_dim = opt['rnn_hidden'] * 2
        self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        self.rel_drop_rate = opt['relation_dropout']

        # gcn layer

        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))


    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def bert_encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'],use_cuda=self.use_cuda)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.bert_rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs,ht

    def elmo_encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'],use_cuda=self.use_cuda)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.elmo_rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs,ht

    def mask(self, relationvec, rate):
        mask_num = int(1000 * rate)
        mask_list = random.sample(range(1000), mask_num)
        for mask_index in mask_list:
            relationvec[mask_index] = 0.0
        return relationvec

    def forward(self, adj,inputs):
        tokens_elmoid, masks, pos, head, subj_mask, obj_mask, dis1, dis2, all_two_mesh_index, token_id, subj_positions, obj_positions, ctd_label, input_ids, attention_mask,token_type_ids = inputs # unpack

        batch_size = tokens_elmoid.size()[0]

        subj_mask, obj_mask = subj_mask.eq(0).eq(0).unsqueeze(2), obj_mask.eq(0).eq(0).unsqueeze(2)

        mesh_embs = self.mesh_emb(all_two_mesh_index)

        word_embs = self.word_emb_model(token_id)
        word_input = [word_embs]

        elmo_embs = self.elmo(tokens_elmoid)['elmo_representations'][0]
        elmo_input = [elmo_embs]

        bert_embs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_input = bert_embs[0]

        pos_len = pos.size()[1]
        if pos_len > len(bert_embs[0][0]):
            bert_padding = torch.zeros(batch_size, pos_len - len(bert_embs[0][0]), len(bert_embs[0][0])).cuda()
            bert_input = torch.cat([bert_embs, bert_padding], dim=1)

        bert_input = [bert_input]

        # 做特征有效性实验时， 设置 --pos_dim -1
        if self.opt['pos_dim'] > 0:
            pos_embedding = self.pos_emb(pos)
            word_input += [pos_embedding]
            elmo_input += [pos_embedding]

            if pos_len < len(bert_embs[0][0]):
                pos_padding = torch.zeros(batch_size, len(bert_embs[0][0]) - pos_len, pos_embedding.size()[2]).cuda()
                pos_embedding = torch.cat([pos_embedding, pos_padding], dim=1)
            bert_input += [pos_embedding]

        # 做特征有效性实验时， 设置 --dis_dim -1
        if self.opt['dis_dim'] > 0:
            dis1_embedding = self.dis_emb(dis1)
            dis2_embedding = self.dis_emb(dis2)
            word_input += [dis1_embedding]
            word_input += [dis2_embedding]

            elmo_input += [dis1_embedding]
            elmo_input += [dis2_embedding]

            if pos_len < len(bert_embs[0][0]):
                dis_padding = torch.zeros(batch_size, len(bert_embs[0][0]) - pos_len, dis1_embedding.size()[2]).cuda()
                dis1_embedding = torch.cat([dis1_embedding, dis_padding], dim=1)
                dis2_embedding = torch.cat([dis2_embedding, dis_padding], dim=1)
            bert_input += [dis1_embedding]
            bert_input += [dis2_embedding]

        # 加入实体向量
        entity_embs = torch.zeros(batch_size, pos.size()[1], 1024).cuda()
        for i in range(subj_positions.size()[0]):
            for j in range(subj_positions.size()[1]):
                if subj_positions[i][j] == 0:
                    entity_embs[i][j] = mesh_embs[i][0]
                    break
            for k in range(obj_positions.size()[1]):
                if obj_positions[i][k] == 0:
                    entity_embs[i][k] = mesh_embs[i][1]
                    break
            break

        elmo_input += [entity_embs]
        bert_entity_embs = torch.zeros(batch_size, pos_embedding.size()[1], 1024).cuda()
        for i in range(subj_positions.size()[0]):
            for j in range(subj_positions.size()[1]):
                if subj_positions[i][j] == 0:
                    bert_entity_embs[i][j] = mesh_embs[i][0]
                    break
            for k in range(obj_positions.size()[1]):
                if obj_positions[i][k] == 0:
                    bert_entity_embs[i][k] = mesh_embs[i][1]
                    break
            break
        bert_input += [bert_entity_embs]

        word_input = torch.cat(word_input, dim=2)
        elmo_input = torch.cat(elmo_input, dim=2)
        bert_input = torch.cat(bert_input, dim=2)

        # if self.opt['use_bert']:
        #     model_input = bert_input
        # elif self.opt['use_elmo']:
        #     model_input = elmo_input
        # else:
        #     model_input = word_input

        #model_input = self.in_drop(model_input)
        bert_input = self.in_drop(bert_input)
        elmo_input = self.in_drop(elmo_input)


        # rnn layer
        """
        if self.opt.get('rnn', False):
            rnn_output,ht = self.encode_with_rnn(embs, masks, tokens_elmoid.size()[0])
            rnn_output = self.rnn_drop(rnn_output)

        else:
            gcn_inputs = embs
        """
        if self.opt.get('rnn', False):
            bert_rnn_output, bert_ht = self.bert_encode_with_rnn(bert_input, masks, batch_size)
            bert_rnn_output = self.rnn_drop(bert_rnn_output)

            elmo_rnn_output, elmo_ht = self.elmo_encode_with_rnn(elmo_input, masks, batch_size)
            elmo_rnn_output = self.rnn_drop(elmo_rnn_output)

        else:
            gcn_inputs = elmo_input
        #gcn_inputs = embs
        #subj_mask, obj_mask = subj_mask.eq(0).eq(0).unsqueeze(2), obj_mask.eq(0).eq(0).unsqueeze(2)
        subj_rnn_out = pool(bert_rnn_output, subj_mask, type='max')
        obj_rnn_out = pool(bert_rnn_output, obj_mask, type='max')
        rnn_out = pool(bert_rnn_output, mask=None, type='max')


        # 加入关系向量
        with open('./dataset/mesh/relationVector_1000.txt', 'r') as f:
            relation = f.readlines()
        rows = len(relation)
        datamat = np.zeros((rows, 1000))
        row = 0
        for line in relation:
            line = line.strip().split(',')
            datamat[row, :] = line[::]
            row += 1
        relation2vec = torch.FloatTensor(datamat).cuda()
        relation_vec = torch.empty(batch_size, 1, 1000).cuda()
        for i in range(batch_size):
            relation_mask = self.mask(relation2vec[ctd_label[i]], self.rel_drop_rate)
            relation_vec[i][0] = relation_mask

        rel_att_out, attnself = self.slf_rnn_attn(bert_rnn_output, relation_vec, relation_vec, mask=None)
        rel_att_out_maxpool = pool(rel_att_out, mask=None, type='max')
        # rnn_att_out_avgpool = pool(rnn_att_out, mask=None, type='avg')
        # subj_att_rnn_out = pool(rnn_att_out, subj_mask, type='max')
        # obj_att_rnn_out = pool(rnn_att_out, obj_mask, type='max')
        

        denom = adj.sum(2).unsqueeze(2) + 1
        pool_mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        gcn_inputs = elmo_rnn_output
        #self.word_emb_model(token_id)
        #gcn_inputs = torch.cat([rnn_output,self.word_emb_model(token_id)], dim=2)
        #gcn_inputs = embs
        #gcn_inputs=word_embs
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            #gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
            gcn_inputs = self.gcn_drop(gAxW)

        #gcn_simple_out=gcn_inputs

        subj_gcn_out = pool(gcn_inputs, subj_mask, type='max')
        obj_gcn_out = pool(gcn_inputs, obj_mask, type='max')

        gcn_out_pool = pool(gcn_inputs, pool_mask, type='max')

        gcn_att_out, gcn_attnself = self.slf_gcn_attn(gcn_inputs, gcn_inputs, gcn_inputs, mask=None)
        #gcn_att_out=gcn_inputs
        gcn_att_out_pool = pool(gcn_att_out, pool_mask, type='max')
        subj_att_gcn_out = pool(gcn_att_out, subj_mask, type='max')
        obj_att_gcn_out = pool(gcn_att_out, obj_mask, type='max')
        #print(gcn_out,subj_gcn_out,obj_gcn_out)
        
        # 做消融实验时需要修改GCNRelationModel中的in_dim 73行

        # 总体模型框架：in_dim = opt['rnn_hidden'] * 2 + opt['hidden_dim'] * 3


        outputs = torch.cat([rel_att_out_maxpool, subj_gcn_out, obj_gcn_out, gcn_out_pool], dim=1)
        #outputs = torch.cat([rnn_att_out_maxpool, subj_gcn_out, obj_gcn_out, gcn_out_pool], dim=1)
        # 消融实验：-multi-head attention: in_dim = opt['hidden_dim'] * 3
        # outputs = rel_att_out
        # outputs = torch.cat([rnn_att_out_maxpool, subj_gcn_out, obj_gcn_out, gcn_out_pool, rel_att_out, ], dim=1)
        #outputs = torch.cat([subj_gcn_out, obj_gcn_out, gcn_out_pool], dim=1)
        #outputs = torch.cat([relation_vec, subj_att_gcn_out, obj_att_gcn_out, gcn_att_out_pool], dim=1)
        #outputs = torch.cat([relation_vec, gcn_att_out_pool], dim=1)

        # 消融实验: -GCN: 这个有很多种做法，挑一个结果最好的
        # in_dim = opt['rnn_hidden'] * 6
        #outputs = torch.cat([subj_att_rnn_out, obj_att_rnn_out, rnn_att_out_maxpool], dim=1)
        # in_dim = opt['rnn_hidden'] * 2
        #outputs = rnn_att_out_maxpool
        # in_dim = opt['rnn_hidden'] * 2 + opt['hidden_dim'] * 2
        #outputs = torch.cat([rnn_att_out_maxpool, subj_gcn_out, obj_gcn_out], dim=1)
        #outputs = torch.cat([rnn_att_out_maxpool, gcn_out_pool], dim=1)
        # in_dim = opt['rnn_hidden'] * 8
        #outputs = torch.cat([rnn_att_out_maxpool, rnn_out, subj_rnn_out, obj_rnn_out], dim=1)

        return outputs

def pool(h, mask, type='max'):
    if type == 'max':
        if mask is not None:
            h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        if mask is not None:
            h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        if mask is not None:
            h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)

    h0 = c0 = torch.zeros(*state_shape, requires_grad=False)

    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

