import random
import torch
import numpy as np
import gzip
import pickle as pkl
from utils import constant, helper, vocab
import pandas as pd

from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

class DataLoader(object):
    def __init__(self, filename, batch_size, opt,dic_file, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.mesh_dic,self.word_dic=dic_file

        # self.vocab = vocab
        self.eval = evaluation
        self.tokenizer = AutoTokenizer.from_pretrained('./bert_model/bert_fine')
        # self.label2id = constant.LABEL_TO_ID

        f_tt = gzip.open(filename, 'rb')
        '''
        labels_vec = pkl.load(f_tt)
        all_words_index = pkl.load(f_tt)
        all_words_token = pkl.load(f_tt)
        all_pos_s = pkl.load(f_tt)
        all_dis1 = pkl.load(f_tt)
        all_dis2 = pkl.load(f_tt)
        dep_gcn = pkl.load(f_tt)
        dep_shortest_gcn = pkl.load(f_tt)
        entity = pkl.load(f_tt)
        '''
        all_label_orig = pkl.load(f_tt)
        all_token_orig = pkl.load(f_tt)
        all_pos_orig = pkl.load(f_tt)
        all_dep_orig = pkl.load(f_tt)
        all_offset1_orig = pkl.load(f_tt)
        all_offset2_orig = pkl.load(f_tt)
        all_two_mesh_orig = pkl.load(f_tt)
        all_two_emention_orig = pkl.load(f_tt)
        all_two_etype_orig = pkl.load(f_tt)
        all_doc_id_orig = pkl.load(f_tt)
        all_sentence_dis_orig = pkl.load(f_tt)
        f_tt.close()
        # print(len(all_doc_id_orig))
        # print(all_doc_id_orig[0])
        # print(len(all_two_mesh_orig))
        # print(all_two_mesh_orig[0])
        '''
        all_label_orig_new = []
        all_token_orig_new =  []
        all_pos_orig_new =  []
        all_dep_orig_new =  []
        all_offset1_orig_new =  []
        all_offset2_orig_new = []
        all_two_mesh_orig_new =  []
        all_two_emention_orig_new = []
        all_two_etype_orig_new =  []
        all_doc_id_orig_new =  []
        all_sentence_dis_orig_new=[]

        all_label_orig_new.extend(all_label_orig)
        all_token_orig_new.extend(all_token_orig)
        all_pos_orig_new.extend(all_pos_orig)
        all_dep_orig_new.extend(all_dep_orig)
        all_offset1_orig_new.extend(all_offset1_orig)
        all_offset2_orig_new.extend(all_offset2_orig)
        all_two_mesh_orig_new.extend(all_two_mesh_orig)
        all_two_emention_orig_new.extend(all_two_emention_orig)
        all_two_etype_orig_new.extend(all_two_etype_orig)
        all_doc_id_orig_new.extend(all_doc_id_orig)
        all_sentence_dis_orig_new.extend(all_sentence_dis_orig)

        f_test = gzip.open('./all_data.pkl.gz', 'wb')
        pkl.dump(all_label_orig_new, f_test, -1)
        pkl.dump(all_token_orig_new, f_test, -1)
        pkl.dump(all_pos_orig_new, f_test, -1)
        pkl.dump(all_dep_orig_new, f_test, -1)
        pkl.dump(all_offset1_orig_new, f_test, -1)
        pkl.dump(all_offset2_orig_new, f_test, -1)
        pkl.dump(all_two_mesh_orig_new, f_test, -1)
        pkl.dump(all_two_emention_orig_new, f_test, -1)
        pkl.dump(all_two_etype_orig_new, f_test, -1)
        pkl.dump(all_doc_id_orig_new, f_test, -1)
        pkl.dump(all_sentence_dis_orig_new, f_test, -1)
        f_test.close()
        '''

        #print(all_pos_s)
        #print(constant.POS_DDI_TO_ID)
        #print(all_token_orig[4580:4581])
        #print(all_dep_orig[4580:4581])

        no_dep_list = []
        for i in range(len(all_dep_orig)):
            # stanford得到的依存解析树结果，(依存边的类型，父节点，子节点（按顺序排列的）)
            # 所以如果第二个数不是按顺序排列的，得到的解析树是有问题的，把该实例移除。。。
            temp_dep_list = [instance[2] for instance in all_dep_orig[i]]
            temp_dep_index = np.arange(len(all_dep_orig[i]))
            for j in range(len(temp_dep_index)):
                if temp_dep_list[j] != temp_dep_index[j] + 1:
                    no_dep_list.append(i)
                    break

        all_dep_list=list(np.arange(len(all_dep_orig)))

        keep_dep_list=list(set(all_dep_list).difference(set(no_dep_list)))

        print("remove nodep num:",len(no_dep_list))


        data_zip_orig = list(
            zip(all_label_orig, all_token_orig, all_pos_orig,
                all_dep_orig, all_offset1_orig,
                all_offset2_orig, all_two_mesh_orig, all_two_emention_orig,
                all_two_etype_orig, all_doc_id_orig,
                all_sentence_dis_orig))

        data_zip_orig = [data_zip_orig[i] for i in keep_dep_list]

        data_zip_orig_list  =list(zip(*data_zip_orig))


        all_label_orig=data_zip_orig_list[0]
        all_token_orig = data_zip_orig_list[1]
        all_pos_orig = data_zip_orig_list[2]
        all_dep_orig = data_zip_orig_list[3]
        all_offset1_orig = data_zip_orig_list[4]
        all_offset2_orig = data_zip_orig_list[5]
        all_two_mesh_orig = data_zip_orig_list[6]
        all_two_emention_orig = data_zip_orig_list[7]
        all_two_etype_orig = data_zip_orig_list[8]
        all_doc_id_orig = data_zip_orig_list[9]
        all_sentence_dis_orig = data_zip_orig_list[10]



        all_token_id_orig= [[self.word_dic[token] for token in instance] for instance in all_token_orig]
        #bert
        all_token_id_bert = []
        text_a = [' '.join(instance) for instance in all_token_orig]
        for text_a_i in text_a:
            token_id_bert = self.tokenizer.encode_plus(text_a_i, max_length=256, pad_to_max_length=True,truncation=True)
            all_token_id_bert.append(token_id_bert)
        all_input_ids = torch.tensor([f.input_ids for f in all_token_id_bert], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in all_token_id_bert], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in all_token_id_bert], dtype=torch.long)

        #print(all_token_orig)
        all_offset1_one,all_offset2_one=[],[]


        for i in range (len(all_offset1_orig)):

            if len(all_offset1_orig[i][0])==1 and len(all_offset2_orig[i][0])==1:
                all_offset1_one.append((all_offset1_orig[i][0],all_offset1_orig[i][1]))
                all_offset2_one.append((all_offset2_orig[i][0], all_offset2_orig[i][1]))
            else:
                temp_min=500
                temp_j=500
                temp_k=500
                for j in range (len(all_offset1_orig[i][0])):
                    for k in range (len(all_offset2_orig[i][0])):
                        if max(all_offset1_orig[i][0][j],all_offset2_orig[i][0][k])-min(all_offset1_orig[i][0][j],all_offset2_orig[i][0][k])<temp_min:
                            temp_min=max(all_offset1_orig[i][0][j],all_offset2_orig[i][0][k])-min(all_offset1_orig[i][0][j],all_offset2_orig[i][0][k])
                            temp_j=j
                            temp_k=k

                all_offset1_one.append(([all_offset1_orig[i][0][temp_j]], [all_offset1_orig[i][1][temp_j]]))
                all_offset2_one.append(([all_offset2_orig[i][0][temp_k]], [all_offset2_orig[i][1][temp_k]]))


        subj_positions, obj_positions = [], []
        for i in range(len(all_offset1_one)):
            #print(all_offset1_orig[i][0][0])
            #print(all_offset1_orig[i][1][0]-1)
            #print(all_offset1_orig[i][0][0],all_offset1_orig[i][1][0]-1, len(all_token_id_orig[i]))
            #print(all_offset2_orig[i][0][0], all_offset2_orig[i][1][0] - 1, len(all_token_id_orig[i]))
            subj_positions.append(get_positions(all_offset1_orig[i][0][0],all_offset1_orig[i][1][0]-1, len(all_token_id_orig[i])))
            obj_positions.append(
                get_positions(all_offset2_orig[i][0][0], all_offset2_orig[i][1][0] - 1, len(all_token_id_orig[i])))


        #print(all_offset1_orig)
        #print(all_offset2_orig)

        #print(all_offset1_one)
        #print(all_offset2_one)

        all_offset1_orig=all_offset1_one
        all_offset2_orig= all_offset2_one
        #print(all_offset1_orig)
        #print(all_offset2_orig)

        label_map = constant.LABEL_CDR_TO_ID
        all_label_narry = np.array(
            [label_map[instance] if instance in label_map else constant.UNK_ID for instance in all_label_orig])
        #print(all_label_narry)
        pos_map=constant.POS_DDI_TO_ID
        all_pos = [[pos_map[each_pos[1]] if each_pos[1] in pos_map else constant.UNK_ID for each_pos in instance] for instance in all_pos_orig]


        all_two_mesh_index = [ [self.mesh_dic[instance[0]],self.mesh_dic[instance[1]]] for instance in all_two_mesh_orig]



        #print(all_two_mesh_index)
        # all_offset1_orig

        #temp_offset

        all_subj_mask,all_obj_mask=[],[]
        all_dis1,all_dis2=[],[]
        for i in range(len(all_pos)):
            temp_subj,temp_obj=[1]*len(all_pos[i]),[1]*len(all_pos[i])

            assert len(all_offset1_orig[i][0])==len(all_offset1_orig[i][1])
            for j in range(len(all_offset1_orig[i][0])):
                temp_subj[all_offset1_orig[i][0][j]:all_offset1_orig[i][1][j]]=[0]*(all_offset1_orig[i][1][j]-all_offset1_orig[i][0][j])

            all_subj_mask.append(temp_subj)
            for j in range(len(all_offset2_orig[i][0])):
                temp_obj[all_offset2_orig[i][0][j]:all_offset2_orig[i][1][j]]=[0]*(all_offset2_orig[i][1][j]-all_offset2_orig[i][0][j])

            all_obj_mask.append(temp_obj)

            temp_dis1, temp_dis2 =  [i for i in range(0,len(all_pos[i]))],[i for i in range(0,len(all_pos[i]))]

            #print(all_offset1_orig[i][0][0])
            #print(all_offset1_orig[i][1][0])

            temp_dis1[all_offset1_orig[i][0][0]:all_offset1_orig[i][1][0]]=[0]*(all_offset1_orig[i][1][0]-all_offset1_orig[i][0][0])
            temp_dis2[all_offset2_orig[i][0][0]:all_offset2_orig[i][1][0]] = [0] * (
                        all_offset2_orig[i][1][0] - all_offset2_orig[i][0][0])

            #print(temp_dis1)
            #print(temp_dis2)

            for j in range(len(temp_dis1)):
                if j !=all_offset1_orig[i][0][0]:
                    temp_dis1[j]=temp_dis1[j]-all_offset1_orig[i][0][0]

            for j in range(len(temp_dis2)):
                if j !=all_offset2_orig[i][0][0]:
                    temp_dis2[j]=temp_dis2[j]-all_offset2_orig[i][0][0]
            #print(temp_dis1)
            #print(temp_dis2)

            temp_dis1_new=[instance+opt['dis_range'] for instance in temp_dis1]
            temp_dis2_new = [instance + opt['dis_range'] for instance in temp_dis2]
            #print(temp_dis1_new)
            #print(temp_dis2_new)
            all_dis1.append(temp_dis1_new)
            all_dis2.append(temp_dis2_new)
            #print(all_dis1)
            #print(all_dis2)

        all_head=[[each_dep[1] for each_dep in instance] for instance in all_dep_orig]

        #print(all_dis1)
        #print(all_dis2)


        '''
        pos_dic={}
        pos_index=0
        for instance in all_pos_orig:
            for each_pos in instance:
                if each_pos[1] not in pos_dic:
                    pos_dic[each_pos[1]]=pos_index
                    pos_index+=1
        print(pos_dic)
        '''
        #引入CTD标签
        if filename == './processed_data/outputTraining_Development.pkl.gz':
            # print('ctd_label_train saving')
            # ctd_label_train = list(map(str, all_ctd_label))
            # with open('./CTD/ctd_label_train.txt', 'w') as f_ctd:
            #     for i in range(len(ctd_label_train)):
            #         f_ctd.write(ctd_label_train[i] + '\n')
            with open('./CTD/ctd_label_train.txt', 'r') as f_ctd:
                all_ctd_label = f_ctd.readlines()
            for i in range(len(all_ctd_label)):
                all_ctd_label[i] = int(all_ctd_label[i].strip())

        if filename == './processed_data/outputTest_min.pkl.gz':
            # print('ctd_label_test saving')
            # ctd_label_test = list(map(str, all_ctd_label))
            # with open('./CTD/ctd_label_test.txt', 'w') as f_ctd:
            #     for i in range(len(ctd_label_test)):
            #         f_ctd.write(ctd_label_test[i] + '\n')
            with open('./CTD/ctd_label_test.txt', 'r') as f_ctd:
                all_ctd_label = f_ctd.readlines()
            for i in range(len(all_ctd_label)):
                all_ctd_label[i] = int(all_ctd_label[i].strip())


        #lables=np.argmax(labels_vec,1)

        data_zip = list(zip(all_label_narry, all_token_orig,all_token_id_orig,all_pos, all_head,
                            all_subj_mask, all_obj_mask,all_doc_id_orig,all_two_mesh_orig,all_dis1,all_dis2,all_two_mesh_index,subj_positions,obj_positions,
                            all_two_emention_orig, all_sentence_dis_orig, all_ctd_label, all_input_ids, all_attention_mask, all_token_type_ids))

        '''
        data_zip = list(zip(all_label_narry[1: 3], all_token_orig[1: 3],all_token_id_orig[1: 3], all_pos[1: 3], all_head[1: 3],
                            all_subj_mask[1: 3], all_obj_mask[1: 3], all_doc_id_orig[1: 3], all_two_mesh_orig[1: 3], all_dis1[1: 3], all_dis2[1: 3],
                            all_two_mesh_index[1: 3],subj_positions[1: 3],obj_positions[1: 3]))
        '''


        '''
        data_zip = list(
            zip(all_label_narry[4580: 4582], all_token_orig[4580: 4582], all_token_id_orig[4580: 4582], all_pos[4580: 4582], all_head[4580: 4582],
                all_subj_mask[4580: 4582], all_obj_mask[4580: 4582], all_doc_id_orig[4580: 4582], all_two_mesh_orig[4580: 4582], all_dis1[4580: 4582],
                all_dis2[4580: 4582],
                all_two_mesh_index[4580: 4582], subj_positions[4580: 4582], obj_positions[4580: 4582]))
        '''
        #print(all_words_index)
        #print(all_pos)
        #self.raw_data = data_zip
        #print(data)
        #data = self.preprocess(data_zip,  opt)

        # shuffle for training

        if not evaluation:
            indices = list(range(len(data_zip)))
            random.shuffle(indices)
            data_zip = [data_zip[i] for i in indices]

        #self.id2label = dict([(v, k) for k, v in self.label2id.items()])
        #self.labels = [self.id2label[d[-1]] for d in data]

        self.num_examples = len(data_zip)
        # chunk into batches
        data_zip = [data_zip[i:i + batch_size] for i in range(0, len(data_zip), batch_size)]
        #data_zip = [data_zip[i:i + batch_size] for i in range(0, 20, batch_size)]
        self.data = data_zip
        #print(self.data )
        print("{} batches created for {}".format(len(data_zip), filename))
    '''
    def preprocess(self, data,  opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            processed += [
                (tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels
    '''
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        #print(len(batch))
        """
        all_label_narry, all_token_orig, all_token_id_orig, all_pos, \
        all_head, all_subj_mask, all_obj_mask, all_doc_id_orig, \
        all_two_mesh_orig, all_dis1, all_dis2, all_two_mesh_index, \
        subj_positions, obj_positions, all_two_emention_orig, all_sentence_dis_orig
        """
        assert len(batch) == 20

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[1]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        '''
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[1]]
        else:
            words = batch[1]
        '''

        # convert to tensors

        #words = get_long_tensor(words, batch_size)
        #masks = torch.eq(words, 0)

        tokens=batch[1]
        tokens_elmoid=batch_to_ids(tokens)
        token_id=get_long_tensor(batch[2], batch_size)

        #masks = torch.zeros_like(batch[3], dtype=torch.int)
        pos = get_long_tensor(batch[3], batch_size)

        masks=torch.eq(pos,0)
        #print("llll",pos,masks)
        #ner = get_long_tensor(batch[2], batch_size)
        #deprel = get_long_tensor(batch[3], batch_size)
        #head = get_long_tensor(batch[4], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_mask = get_long_tensor(batch[5], batch_size)
        obj_mask = get_long_tensor(batch[6], batch_size)

        dis1 = get_long_tensor(batch[9], batch_size)
        dis2 = get_long_tensor(batch[10], batch_size)

        rels = torch.FloatTensor(batch[0])
        #rels = torch.LongTensor(batch[0])
        all_two_mesh_index=torch.LongTensor(batch[11])

        subj_positions=get_long_tensor(batch[12], batch_size)
        obj_positions=get_long_tensor(batch[13], batch_size)

        ctd_labels = torch.IntTensor(batch[16])
        input_ids = get_long_tensor(batch[17], batch_size)
        attention_mask = get_long_tensor(batch[18], batch_size)
        token_type_ids = get_long_tensor(batch[19], batch_size)
        #print(rels)
        #dep_shortest_gcn=batch[8]
        #dep_shortest_gcn=torch.Tensor(dep_shortest_gcn)
        # print(all_two_mesh_index[0][0].item())
        return (
            tokens_elmoid, masks, pos, head,  subj_mask, obj_mask, dis1,dis2,all_two_mesh_index,token_id,subj_positions,
            obj_positions, ctd_labels, input_ids, attention_mask, token_type_ids, rels, orig_idx,batch[0],batch[7],batch[8],batch[14],batch[15])

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
           list(range(1, length - end_idx))


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
                else x for x in tokens]
