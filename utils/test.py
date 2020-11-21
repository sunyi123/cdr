import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import gzip
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils  import evaluate

from data.loader_cid_elmo import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from gensim.models import KeyedVectors,FastText

parser = argparse.ArgumentParser()
parser.add_argument('--emb_dim', type=int, default=200, help='Word embedding dimension.')
#parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=100, help='POS embedding dimension.')
parser.add_argument('--dis_dim', type=int, default=100, help='dis position embedding dimension.')
parser.add_argument('--elmo_dim', type=int, default=1024, help='elmo embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=500, help='GCN and MLP hidden state size.')
parser.add_argument('--use_elmo', type=bool, default=True, help='Use elmo or not')

parser.add_argument('--dis_range', type=int, default=200, help='dis position range.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--prune_k', default=0, type=int, help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=1, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false',help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=500, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')
parser.add_argument('--mlp_dropout', type=float, default=0.5, help='mlp dropout rate.')


parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax','RMSprop'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=20, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
#parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=20, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load',  default=True, help='Load pretrained model.')
parser.add_argument('--model_file', type=str, default='./saved_models/00/best_model.pt',help='Filename of the pretrained model.')
#parser.add_argument('-developmnetfile', dest="developmnetfile", default='./dataset/cdr/outputDevelopment_min.pkl.gz',  help='development file')
#parser.add_argument('-trainfile', dest="trainfile", default='./dataset/cdr/outputTraining_min.pkl.gz', help='train file')

parser.add_argument('-trainfile', dest="trainfile", default='./processed_data/outputTraining_Development.pkl.gz', help='train file')
parser.add_argument('-testfile', dest="testfile", default='./processed_data/outputTest_min.pkl.gz', help='test file')
parser.add_argument('-num_class', dest="num_class",type=int, default=1, help='num class')
parser.add_argument('-elmooptionsfile', dest="elmooptionsfile", default='./dataset/elmo/elmo_2x4096_512_2048cnn_2xhighway_options_pubMed.json',  help='elmo_options_file')
parser.add_argument('-elmoweightfile', dest="elmoweightfile", default='./dataset/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5',  help='elmo_weight_file')
parser.add_argument('-position_num', dest="position_num",  default=6,  help='entity position num')
parser.add_argument('-meshembedding_dim', dest="meshembedding_dim",  default=200,  help='meshembedding_dim')
parser.add_argument('-wordembedding_dim', dest="wordembedding_dim",  default=200,  help='wordembedding_dim')

parser.add_argument('-meshembedding_file', dest="meshembedding_file", default='./dataset/mesh/mesh_5.txt', help='meshembedding')
parser.add_argument('-meshdic_file', dest="meshdic_file", default='./mesh_dic.pkl.gz', help='meshdic_file')
parser.add_argument('-worddic_file', dest="worddic_file", default='./word_dic.pkl.gz', help='worddic_file')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)

#print(args.cpu)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()

# make opt
opt = vars(args)
'''
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']
'''
f_tt = gzip.open(opt['meshdic_file'], 'rb')

mesh_id_dic = pkl.load(f_tt)
f_tt.close()


nb_words = len(mesh_id_dic) + 1
meshvec_table = np.zeros((nb_words, opt['meshembedding_dim']))

mesh2vec = KeyedVectors.load_word2vec_format(opt['meshembedding_file'],
                                              binary=False)
print('Found %s word vectors of word2vec' % len(mesh2vec.vocab))
mesh_miss_num = 0
for word_instance, word_index in mesh_id_dic.items():
    if word_instance in mesh2vec.vocab:
        meshvec_table[word_index] = mesh2vec.word_vec(word_instance)
    else:
        mesh_miss_num += 1

#print (mesh_miss_num)

opt['meshembedding_size']=len(meshvec_table)

'''
f_tt = gzip.open(opt['worddic_file'], 'rb')

word_id_dic=word_dic
nb_words = len(word_id_dic)
wordvec_table = np.zeros((nb_words, 200))

word2vec = KeyedVectors.load_word2vec_format('F:/zyj/gcn-over-pruned-trees_cdrtest/dataset/cdr/PMC-w2v.bin',     binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))
word_miss_num = 0
for word_instance, word_index in word_id_dic.items():
    if word_instance in word2vec.vocab:
        wordvec_table[word_index] = word2vec.word_vec(word_instance)
    else:
        word_miss_num += 1

print (word_miss_num)

'''

f_tt = gzip.open('./wordvec_table.pkl.gz', 'rb')


wordvec_table = pkl.load(f_tt)
f_tt.close()

opt['pubembedding_size']=len(wordvec_table)

#加载词典，用于将单词转化为id，然后映射成向量
f_tt = gzip.open(opt['worddic_file'], 'rb')
word_id_dic=pkl.load(f_tt)
f_tt.close()

dic_file=(mesh_id_dic,word_id_dic)

# load data
print("Loading data from {} with batch size {}...".format(opt['trainfile']+' and '+opt['testfile'], opt['batch_size']))
train_batch = DataLoader(opt['trainfile'], opt['batch_size'], opt,dic_file,evaluation=False)
#developmnet_batch = DataLoader(opt['developmnetfile'], opt['batch_size'], opt,dic_file,  evaluation=True)
test_batch = DataLoader(opt['testfile'], opt['batch_size'], opt, dic_file, evaluation=True)



'''
opt['vocab_size'] = wordvec.shape[0]
print(opt['vocab_size'])
'''
#模型保存的地址
#model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
#model_save_dir = opt['save_dir'] + '/' + model_id
#opt['model_save_dir'] = model_save_dir
#helper.ensure_dir(model_save_dir, verbose=True)

# save config
#helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
#vocab.save(model_save_dir + '/vocab.pkl')
#file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_precision\tdev recall\tdev f1\tbest f1")

# print model info
#helper.print_config(opt)
#print(np.array(meshvec_table))
# model
if not opt['load']:
    trainer = GCNTrainer(opt,knowledge_emb=np.array(meshvec_table),word_emb=np.array(wordvec_table))
    #print (trainer)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = GCNTrainer(model_opt)
    trainer.load(model_file)


    print("Evaluating on dev set...")
    gold_answer=[]
    predictions = []
    doc_id_list = []
    mesh_id_list = []
    distance_list = []
    dev_loss = 0
    for i, batch in enumerate(test_batch):
        """
        19
        tokens_elmoid, masks, pos, head,  
        subj_mask, obj_mask, dis1, dis2,
        all_two_mesh_index, token_id, subj_positions, obj_positions,
        rels, orig_idx, batch[0], batch[7], 
        batch[8], batch[14], batch[15]
        """
        label=batch[12]
        doc_id = batch[15]
        two_mesh_id = batch[16]
        # 0表示句内，>0都是跨句子的，现在没有按论文中写的预处理规则进行过滤，所以有很多跨句子数目大于3
        sentence_distance = batch[18]

        gold_answer.extend(label)
        doc_id_list += doc_id
        mesh_id_list += two_mesh_id
        distance_list += sentence_distance

        preds, probs, loss = trainer.predict(batch)


        predictions += probs
        dev_loss += loss

    predictions_list=[instance[0] for instance in predictions]

    intra_predictions = []
    intra_answer = []
    inter_predictions = []
    inter_answer = []

    for i in range(len(gold_answer)):
        if distance_list[i] == 0:
            intra_predictions.append(predictions_list[i])
            intra_answer.append(gold_answer[i])
        else:
            inter_predictions.append(predictions_list[i])
            inter_answer.append(gold_answer[i])

    f_auc, dev_f, dev_p, dev_r, maxac, maxj = evaluate.f_compute(gold_answer,predictions_list)
    _, intra_dev_f, intra_dev_p, intra_dev_r, _, _ = evaluate.f_compute(intra_answer, intra_predictions)
    _, inter_dev_f, inter_dev_p, inter_dev_r, _, _ = evaluate.f_compute(inter_answer, inter_predictions)

    f_test_pred = open('./output_test_prediction.txt', "w")
    for i in range(len(predictions_list)):
        if predictions_list[i] >= maxj:
            f_test_pred.write('\t'.join([doc_id_list[i], 'CID', mesh_id_list[i][0], mesh_id_list[i][1], str(distance_list[i])]) + '\n')
    f_test_pred.close()

    print('Test results: Prec:{} | Rec:{} | F1:{} '.format(dev_p, dev_r, dev_f))
    print('Test results for intra-sentence level: Prec:{} | Rec:{} | F1:{} '.format(intra_dev_p, intra_dev_r, intra_dev_f))
    print('Test results for inter-sentence level: Prec:{} | Rec:{} | F1:{} '.format(inter_dev_p, inter_dev_r, inter_dev_f))

    test_result_file = model_save_dir + '/' + 'test_result.txt'
    test_file_logger = helper.FileLogger(test_result_file)
    test_file_logger.log('Test results: Prec:{} | Rec:{} | F1:{} '.format(dev_p, dev_r, dev_f))
    test_file_logger.log('Test results for intra-sentence level: Prec:{} | Rec:{} | F1:{} '.format(intra_dev_p, intra_dev_r, intra_dev_f))
    test_file_logger.log('Test results for inter-sentence level: Prec:{} | Rec:{} | F1:{} '.format(inter_dev_p, inter_dev_r, inter_dev_f))
    print('Test end.....')



"""
#id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

best_F1=0.
print("There are %d train instances!!!" % train_batch.num_examples)
print("There are %d test instances!!!" % test_batch.num_examples)
# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            # 打印的是当前batch的loss
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    gold_answer=[]
    gold_answer_int = []
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(test_batch):

        label = batch[12]
        label_bak = batch[14]


        gold_answer.extend(label)
        gold_answer_int.extend(label_bak)

        preds, probs, loss = trainer.predict(batch)

        predictions += probs
        dev_loss += loss
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / test_batch.num_examples * opt['batch_size']

    predictions_list=[instance[0] for instance in predictions]

    f_auc, dev_f, dev_p, dev_r, maxac, maxj = evaluate.f_compute(gold_answer,predictions_list)

    print('the classification threshold of epoch %d is %f' % (epoch, maxj))

    #gold_one_hot = torch.zeros(len(gold_answer), opt['num_class']).scatter_(1, label, 1)
    ''''
    gold_one_hot= [[0,1] if instance ==1 else [1,0] for instance in gold_answer_int]
    print(gold_one_hot)
    dev_p, dev_r, dev_f= binary_result_evaluation(gold_one_hot, predictions)
    '''
    print('Test results: Prec:{} | Rec:{} | F1:{} '.format(dev_p, dev_r, dev_f))


    #dev_p, dev_r, dev_f1 = scorer.score(test_batch.gold(), predictions)
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
        train_loss, dev_loss, dev_f))
    dev_score = dev_f
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_p, dev_r, dev_f, max([dev_score] + dev_score_history)))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file)
    if epoch == 1 or dev_score > max(dev_score_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
            .format(epoch, dev_p*100, dev_r*100, dev_score*100))
    # 保存当前epoch的模型，上面是保存最好的模型，目前是保存了最后一个模型
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))
"""
