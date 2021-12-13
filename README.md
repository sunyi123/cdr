# cdr
model for BioCreative V Track 3 CDR extraction

## Necessary documents
* finetuned bert-model: [bert-fine](https://pan.baidu.com/s/1qZU6B7Z1KIT03p_y1EZTVg) access code: ejgk
* elmo embedding: [elmo-embedding](https://pan.baidu.com/s/1_8q9dlCfquBwGm_-1nh2gw) access code: e4kp
* knowledge embedding: You should train them on CTD with TransE

## Key Code
* data/loader_cid_elmo.py: The code for preprocessing the input
* model/gcn_lstm_elmo_cdr.py: Our model framework code
* model/multihead_att.py: The code for the multi-head attention mechanism
* model/trainer.py: The code for training and prediction

## contact to me

* isakyi@dlut.mail.edu.cn
