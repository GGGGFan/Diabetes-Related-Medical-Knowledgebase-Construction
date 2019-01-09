import os
import sys
import pandas as pd
import re
import numpy as np
import pickle
import codecs
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as D
from torch.autograd import Variable
import torch.nn.functional as F
from models.BiLSTM_ATT import BiLSTM_ATT
from models.PCNN_ATT import PCNN_ATT
from sklearn.model_selection import train_test_split
import lightgbm as lgb

char2id = {'unknown':0}  
relation2id = {"unknown":0,
            "Test_Disease":1,
            "Symptom_Disease":2,
            "Treatment_Disease":3,
            "Drug_Disease":4,
            "Anatomy_Disease":5,
            "Frequency_Drug":6,
            "Duration_Drug":7,
            "Amount_Drug":8,
            "Method_Drug":9,
            "SideEff_Drug":10}

id2relation = {0: "unknown",
            1: "Test_Disease",
            2: "Symptom_Disease",
            3: "Treatment_Disease",
            4: "Drug_Disease",
            5: "Anatomy_Disease",
            6: "Frequency_Drug",
            7: "Duration_Drug",
            8: "Amount_Drug",
            9: "Method_Drug",
            10: "SideEff_Drug"}

relations = {frozenset(["Test","Disease"]):"Test_Disease",
            frozenset(["Symptom","Disease"]):"Symptom_Disease",
            frozenset(["Treatment","Disease"]):"Treatment_Disease",
            frozenset(["Drug","Disease"]):"Drug_Disease",
            frozenset(["Anatomy","Disease"]):"Anatomy_Disease",
            frozenset(["Frequency","Drug"]):"Frequency_Drug",
            frozenset(["Duration","Drug"]):"Duration_Drug",
            frozenset(["Amount","Drug"]):"Amount_Drug",
            frozenset(["Method","Drug"]):"Method_Drug",
            frozenset(["SideEff","Drug"]):"SideEff_Drug"}

def f1score(y_test, pred):
    corrct = [y_test[i] for i in range(len(y_test)) if y_test[i]==pred[i]]
    tp = len(list(filter(lambda a: a != 0, corrct)))
    truePos = len(list(filter(lambda a: a != 0, y_test)))
    predPos = len(list(filter(lambda a: a != 0, pred)))
    try:
        precision = tp/predPos
        recall = tp/truePos
        f1 = 2*precision*recall/(precision+recall)
        print("Precision:",precision,"Recall:",recall)
        print("F1 score: ",f1)
    except ZeroDivisionError:
        pass
    
def feat_mega(file):
    # This funtion is similar but different with feat_nn 
    # Character Embedding and Position Embedding
    # Read Text Files
    with open(origin_data_dir+file+".txt", "rb") as f:
        text = f.read().decode('utf-8').replace("  。", " "*3).replace(" "*2, " 。")
    # Read annotations and build dataframe
    total_df = pd.read_csv(origin_data_dir+file+".ann", header=None, sep='\t')
    ents_df = total_df[total_df[0].str.contains('T')]
    ents_df = ents_df.rename(index=str, columns={0: "id", 1:"infs", 2:"name"})
    ents_df['type'] = pd.Series([e[0] for e in ents_df["infs"].str.split()]).values
    ents_df['start'] = pd.Series([int(e[1]) for e in ents_df["infs"].str.split()]).values
    ents_df['end'] = pd.Series([int(e[-1]) for e in ents_df["infs"].str.split()]).values
    ents_df = ents_df.drop(columns="infs").sort_values(by=['start']).reset_index(drop=True)
    # Build relation annotations for query
    rel_df = total_df[total_df[0].str.contains('R')]
    rel_df = rel_df.drop(columns=2).reset_index(drop=True)
    rel_df['pairs'] = pd.Series([set((r.split()[1].split(':')[1], r.split()[2].split(':')[1])) for r in rel_df[1]])
    rel_df['relation'] = pd.Series([r.split()[0].replace('-','_') for r in rel_df[1]])
    for i, row in ents_df.iterrows():
        for ii, rrow in ents_df[min(i+1, ents_df.shape[0]): min(find_pair(i, ents_df), ents_df.shape[0])].iterrows():
            if frozenset([row.type, rrow.type]) not in relations: 
                continue
            # Extract entities
            ent_1 = re.sub('\s', '', row['name'])
            ent_2 = re.sub('\s', '', rrow['name'])
            # Find sentence containing entities
            ent_start = row.start
            ent_end = rrow.end
            sent_start = max(ent_start-50, 0)
            sent_end = min(ent_end+50, len(text))
            for s in [';','。','!','?']:
                if text[:ent_start].rfind(s)+1 != -1: sent_start = max(text[:ent_start].rfind(s)+1, sent_start)
                if text.find(s, ent_end) != -1: sent_end = min(text.find(s, ent_end),sent_end)
            sent_raw = text[sent_start: sent_end]
            if len(sent_raw)>200: continue
            sent = re.sub('\s', '', sent_raw)
            # Write location of entities in each sentence sample
            rs_space = len(text[sent_start: row.start]) - len(re.sub('\s', '', text[sent_start: row.start]))
            re_space = len(text[sent_start: row.end]) - len(re.sub('\s', '', text[sent_start: row.end]))
            rrs_space = len(text[sent_start: rrow.start]) - len(re.sub('\s', '', text[sent_start: rrow.start]))
            rre_space = len(text[sent_start: rrow.end]) - len(re.sub('\s', '', text[sent_start: rrow.end]))
            pos1 = (row.start-sent_start-rs_space,row.end-sent_start-re_space)
            pos2 = (rrow.start-sent_start-rrs_space,rrow.end-sent_start-rre_space)
            # Make some features
            fakeResTT.append(relation2id[relations[frozenset([row.type, rrow.type])]])
            nStripTT.append(len(sent_raw) - len(sent))
            senLenTT.append(len(sent))
            nSepTT.append(sent.count('。')+sent.count('!')+sent.count('?'))
            if pos1[0] > pos2[0]:
                disBwTT.append(pos1[0]-pos2[1])
            else:
                disBwTT.append(pos2[0]-pos1[1])
            # Add sentence matrix
            sentIdx = [char2id[c] if c in char2id else 0 for c in sent]
            sentIdx.extend([0] * (maxLen - len(sentIdx)))
            sent = np.array(sentIdx)
            # Add position matrix
            pos1List = [i-pos1[0] if i<pos1[0] else 0 if i in range(pos1[0],pos1[1]) else i-pos1[1] for i in range(len(sentIdx))]
            pos1List = [i+maxDis if abs(i)<=maxDis else 2*maxDis if i>maxDis else 0 for i in pos1List]
            pos2List = [i-pos2[0] if i<pos2[0] else 0 if i in range(pos2[0],pos2[1]) else i-pos2[1] for i in range(len(sentIdx))]
            pos2List = [i+maxDis if abs(i)<=maxDis else 2*maxDis if i>maxDis else 0 for i in pos2List]
            pos1 = np.array(pos1List)
            pos2 = np.array(pos2List)
            # Load features for pytorch
            sent = Variable(torch.cat([torch.cuda.LongTensor(sent).unsqueeze(0)]*32))
            pos1 = Variable(torch.cat([torch.cuda.LongTensor(pos1).unsqueeze(0)]*32))
            pos2 = Variable(torch.cat([torch.cuda.LongTensor(pos2).unsqueeze(0)]*32))
            # Get predictions from BiLSTM-Attention and PCNN-Attention model
            y_lstm = lstm_model(sent,pos1,pos2)
            y_pcnn = pcnn_model(sent,pos1,pos2)
            prob_lstm.append(float(y_lstm[:,0].mean()))
            prob_pcnn.append(float(y_pcnn[:,0].mean()))
            # Add relationId if mode=="train"
            query = set({row.id, rrow.id})
            q_rel = rel_df.relation[rel_df['pairs']==query].to_string(index=False)
            if q_rel != 'Series([], )':
                relation = rel_df.relation[rel_df['pairs']==query].to_string(index=False)
            else:
                relation = 'unknown'
            try:
                relTT.append(relation2id[relation])
            except KeyError:
                relTT.append(relation2id[relation.split()[0]])
            #print(nStripTT, senLenTT, disBwTT, nSepTT, prob_lstm, prob_pcnn, fakeResTT, rel)
        
def find_pair(i, ents_df):
# Find the most distant entity that is within 90 character
    j = i + 1
    d = 0
    while j<ents_df.shape[0]:
        d  = ents_df.start.loc[j] - ents_df.start.loc[i]
        if d > 100:
            break
        else:
            j += 1
    return j

origin_data_dir = 'datasets/train/'
# Load files
with open ('itmd_files/fs_lgb.pkl', 'rb') as fp:
    lgb_files = pickle.load(fp)
# Load dictionary
with open ('itmd_files/dict.pkl', 'rb') as fp:
    char2id = pickle.load(fp)
# Load trained BiLSTM-Attention and PCNN-Attention model
pcnn_model = torch.load('models/pcnn_att.pt')
lstm_model = torch.load('models/bilstm_att.pt')
# Generate features for Mega-model
nStripTT, senLenTT, disBwTT, nSepTT = [],[],[],[]
fakeResTT, relTT, prob_lstm, prob_pcnn = [],[],[],[]
maxLen = 200
maxDis = 60
for file in lgb_files[:11]:
    feat_mega(file)

# Train LGB as mege-model
train_data = lgb.Dataset(X, label=y)
val_data = lgb.Dataset(X_val, label=y_val)
parameters = {
    'objective': 'softmax',
    'metric': 'softmax',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_class':11,
    'num_leaves': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 20,
    'learning_rate': 0.05
}

lgb_model = lgb.train(parameters,
                  train_data,
                  valid_sets=val_data,
                  num_boost_round=50000,
                  early_stopping_rounds=300,
                  verbose_eval=200)

lgb_smx = lgb_model.predict(X_test)
lgb_pred = np.argmax(lgb_smx, axis=1)
f1score(y_test, lgb_pred)

# save model to file
lgb_model.save_model('models/lgb_model.txt')