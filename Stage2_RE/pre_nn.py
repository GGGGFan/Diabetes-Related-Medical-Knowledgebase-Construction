import os
import sys
import pandas as pd
import re
import numpy as np
import pickle

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

def feat_nn(file, sentTT, pos1TT, pos2TT, relTT):
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
    #For test
    #rel_count += rel_df.shape[0]
    for i, row in ents_df.iterrows():
        for ii, rrow in ents_df[min(i+1, ents_df.shape[0]): min(find_pair(i, ents_df), ents_df.shape[0])].iterrows():
            if frozenset([row.type, rrow.type]) not in relations: 
                continue
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
            sent = text[sent_start: sent_end]
            if len(sent)>200: continue
            sent = re.sub('\s', '', sent)
            # Write location of entities in each sentence sample
            rs_space = len(text[sent_start: row.start]) - len(re.sub('\s', '', text[sent_start: row.start]))
            re_space = len(text[sent_start: row.end]) - len(re.sub('\s', '', text[sent_start: row.end]))
            rrs_space = len(text[sent_start: rrow.start]) - len(re.sub('\s', '', text[sent_start: rrow.start]))
            rre_space = len(text[sent_start: rrow.end]) - len(re.sub('\s', '', text[sent_start: rrow.end]))
            pos1 = (row.start-sent_start-rs_space,row.end-sent_start-re_space)
            pos2 = (rrow.start-sent_start-rrs_space,rrow.end-sent_start-rre_space)
            # Add char2id
            for c in sent:
                if c not in char2id:
                    char2id[c] = len(char2id)
                else:
                    continue
            # Add sentence matrix
            sentIdx = [char2id[c] for c in sent]
            sentIdx.extend([0] * (maxLen - len(sentIdx)))
            sentTT.append(sentIdx)
            # Add position matrix
            pos1List = [i-pos1[0] if i<pos1[0] else 0 if i in range(pos1[0],pos1[1]) else i-pos1[1] for i in range(len(sentIdx))]
            pos1List = [i+maxDis if abs(i)<=maxDis else 2*maxDis if i>maxDis else 0 for i in pos1List]
            pos2List = [i-pos2[0] if i<pos2[0] else 0 if i in range(pos2[0],pos2[1]) else i-pos2[1] for i in range(len(sentIdx))]
            pos2List = [i+maxDis if abs(i)<=maxDis else 2*maxDis if i>maxDis else 0 for i in pos2List]
            pos1TT.append(pos1List)
            pos2TT.append(pos2List)
            # Add relationId if mode=="train"
            query = set({row.id, rrow.id})
            q_rel = rel_df.relation[rel_df['pairs']==query].to_string(index=False)
            if q_rel != 'Series([], )':
                #relation = rel_df.relation[rel_df['pairs']==query].to_string(index=False)
                relTT.append(1)
            else:
                #relation = 'unknown'
                relTT.append(0)
                
maxLen = 200
maxDis = 60

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

char2id = {'unknown':0}

# Data Directory
origin_data_dir = 'datasets/train/'
file_list = os.listdir(origin_data_dir)
file_names = [x.split('.')[0] for x in file_list if x.endswith("txt")]

# Split Data for different models
lstm_data = list(np.random.choice(file_names,int(0.2*len(file_names))))
cnn_data = list(np.random.choice([f for f in file_names if f not in lstm_data],int(0.2*len(file_names))))
lgb_data = [f for f in file_names if f not in lstm_data+cnn_data]
with open('itmd_files/fs_lgb.pkl', 'wb') as fp:
    pickle.dump(lgb_data, fp)

# Generate features for for BiLSTM-Attention model
sentTT, pos1TT, pos2TT, relTT = [],[],[],[]
char2id = {'unknown':0} 
for file in lstm_data:
    feat_nn(file, sentTT, pos1TT, pos2TT, relTT)
sentTT = np.array(sentTT)
pos1TT = np.array(pos1TT)
pos2TT = np.array(pos2TT)
relTT = np.array(relTT)

# Save Embeddings as .npy files
np.save('itmd_files/lstm_sentTT', sentTT)
np.save('itmd_files/lstm_pos1TT', pos1TT)
np.save('itmd_files/lstm_pos2TT', pos2TT)
np.save('itmd_files/lstm_relTT', relTT)

# Generate features for PCNN-Attention model
sentTT, pos1TT, pos2TT, relTT = [],[],[],[]
for file in cnn_data:
    feat_nn(file, sentTT, pos1TT, pos2TT, relTT)
sentTT = np.array(sentTT)
pos1TT = np.array(pos1TT)
pos2TT = np.array(pos2TT)
relTT = np.array(relTT)

# Save Embeddings as .npy files
np.save('itmd_files/pcnn_sentTT', sentTT)
np.save('itmd_files/pcnn_pos1TT', pos1TT)
np.save('itmd_files/pcnn_pos2TT', pos2TT)
np.save('itmd_files/pcnn_relTT', relTT)

# Save embeddings
with open('itmd_files/dict.pkl', 'wb') as fp:
    pickle.dump(char2id, fp)