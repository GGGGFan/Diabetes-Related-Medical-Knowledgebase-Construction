import pandas as pd
import numpy as np
import pickle
import torch
from models.BiLSTM_ATT import BiLSTM_ATT
from models.PCNN_ATT import PCNN_ATT
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Data Directory
origin_data_dir = 'datasets/test/'
file_list = os.listdir(origin_data_dir)
file_names = [x.split('.')[0] for x in file_list if x.endswith("txt")]

# Load dictionary
with open ('itmd_files/dict.pkl', 'rb') as fp:
    char2id = pickle.load(fp)
# Load trained BiLSTM-Attention and PCNN-Attention model
pcnn_model = torch.load('models/pcnn_att.pt')
lstm_model = torch.load('models/bilstm_att.pt')
mega_model = lgb.Booster(model_file='models/lgb_model.txt')
# Test and write submissions
maxLen = 200
maxDis = 60

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

for file in file_names:
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
    rn = 1
    for i, row in ents_df.iterrows():
        for ii, rrow in ents_df[min(i+1, ents_df.shape[0]): min(find_pair(i,ents_df), ents_df.shape[0])].iterrows():
            if frozenset([row.type, rrow.type]) not in relations: 
                continue
            # Most important feature
            resStr = relations[frozenset([row.type, rrow.type])]
            e1_type = row.type
            e2_type = rrow.type  
            if e1_type == resStr.split('_')[0]:
                a1, a2 = row.id, rrow.id
            else:
                a1, a2 = rrow.id, row.id
            fakeRes = relation2id[resStr]
            # Find sentence containing entities
            ent_start = row.start
            ent_end = rrow.end
            sent_start = max(ent_start-50, 0)
            sent_end = min(ent_end+50, len(text))
            for s in [';','。','!','?']:
                if text[:ent_start].rfind(s)+1 != -1: sent_start = max(text[:ent_start].rfind(s)+1, sent_start)
                if text.find(s, ent_end) != -1: sent_end = min(text.find(s, ent_end),sent_end)
            sent_raw = text[sent_start: sent_end]
            sent = re.sub('\s', '', sent_raw)
            # Make some features
            nStrip = len(sent_raw) - len(sent)
            senLen = len(sent)
            nSep = sent.count('。')+sent.count('!')+sent.count('?')
            # Write location of entities in each sentence sample
            rs_space = len(text[sent_start: row.start]) - len(re.sub('\s', '', text[sent_start: row.start]))
            re_space = len(text[sent_start: row.end]) - len(re.sub('\s', '', text[sent_start: row.end]))
            rrs_space = len(text[sent_start: rrow.start]) - len(re.sub('\s', '', text[sent_start: rrow.start]))
            rre_space = len(text[sent_start: rrow.end]) - len(re.sub('\s', '', text[sent_start: rrow.end]))
            pos1 = (row.start-sent_start-rs_space,row.end-sent_start-re_space)
            pos2 = (rrow.start-sent_start-rrs_space,rrow.end-sent_start-rre_space)
            # Add another feature
            if pos1[0] > pos2[0]:
                disBw = pos1[0]-pos2[1]
            else:
                disBw = pos2[0]-pos1[1]
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
            prob_lstm = float(y_lstm[:,0].mean())
            prob_pcnn = float(y_pcnn[:,0].mean())
            # Generate Results for each pair of entities    
            x1 = np.stack((senLen, nStrip, disBw, nSep, prob_lstm, prob_pcnn))
            x2 = np.zeros((11,))
            x2[fakeRes] = 1
            x2 = np.concatenate((x1, x2))
            single_input = np.reshape(x2, (-1, 17))
            temp = mega_model.predict(single_input)
            singlePred = np.argmax(temp, axis=1)[0]
            for rel, idx in relation2id.items():
                 if idx == singlePred:
                    targ = rel
            if targ != 'unknown':
                out = open(origin_data_dir+file+".ann",'a')
                out.write('R'+str(rn)+'\t'+targ+' '+'Arg1:'+a1+' '+'Arg2:'+a2+'\n')
                rn += 1
                out.close()