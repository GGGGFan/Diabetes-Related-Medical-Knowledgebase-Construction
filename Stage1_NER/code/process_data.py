import os
import pandas as pd
import re
import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform

def data_makeup():
    # Find missed label in TRAINING data
    origin_data_dir = "data/train/"
    file_list = os.listdir(origin_data_dir)
    file_names = [x.split('.')[0] for x in file_list if x.endswith("txt")]
    
    for file in file_names:
        # Read all entites mentioned in ann files
        tags = pd.read_csv(origin_data_dir+file+".ann", header=None, sep='\t')
        # Sort by entities length
        tags.index = tags[2].str.len()
        tags = tags.sort_index(ascending=False).reset_index(drop=True)
        starts = [int(i.split()[1]) for i in tags[1]]
        anns = {}
        for i in range(tags.shape[0]):
            if tags.iloc[i][2] not in anns:
                anns[tags.iloc[i][2]] = tags.iloc[i][1].split(' ')[0]
        # Open txt file
        with open(origin_data_dir+file+".txt", 'rb') as f:
            text = f.read().decode('utf-8')
        # Add entities to ann
        ann_file = open(origin_data_dir+file+".ann",'a')
        for i in anns:
            # Extract locations of entities in original .ann files
            found = tags[tags[2]==i].shape[0]
            try:
                missed = 0
                for m in re.finditer(i, text):
                    if m.start() not in starts: # Find how many times entities are missed
                        missed += 1.0
                if found / missed > 0.8:
                    for m in re.finditer(i, text):
                        if m.start() not in starts: # Add to original .ann file
                            ann_file.write('Tn'+'\t'+anns[i]+' '+str(m.start())+' '+str(m.end())+'\t'+i+'\n')
                            starts.append(m.start())
            except:
                pass
        ann_file.close()

def load_data():
    train = _parse_data(open('data/ruijin_train.data', 'rb'))
    test = _parse_data(open('data/ruijin_dev.data', 'rb'))
    
    word_counts = Counter(row[0] for sample in train+test for row in sample)
    
    vocab = [w for w, f in iter(word_counts.items()) if f >= 0]
    
    word2idx = dict((w, i+2) for i, w in enumerate(vocab))
    
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    
    chunk_tags = ['O', 'B-Disease', 'I-Disease', 'B-Reason', 'I-Reason', "B-Symptom", "I-Symptom", "B-Test", "I-Test", "B-Test_Value", "I-Test_Value", "B-Drug", "I-Drug", "B-Frequency", "I-Frequency", "B-Amount", "I-Amount", "B-Treatment", "I-Treatment", "B-Operation", "I-Operation", "B-Method", "I-Method", "B-SideEff","I-SideEff","B-Anatomy", "I-Anatomy", "B-Level", "I-Level", "B-Duration", "I-Duration"]

    with open('data/dict.pkl', 'wb') as outp:
        pickle.dump((word2idx,chunk_tags), outp)

    train = _process_data(train, word2idx, chunk_tags)
    test = _process_data(test, word2idx, chunk_tags)
    return train, test, (word2idx, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def _process_data(data, word2idx, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    
    
    x = [[word2idx.get(w[0], 1) for w in s] for s in data]

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    
    # debug 
    #y_chunk = []
    #for i,s in enumerate(data):
    #    for j,w in enumerate(s):
    #        #print(i,j)
    #        y_chunk.append(chunk_tags.index(w[1]))

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, word2idx, maxlen=2000):
    x = [word2idx.get(w[0], 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length
