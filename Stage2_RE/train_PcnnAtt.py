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
from models.PCNN_ATT import PCNN_ATT
import multiprocessing as mp
mp.set_start_method('spawn')

# Define evaluation matrix
def f1score(y_test, pred):
    corrct = [y_test[i] for i in range(len(y_test)) if y_test[i]==pred[i]]
    tp = len(list(filter(lambda a: a != 0, corrct)))
    truePos = len(list(filter(lambda a: a != 0, y_test)))
    predPos = len(list(filter(lambda a: a != 0, pred)))
    precision = tp/predPos
    recall = tp/truePos
    try:
        f1 = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        pass
    print("Precision:",precision,"Recall:",recall)
    print("F1 score: ",f1)


# Read saved embedding features
sentTT = np.load('itmd_files/pcnn_sentTT.npy')
pos1TT = np.load('itmd_files/pcnn_pos1TT.npy')
pos2TT = np.load('itmd_files/pcnn_pos2TT.npy')
relTT = np.load('itmd_files/pcnn_relTT.npy')

# Set parameters
config = {}
config['EMBEDDING_SIZE'] = 2000
config['EMBEDDING_DIM'] = 100
config['POS_SIZE'] = 200
config['POS_DIM'] = 100
config['HIDDEN_DIM'] = 200
config['TAG_SIZE'] = 11
config['BATCH'] = 32
config["pretrained"]=False
config['FILT_NUM'] = 120

EPOCHS = 25
BATCH = 32
learning_rate = 0.0005

# Initialize model
sample_n = sentTT.shape[0]
model = PCNN_ATT(config).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')
# Load data
sent0 = torch.cuda.LongTensor(sentTT[:sample_n-sample_n%BATCH])
position1 = torch.cuda.LongTensor(pos1TT[:sample_n-sample_n%BATCH])
position2 = torch.cuda.LongTensor(pos2TT[:sample_n-sample_n%BATCH])
labels = torch.cuda.LongTensor(relTT[:sample_n-sample_n%BATCH])
train_datasets = D.TensorDataset(sent0,position1,position2,labels)
train_dataloader = D.DataLoader(train_datasets,BATCH,True)

# Train
for epoch in range(EPOCHS):
    print("epoch:",epoch)
    tag_true = []
    preds = []
    for sentence,pos1,pos2,tag in train_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)
        tags = Variable(tag)
        loss = criterion(y, tags)      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        prob,y = torch.max(y,1)
        tag_true.extend(list(tag.data.cpu().numpy()))
        preds.extend(list(y.data.cpu().numpy()))
    try:
        f1score(tag_true, preds)
    except:
        continue

#Save model
torch.save(model, 'models/pcnn_att.pt')