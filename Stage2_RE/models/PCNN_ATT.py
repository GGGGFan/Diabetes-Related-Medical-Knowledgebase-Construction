import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNN_ATT(nn.Module):
    def __init__(self, config):
        super(PCNN_ATT, self).__init__()
        self.batch = config['BATCH']
        self.embedding_size = config['EMBEDDING_SIZE']
        self.embedding_dim = config['EMBEDDING_DIM']
        self.hidden_dim = config['HIDDEN_DIM']
        self.tag_size = config['TAG_SIZE']
        self.pos_size = config['POS_SIZE']
        self.pos_dim = config['POS_DIM']
        self.filters_num = config['FILT_NUM']
        self.filters = [3]
        self.feature_dim = self.embedding_dim + 2 * self.pos_dim
        self.word_embeds = nn.Embedding(self.embedding_size,
                                        self.embedding_dim)
        self.pos1_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size, self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size, self.filters_num)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.filters_num,
                                    (k, self.feature_dim),
                                    padding=(int(k / 2), 0))
                                    for k in self.filters])
        self.cnn_linear = nn.Linear(self.filters_num * len(self.filters),
                                    self.filters_num)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.dropout_emb = nn.Dropout(p=0.1)
        self.dropout_att = nn.Dropout(p=0.1)
        self.hidden = self.init_hidden()
        self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.filters_num))
        self.relation_bias = nn.Parameter(torch.randn(self.batch, self.tag_size, 1))

    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2).to('cuda')

    def attention(self, H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight, M), 2)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)

    def forward(self, sentence, pos1, pos2):
        embeds = torch.cat((self.word_embeds(sentence),
                            self.pos1_embeds(pos1),
                            self.pos2_embeds(pos2)), 2)
        embeds = embeds.unsqueeze(1)
        # CNN
        cnn_out = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        # Piece-wise Pooling
        ent1_pos = [torch.nonzero(i) for i in pos1 == 60]
        ent2_pos = [torch.nonzero(i) for i in pos2 == 60]
        pcnn_out = []
        for i in range(cnn_out[0].size()[0]):
            if ent1_pos[i].min() < ent2_pos[i].min():
                m1, m2, m3, m4 = ent1_pos[i].min(),ent1_pos[i].max(),ent2_pos[i].min(),ent2_pos[i].max()
            else:
                m1, m2, m3, m4 = ent2_pos[i].min(),ent2_pos[i].max(),ent1_pos[i].min(),ent1_pos[i].max()
            a = cnn_out[0][i, :, :max(m1, 1)].max(1)[0]
            b = cnn_out[0][i, :, m2:max(m3, m2 + 1)].max(1)[0]
            c = cnn_out[0][i, :, m4:].max(1)[0]
            single_out = torch.stack((a, b, c)).transpose(0, 1)
            pcnn_out.append(single_out)
        pcnn_out = torch.stack(pcnn_out)

        att_out = F.tanh(self.attention(pcnn_out))
        relation = torch.tensor([i for i in range(self.tag_size)],
                                dtype = torch.long).repeat(self.batch, 1).to('cuda')

        relation = self.relation_embeds(relation)
        wei = torch.bmm(relation, att_out)
        res = torch.add(wei, self.relation_bias)
        res = F.softmax(res, 1)
        return res.view(self.batch, -1)
