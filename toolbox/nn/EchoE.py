import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from toolbox.nn.Highway import Highway


class GraphEncoder(nn.Module):
    def __init__(self, entity_dim, relation_dim):
        super(GraphEncoder, self).__init__()
        self.a_i = nn.Linear(entity_dim, 1, bias=False)  # h
        self.a_j = nn.Linear(entity_dim, 1, bias=False)  # t
        self.a_k = nn.Linear(relation_dim, 1, bias=False)  # r
        self.out_dim = entity_dim + entity_dim + relation_dim

    def forward(self, h, r, t):
        """
        h,r,t:BxNxd
        """
        e_i = self.a_i(h).squeeze(dim=-1)
        r_k = self.a_k(r).squeeze(dim=-1)
        e_j = self.a_j(t).squeeze(dim=-1)
        a = e_i + e_j + r_k
        alpha = F.leaky_relu(a).float().softmax(dim=-1)

        v = torch.cat([h, r, t], dim=-1)
        # print(alpha.shape, v.shape)
        ans = alpha.unsqueeze(dim=-2)
        # print(ans.shape, v.shape)
        ans = ans.bmm(v).squeeze(dim=-2)
        # print(ans.shape)
        return ans


class BatchGraphEncoder(nn.Module):
    def __init__(self, entity_dim, relation_dim):
        super(BatchGraphEncoder, self).__init__()
        self.a_i = nn.Linear(entity_dim, 1, bias=False)  # h
        self.a_j = nn.Linear(entity_dim, 1, bias=False)  # t
        self.a_k = nn.Linear(relation_dim, 1, bias=False)  # r
        self.out_dim = entity_dim + entity_dim + relation_dim

    def forward(self, h, r, t):
        """
        h:bxd
        r:bxd
        t:bxNxd
        """
        h = h.unsqueeze(dim=1)
        r = r.unsqueeze(dim=1)
        e_i = self.a_i(h)
        r_k = self.a_k(r)
        e_j = self.a_j(t)
        a = e_i + e_j + r_k
        alpha = softmax(F.leaky_relu(a).float(), torch.range(start=1, end=a.size(0)).long().to(a.device))

        h = torch.cat([h for i in range(t.size(1))], dim=1)
        r = torch.cat([r for i in range(t.size(1))], dim=1)
        v = torch.cat([h, r, t], dim=-1)
        alpha = alpha.transpose(0, 1)
        alpha = alpha.view(alpha.size(0), 1, -1)
        v = v.transpose(0, 1)
        ans = alpha.bmm(v).view(-1, self.out_dim)
        return ans


class Composer(nn.Module):
    def __init__(self, entity_dim, relation_dim):
        super(Composer, self).__init__()
        self.out_dim = entity_dim + entity_dim + relation_dim
        self.highway = Highway(self.out_dim)

    def forward(self, h, r, t, g):
        # print(h.shape, r.shape, t.shape, g.shape)
        if len(t.size()) == 3:
            h = h.unsqueeze(dim=1)
            r = r.unsqueeze(dim=1)
            h = torch.cat([h for i in range(t.size(1))], dim=1)
            r = torch.cat([r for i in range(t.size(1))], dim=1)
        # print(h.shape, r.shape, t.shape, g.shape)
        x = torch.cat([h, r, t], dim=-1)
        # print(x.size(), g.size())
        x = self.highway(x, g)
        return x


class ConvE(nn.Module):
    def __init__(self, embedding_dim, img_h=10, input_dropout=0.2, hidden_dropout1=0.2, hidden_dropout2=0.3):
        super(ConvE, self).__init__()
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout1)
        self.feature_map_drop = nn.Dropout2d(hidden_dropout2)

        self.img_h = img_h
        self.img_w = embedding_dim // self.img_h

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        hidden_size = (self.img_h * 2 - 3 + 1) * (self.img_w - 3 + 1) * 32
        self.fc = nn.Linear(hidden_size, embedding_dim)

    def forward(self, h, r):
        h = h.view(-1, 1, self.img_h, self.img_w)
        r = r.view(-1, 1, self.img_h, self.img_w)

        x = torch.cat([h, r], 2)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, entity_dim, relation_dim, input_dropout=0.2, hidden_dropout1=0.2, hidden_dropout2=0.3):
        super(Decoder, self).__init__()
        self.model = ConvE(entity_dim, 10, input_dropout, hidden_dropout1, hidden_dropout2)

    def forward(self, E, R, head, rel, g):
        h = E(head)
        r = R(rel)
        x = self.model(h, r)
        return x


class L1_Loss(nn.Module):
    def __init__(self, gamma=3):
        super(L1_Loss, self).__init__()
        self.gamma = gamma

    def dis(self, x, y):
        return torch.sum(torch.abs(x - y), dim=-1)

    def forward(self, x1, x2, train_set, train_batch, false_pair):
        x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
        x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1))

        dis_x1_x2 = self.dis(x1_train, x2_train)
        loss11 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg1)))
        loss12 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x1_train, x1_neg2)))
        loss21 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg1)))
        loss22 = torch.mean(F.relu(self.gamma + dis_x1_x2 - self.dis(x2_train, x2_neg2)))
        if false_pair is not None:
            x1_test_false, x2_test_false = x1[false_pair[:, 0]], x2[false_pair[:, 1]]
            loss3 = torch.mean(F.relu(self.gamma - self.dis(x1_test_false, x2_test_false)))
            loss = (loss11 + loss12 + loss21 + loss22 + loss3) / 5
        else:
            loss = (loss11 + loss12 + loss21 + loss22) / 4
        return loss


class EchoE(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim):
        super(EchoE, self).__init__()
        self.emb_e = nn.Embedding(num_entities, entity_dim)
        self.emb_rel = nn.Embedding(num_relations, relation_dim)

        self.encoder = GraphEncoder(entity_dim, relation_dim)
        self.decoder = Decoder(entity_dim, relation_dim)
        self.composer = Composer(entity_dim, relation_dim)
        self.proj = nn.Linear(self.composer.out_dim, self.composer.out_dim)

    def init(self):
        nn.init.xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, head0, rel0, tail0, head, rel, tail):
        g, h, r = self.ghr(head0, rel0, tail0, head, rel)

        t1 = self.decoder(self.emb_e, self.emb_rel, head, rel, g)
        g1 = self.composer(h, r, t1, g)
        g1 = self.proj(g1)

        t2 = self.emb_e(tail)
        g2 = self.composer(h, r, t2, g).detach()  # 截断梯度

        loss = torch.mean(F.relu(self.dis(g1, g2)))
        return loss

    def ghr(self, head0, rel0, tail0, head, rel):
        g = self.encode(head0, rel0, tail0)
        h = self.emb_e(head)
        r = self.emb_rel(rel)
        return g, h, r

    def encode(self, head0, rel0, tail0):
        h0 = self.emb_e(head0)
        r0 = self.emb_rel(rel0)
        t0 = self.emb_e(tail0)
        g = self.encoder(h0, r0, t0)
        return g

    def dis(self, x, y):
        return torch.sum(torch.abs(x - y), dim=-1)
