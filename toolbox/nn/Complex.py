import torch
import torch.nn as nn


class Complex(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout_rate=0.2):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.E_real = nn.Embedding(num_entities, embedding_dim)
        self.E_img = nn.Embedding(num_entities, embedding_dim)
        self.R_real = nn.Embedding(num_relations, embedding_dim)
        self.R_img = nn.Embedding(num_relations, embedding_dim)
        self.dropout = nn.Dropout(input_dropout_rate)
        self.loss = nn.BCELoss()

    def init(self):
        nn.init.xavier_normal_(self.E_real.weight.data)
        nn.init.xavier_normal_(self.E_img.weight.data)
        nn.init.xavier_normal_(self.R_real.weight.data)
        nn.init.xavier_normal_(self.R_img.weight.data)

    def forward(self, e1, rel):
        e1_embedded_real = self.E_real(e1).view(-1, self.embedding_dim)
        rel_embedded_real = self.R_real(rel).view(-1, self.embedding_dim)
        e1_embedded_img = self.E_img(e1).view(-1, self.embedding_dim)
        rel_embedded_img = self.R_img(rel).view(-1, self.embedding_dim)

        e1_embedded_real = self.dropout(e1_embedded_real)
        rel_embedded_real = self.dropout(rel_embedded_real)
        e1_embedded_img = self.dropout(e1_embedded_img)
        rel_embedded_img = self.dropout(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real * rel_embedded_real, self.E_real.weight.transpose(1, 0))
        realimgimg = torch.mm(e1_embedded_real * rel_embedded_img, self.E_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(e1_embedded_img * rel_embedded_real, self.E_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(e1_embedded_img * rel_embedded_img, self.E_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred
