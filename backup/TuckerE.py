"""
@date: 2021/10/14
@description: null
"""
import torch
import torch.nn as nn

from toolbox.nn.ComplexEmbedding import ComplexEmbedding, ComplexDropout, ComplexBatchNorm1d
from toolbox.nn.Regularizer import N3
from toolbox.nn.TuckER import CoreTuckER
from toolbox.nn.TuckerMobiusE import BatchComplexScoringAll


class TuckerE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3, regularization_weight=0.1):
        super(TuckerE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = nn.BCELoss()
        self.flag_hamilton_mul_norm = norm_flag
        self.E = ComplexEmbedding(self.num_entities, self.embedding_dim, 2)  # a + bi
        self.R2 = nn.Embedding(num_relations, embedding_dim)
        self.real_tucker = CoreTuckER(embedding_dim, embedding_dim, hidden_dropout)
        self.img_tucker = CoreTuckER(embedding_dim, embedding_dim, hidden_dropout)
        self.E_dropout = ComplexDropout([input_dropout, input_dropout])
        self.hidden_dp = ComplexDropout([hidden_dropout, hidden_dropout])
        self.E_bn = ComplexBatchNorm1d(self.embedding_dim, 2)

        self.scoring_all = BatchComplexScoringAll()
        self.regularizer = N3(regularization_weight)

    def forward(self, h_idx, r_idx):
        h_idx = h_idx.view(-1)
        r_idx = r_idx.view(-1)
        return self.forward_head_batch(h_idx, r_idx)

    def forward_head_batch(self, e1_idx, rel_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        h = self.E(e1_idx)
        r2 = self.R2(rel_idx)

        h_a, h_b = h
        t_a = self.real_tucker(h_a, r2)
        t_b = self.img_tucker(h_b, r2)
        h = (t_a, t_b)

        E_a, E_b = self.E.get_embeddings()
        E_a = self.real_tucker.w(E_a, r2)
        E_b = self.img_tucker.w(E_b, r2)
        E = (E_a, E_b)
        # E = self.E.get_embeddings()
        if self.flag_hamilton_mul_norm:
            score_a, score_b = self.scoring_all(h, E)  # a + b i
        else:
            score_a, score_b = self.scoring_all(self.E_dropout(h), self.E_dropout(self.E_bn(E)))
        score = score_a + score_b
        score = torch.sigmoid(score)

        return score

    def init(self):
        self.E.init()


if __name__ == "__main__":
    h = torch.LongTensor([[i] for i in range(5)])
    r = torch.LongTensor([[i] for i in range(5)])
    model = TuckerE(10, 10, 5)
    pred = model(h, r)
    print(pred)
    print(pred.shape)
    y = torch.rand_like(pred)
    print(model.loss(pred, y))
