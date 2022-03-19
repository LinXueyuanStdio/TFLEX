# https://github.com/cheungdaven/QuatE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import RandomState


def quaternion_init(in_features, out_features, criterion='he'):
    fan_in = in_features
    fan_out = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ', criterion)
    rng = RandomState(123)

    # Generating randoms and purely imaginary quaternions :
    kernel_shape = (in_features, out_features)

    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(0.0, 1.0, number_of_weights)
    v_j = np.random.uniform(0.0, 1.0, number_of_weights)
    v_k = np.random.uniform(0.0, 1.0, number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


class QuatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, ent_dropout=0.2, rel_dropout=0.2):
        super(QuatE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.emb_s_a = nn.Embedding(num_entities, embedding_dim)
        self.emb_x_a = nn.Embedding(num_entities, embedding_dim)
        self.emb_y_a = nn.Embedding(num_entities, embedding_dim)
        self.emb_z_a = nn.Embedding(num_entities, embedding_dim)
        self.rel_s_b = nn.Embedding(num_relations, embedding_dim)
        self.rel_x_b = nn.Embedding(num_relations, embedding_dim)
        self.rel_y_b = nn.Embedding(num_relations, embedding_dim)
        self.rel_z_b = nn.Embedding(num_relations, embedding_dim)
        self.rel_w = nn.Embedding(num_relations, embedding_dim)
        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = nn.Dropout(ent_dropout)
        self.rel_dropout = nn.Dropout(rel_dropout)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def init(self):
        r, i, j, k = quaternion_init(self.num_entities, self.embedding_dim)
        r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
        self.emb_s_a.weight.data = r.type_as(self.emb_s_a.weight.data)
        self.emb_x_a.weight.data = i.type_as(self.emb_x_a.weight.data)
        self.emb_y_a.weight.data = j.type_as(self.emb_y_a.weight.data)
        self.emb_z_a.weight.data = k.type_as(self.emb_z_a.weight.data)

        s, x, y, z = quaternion_init(self.num_entities, self.embedding_dim)
        s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
        self.rel_s_b.weight.data = s.type_as(self.rel_s_b.weight.data)
        self.rel_x_b.weight.data = x.type_as(self.rel_x_b.weight.data)
        self.rel_y_b.weight.data = y.type_as(self.rel_y_b.weight.data)
        self.rel_z_b.weight.data = z.type_as(self.rel_z_b.weight.data)
        nn.init.xavier_uniform_(self.rel_w.weight.data)

    def _calc(self, s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b):
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        score_r = (A * s_c + B * x_c + C * y_c + D * z_c)
        # print(score_r.size())
        # score_i = A * x_c + B * s_c + C * z_c - D * y_c
        # score_j = A * y_c - B * z_c + C * s_c + D * x_c
        # score_k = A * z_c + B * y_c - C * x_c + D * s_c
        return -torch.sum(score_r, -1)

    def loss(self, score, regul, regul2):
        # self.batch_y = ((1.0-0.1)*self.batch_y) + (1.0/self.batch_y.size(1)) /// (1 + (1 + self.batch_y)/2) *
        return (
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + self.config.lmbda * regul2
        )

    def forward(self):
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)

        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        regul = (torch.mean(torch.abs(s_a) ** 2)
                 + torch.mean(torch.abs(x_a) ** 2)
                 + torch.mean(torch.abs(y_a) ** 2)
                 + torch.mean(torch.abs(z_a) ** 2)
                 + torch.mean(torch.abs(s_c) ** 2)
                 + torch.mean(torch.abs(x_c) ** 2)
                 + torch.mean(torch.abs(y_c) ** 2)
                 + torch.mean(torch.abs(z_c) ** 2)
                 )
        regul2 = (torch.mean(torch.abs(s_b) ** 2)
                  + torch.mean(torch.abs(x_b) ** 2)
                  + torch.mean(torch.abs(y_b) ** 2)
                  + torch.mean(torch.abs(z_b) ** 2))

        '''
        + torch.mean(s_b ** 2)
            + torch.mean(x_b ** 2)
            + torch.mean(y_b ** 2)
            + torch.mean(z_b ** 2)
        '''

        return self.loss(score, regul, regul2)

    def predict(self):
        s_a = self.emb_s_a(self.batch_h)
        x_a = self.emb_x_a(self.batch_h)
        y_a = self.emb_y_a(self.batch_h)
        z_a = self.emb_z_a(self.batch_h)

        s_c = self.emb_s_a(self.batch_t)
        x_c = self.emb_x_a(self.batch_t)
        y_c = self.emb_y_a(self.batch_t)
        z_c = self.emb_z_a(self.batch_t)

        s_b = self.rel_s_b(self.batch_r)
        x_b = self.rel_x_b(self.batch_r)
        y_b = self.rel_y_b(self.batch_r)
        z_b = self.rel_z_b(self.batch_r)

        score = self._calc(s_a, x_a, y_a, z_a, s_c, x_c, y_c, z_c, s_b, x_b, y_b, z_b)
        return score.cpu().data.numpy()


def quaternion_mul_with_unit_norm(Q_1, Q_2):
    a_h, b_h, c_h, d_h = Q_1  # = {a_h + b_h i + c_h j + d_h k : a_r, b_r, c_r, d_r \in R^k}
    a_r, b_r, c_r, d_r = Q_2  # = {a_r + b_r i + c_r j + d_r k : a_r, b_r, c_r, d_r \in R^k}

    # Normalize the relation to eliminate the scaling effect
    denominator = torch.sqrt(a_r ** 2 + b_r ** 2 + c_r ** 2 + d_r ** 2)
    p = a_r / denominator
    q = b_r / denominator
    u = c_r / denominator
    v = d_r / denominator
    #  Q'=E Hamilton product R
    r_val = a_h * p - b_h * q - c_h * u - d_h * v
    i_val = a_h * q + b_h * p + c_h * v - d_h * u
    j_val = a_h * u - b_h * v + c_h * p + d_h * q
    k_val = a_h * v + b_h * u - c_h * q + d_h * p
    return r_val, i_val, j_val, k_val


def quaternion_mul(Q_1, Q_2):
    a_h, b_h, c_h, d_h = Q_1  # = {a_h + b_h i + c_h j + d_h k : a_r, b_r, c_r, d_r \in R^k}
    a_r, b_r, c_r, d_r = Q_2  # = {a_r + b_r i + c_r j + d_r k : a_r, b_r, c_r, d_r \in R^k}
    r_val = a_h * a_r - b_h * b_r - c_h * c_r - d_h * d_r
    i_val = a_h * b_r + b_h * a_r + c_h * d_r - d_h * c_r
    j_val = a_h * c_r - b_h * d_r + c_h * a_r + d_h * b_r
    k_val = a_h * d_r + b_h * c_r - c_h * b_r + d_h * a_r
    return r_val, i_val, j_val, k_val


class QMult(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3):
        super(QMult, self).__init__()
        self.name = 'QMult'
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = nn.BCELoss()
        self.flag_hamilton_mul_norm = norm_flag
        # Quaternion embeddings of entities
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i
        self.emb_ent_j = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary j
        self.emb_ent_k = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary k
        # Quaternion embeddings of relations.
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        self.emb_rel_j = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary j
        self.emb_rel_k = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary k
        # Dropouts for quaternion embeddings of ALL entities.
        self.input_dp_ent_real = nn.Dropout(input_dropout)
        self.input_dp_ent_i = nn.Dropout(input_dropout)
        self.input_dp_ent_j = nn.Dropout(input_dropout)
        self.input_dp_ent_k = nn.Dropout(input_dropout)
        # Dropouts for quaternion embeddings of relations.
        self.input_dp_rel_real = nn.Dropout(input_dropout)
        self.input_dp_rel_i = nn.Dropout(input_dropout)
        self.input_dp_rel_j = nn.Dropout(input_dropout)
        self.input_dp_rel_k = nn.Dropout(input_dropout)
        # Dropouts for quaternion embeddings obtained from quaternion multiplication.
        self.hidden_dp_real = nn.Dropout(hidden_dropout)
        self.hidden_dp_i = nn.Dropout(hidden_dropout)
        self.hidden_dp_j = nn.Dropout(hidden_dropout)
        self.hidden_dp_k = nn.Dropout(hidden_dropout)
        # Batch normalization for quaternion embeddings of ALL entities.
        self.bn_ent_real = nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_j = nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_k = nn.BatchNorm1d(self.embedding_dim)
        # Batch normalization for quaternion embeddings of relations.
        self.bn_rel_real = nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_j = nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_k = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, h_idx, r_idx):
        return self.forward_head_batch(h_idx.view(-1), r_idx.view(-1))

    def forward_head_batch(self, h_idx, r_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(h_idx)
        emb_head_i = self.emb_ent_i(h_idx)
        emb_head_j = self.emb_ent_j(h_idx)
        emb_head_k = self.emb_ent_k(h_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(r_idx)
        emb_rel_i = self.emb_rel_i(r_idx)
        emb_rel_j = self.emb_rel_j(r_idx)
        emb_rel_k = self.emb_rel_k(r_idx)

        if self.flag_hamilton_mul_norm:
            # (2) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (3) Inner product of (2) with all entities.
            real_score = torch.mm(r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(i_val, self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(j_val, self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(k_val, self.emb_ent_k.weight.transpose(1, 0))
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2)-relations.
            # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                     self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                     self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                     self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))
            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Apply BN + DP on ALL entities.
            # (3.3) Inner product
            real_score = torch.mm(self.hidden_dp_real(r_val),
                                  self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)).transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(i_val),
                               self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight)).transpose(1, 0))
            j_score = torch.mm(self.hidden_dp_j(j_val),
                               self.input_dp_ent_j(self.bn_ent_j(self.emb_ent_j.weight)).transpose(1, 0))
            k_score = torch.mm(self.hidden_dp_k(k_val),
                               self.input_dp_ent_k(self.bn_ent_k(self.emb_ent_k.weight)).transpose(1, 0))
        score = real_score + i_score + j_score + k_score
        return torch.sigmoid(score)

    def forward_tail_batch(self, r_idx, e2_idx):
        """
        Completed.
        Given a relation and a tail entity(r,t), we compute scores for all possible triples,i.e.,
        [score(x,r,t)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Quaternion embeddings of relations.
        emb_rel_real = self.emb_rel_real(r_idx)
        emb_rel_i = self.emb_rel_i(r_idx)
        emb_rel_j = self.emb_rel_j(r_idx)
        emb_rel_k = self.emb_rel_k(r_idx)
        # (1.2)  Reshape Quaternion embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx).view(-1, self.embedding_dim, 1)
        emb_tail_i = self.emb_ent_i(e2_idx).view(-1, self.embedding_dim, 1)
        emb_tail_j = self.emb_ent_j(e2_idx).view(-1, self.embedding_dim, 1)
        emb_tail_k = self.emb_ent_k(e2_idx).view(-1, self.embedding_dim, 1)
        if self.flag_hamilton_mul_norm:
            # (2) Reshape (1.1)-relations.
            emb_rel_real = emb_rel_real.view(-1, 1, self.embedding_dim)
            emb_rel_i = emb_rel_i.view(-1, 1, self.embedding_dim)
            emb_rel_j = emb_rel_j.view(-1, 1, self.embedding_dim)
            emb_rel_k = emb_rel_k.view(-1, 1, self.embedding_dim)
            # (3) Quaternion multiplication of ALL entities and unit normalized (2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(self.emb_ent_real.weight, self.emb_ent_i.weight,
                     self.emb_ent_j.weight, self.emb_ent_k.weight),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (4) Inner product of (3) with (1.2).
            real_score = torch.matmul(r_val, emb_tail_real)
            i_score = torch.matmul(i_val, emb_tail_i)
            j_score = torch.matmul(j_val, emb_tail_j)
            k_score = torch.matmul(k_val, emb_tail_k)
        else:
            # (2) BN + Dropout + Reshape (1.1)-relations.
            emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)).view(-1, 1, self.embedding_dim)
            emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)).view(-1, 1, self.embedding_dim)
            emb_rel_j = self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)).view(-1, 1, self.embedding_dim)
            emb_rel_k = self.input_dp_rel_k(self.bn_rel_k(emb_rel_k)).view(-1, 1, self.embedding_dim)

            # (3)
            # (3.1) BN + Dropout on ALL entities.
            # (3.2) Quaternion multiplication of (3.1) and (2).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)),
                     self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight)),
                     self.input_dp_ent_j(self.bn_ent_i(self.emb_ent_j.weight)),
                     self.input_dp_ent_k(self.bn_ent_k(self.emb_ent_k.weight))),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))

            # (4)
            # (4.1) Dropout on (3).
            # (4.2) Inner product on (4.1) with (1.2).
            real_score = torch.matmul(self.hidden_dp_real(r_val), emb_tail_real)
            i_score = torch.matmul(self.hidden_dp_i(i_val), emb_tail_i)
            j_score = torch.matmul(self.hidden_dp_j(j_val), emb_tail_j)
            k_score = torch.matmul(self.hidden_dp_k(k_val), emb_tail_k)
        score = torch.sigmoid(real_score + i_score + j_score + k_score)
        score = score.squeeze()
        return score

    def forward_head_and_loss(self, h_idx, r_idx, targets):
        return self.loss(self.forward_head_batch(h_idx=h_idx, r_idx=r_idx), targets)

    def forward_tail_and_loss(self, r_idx, e2_idx, targets):
        return self.loss(self.forward_tail_batch(r_idx=r_idx, e2_idx=e2_idx), targets)

    def init(self):
        nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        nn.init.xavier_normal_(self.emb_ent_i.weight.data)
        nn.init.xavier_normal_(self.emb_ent_j.weight.data)
        nn.init.xavier_normal_(self.emb_ent_k.weight.data)
        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_i.weight.data)
        nn.init.xavier_normal_(self.emb_rel_j.weight.data)
        nn.init.xavier_normal_(self.emb_rel_k.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data,
                                self.emb_ent_j.weight.data, self.emb_ent_k.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data,
                             self.emb_rel_j.weight.data, self.emb_rel_k.weight.data), 1)

        return entity_emb, rel_emb


class ConvQ(nn.Module):
    """ Convolutional Quaternion Knowledge Graph Embeddings"""

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 kernel_size=3, num_of_output_channels=16, feature_map_dropout=0.3,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3):
        super(ConvQ, self).__init__()
        self.name = 'ConvQ'
        self.loss = nn.BCELoss()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.kernel_size = kernel_size
        self.num_of_output_channels = num_of_output_channels
        self.flag_hamilton_mul_norm = norm_flag
        # Embeddings.
        self.emb_ent_real = nn.Embedding(self.num_entities, self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary i
        self.emb_ent_j = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary j
        self.emb_ent_k = nn.Embedding(self.num_entities, self.embedding_dim)  # imaginary k
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        self.emb_rel_j = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary j
        self.emb_rel_k = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary k
        # Dropouts
        self.input_dp_ent_real = nn.Dropout(input_dropout)
        self.input_dp_ent_i = nn.Dropout(input_dropout)
        self.input_dp_ent_j = nn.Dropout(input_dropout)
        self.input_dp_ent_k = nn.Dropout(input_dropout)
        self.input_dp_rel_real = nn.Dropout(input_dropout)
        self.input_dp_rel_i = nn.Dropout(input_dropout)
        self.input_dp_rel_j = nn.Dropout(input_dropout)
        self.input_dp_rel_k = nn.Dropout(input_dropout)
        self.hidden_dp_real = nn.Dropout(hidden_dropout)
        self.hidden_dp_i = nn.Dropout(hidden_dropout)
        self.hidden_dp_j = nn.Dropout(hidden_dropout)
        self.hidden_dp_k = nn.Dropout(hidden_dropout)
        # Batch Normalization
        self.bn_ent_real = nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_j = nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_k = nn.BatchNorm1d(self.embedding_dim)

        self.bn_rel_real = nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_j = nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_k = nn.BatchNorm1d(self.embedding_dim)

        # Convolution
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                               kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = self.embedding_dim * 8 * self.num_of_output_channels  # 8 because of 8 real values in 2 quaternions
        self.fc1 = nn.Linear(self.fc_num_input, self.embedding_dim * 4)  # Hard compression.

        self.bn_conv1 = nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = nn.BatchNorm1d(self.embedding_dim * 4)
        self.feature_map_dropout = nn.Dropout2d(feature_map_dropout)

    def forward(self, h_idx, r_idx):
        return self.forward_head_batch(h_idx.view(-1), r_idx.view(-1))

    def residual_convolution(self, Q_1, Q_2):
        emb_ent_real, emb_ent_imag_i, emb_ent_imag_j, emb_ent_imag_k = Q_1
        emb_rel_real, emb_rel_imag_i, emb_rel_imag_j, emb_rel_imag_k = Q_2
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_j.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_k.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_j.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_k.view(-1, 1, 1, self.embedding_dim)], 2)

        # Think of x a n image of two quaternions.
        # Batch norms after fully connnect and Conv layers
        # and before nonlinearity.
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc1(x)))
        return torch.chunk(x, 4, dim=1)

    def forward_head_batch(self, h_idx, r_idx):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Quaternion embeddings of head entities
        emb_head_real = self.emb_ent_real(h_idx)
        emb_head_i = self.emb_ent_i(h_idx)
        emb_head_j = self.emb_ent_j(h_idx)
        emb_head_k = self.emb_ent_k(h_idx)
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(r_idx)
        emb_rel_i = self.emb_rel_i(r_idx)
        emb_rel_j = self.emb_rel_j(r_idx)
        emb_rel_k = self.emb_rel_k(r_idx)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                                        Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
        conv_real, conv_imag_i, conv_imag_j, conv_imag_k = Q_3
        if self.flag_hamilton_mul_norm:
            # (3) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Inner product of (4.1) with ALL entities.
            real_score = torch.mm(conv_real * r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(conv_imag_i * i_val, self.emb_ent_i.weight.transpose(1, 0))
            j_score = torch.mm(conv_imag_j * j_val, self.emb_ent_j.weight.transpose(1, 0))
            k_score = torch.mm(conv_imag_k * k_val, self.emb_ent_k.weight.transpose(1, 0))
        else:
            # (3)
            # (3.1) Apply BN + Dropout on (1.2).
            # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(emb_head_real, emb_head_i, emb_head_j, emb_head_k),
                Q_2=(self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                     self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),
                     self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)),
                     self.input_dp_rel_k(self.bn_rel_k(emb_rel_k))))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Dropout on (4.1).
            # (4.3) Apply BN + DP on ALL entities.
            # (4.4) Inner product
            real_score = torch.mm(self.hidden_dp_real(conv_real * r_val),
                                  self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)).transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(conv_imag_i * i_val),
                               self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight)).transpose(1, 0))
            j_score = torch.mm(self.hidden_dp_j(conv_imag_j * j_val),
                               self.input_dp_ent_j(self.bn_ent_j(self.emb_ent_j.weight)).transpose(1, 0))
            k_score = torch.mm(self.hidden_dp_k(conv_imag_k * k_val),
                               self.input_dp_ent_k(self.bn_ent_k(self.emb_ent_k.weight)).transpose(1, 0))
        score = real_score + i_score + j_score + k_score
        return torch.sigmoid(score)

    def forward_tail_batch(self, r_idx, e2_idx):
        """
        Completed.
        """
        # (1)
        # (1.1) Quaternion embeddings of relations.
        emb_rel_real = self.emb_rel_real(r_idx)
        emb_rel_i = self.emb_rel_i(r_idx)
        emb_rel_j = self.emb_rel_j(r_idx)
        emb_rel_k = self.emb_rel_k(r_idx)
        # (1.2) Quaternion embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)
        emb_tail_j = self.emb_ent_j(e2_idx)
        emb_tail_k = self.emb_ent_k(e2_idx)
        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k),
                                        Q_2=(emb_tail_real, emb_tail_i, emb_tail_j, emb_tail_k))
        conv_real, conv_i, conv_j, conv_k = Q_3
        # (3)
        # (3.1) Reshape (1.2) tail entities.
        emb_tail_real = emb_tail_real.view(-1, self.embedding_dim, 1)
        emb_tail_i = emb_tail_i.view(-1, self.embedding_dim, 1)
        emb_tail_j = emb_tail_j.view(-1, self.embedding_dim, 1)
        emb_tail_k = emb_tail_k.view(-1, self.embedding_dim, 1)
        # (3.2) Reshape (2) output of convolution.
        conv_real = conv_real.view(-1, 1, self.embedding_dim)
        conv_i = conv_i.view(-1, 1, self.embedding_dim)
        conv_j = conv_j.view(-1, 1, self.embedding_dim)
        conv_k = conv_k.view(-1, 1, self.embedding_dim)
        if self.flag_hamilton_mul_norm:
            # (4) Reshape (1.1)-relations to broadcast quaternion multiplication.
            emb_rel_real = emb_rel_real.view(-1, 1, self.embedding_dim)
            emb_rel_i = emb_rel_i.view(-1, 1, self.embedding_dim)
            emb_rel_j = emb_rel_j.view(-1, 1, self.embedding_dim)
            emb_rel_k = emb_rel_k.view(-1, 1, self.embedding_dim)
            # (5) Quaternion multiplication of ALL entities and unit normalized (4).
            r_val, i_val, j_val, k_val = quaternion_mul_with_unit_norm(
                Q_1=(self.emb_ent_real.weight, self.emb_ent_i.weight,
                     self.emb_ent_j.weight, self.emb_ent_k.weight),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (6)
            # (6.1) Hadamard product of (3.2)-reshaped conv. with (5).
            # (6.2) Inner product of (6.1) and (3.1)-reshaped tails.
            real_score = torch.matmul(conv_real * r_val, emb_tail_real)
            i_score = torch.matmul(conv_i * i_val, emb_tail_i)
            j_score = torch.matmul(conv_j * j_val, emb_tail_j)
            k_score = torch.matmul(conv_k * k_val, emb_tail_k)
        else:
            # (4) BN + Dropout + Reshape (1.1) relations.
            emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)).view(-1, 1, self.embedding_dim)
            emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)).view(-1, 1, self.embedding_dim)
            emb_rel_j = self.input_dp_rel_j(self.bn_rel_j(emb_rel_j)).view(-1, 1, self.embedding_dim)
            emb_rel_k = self.input_dp_rel_k(self.bn_rel_k(emb_rel_k)).view(-1, 1, self.embedding_dim)
            # (5)
            # (5.1) BN + Dropout on ALL entities.
            # (5.2) Quaternion multiplication of (5.1) and (4).
            r_val, i_val, j_val, k_val = quaternion_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)),
                     self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight)),
                     self.input_dp_ent_j(self.bn_ent_i(self.emb_ent_j.weight)),
                     self.input_dp_ent_k(self.bn_ent_k(self.emb_ent_k.weight))),
                Q_2=(emb_rel_real, emb_rel_i, emb_rel_j, emb_rel_k))
            # (6)
            # (6.1) Hadamard product of (3.2) and (5).
            # (6.2) Dropout on (6.1).
            # (6.3) Inner product on (6.2) with (3.1).
            real_score = torch.matmul(self.hidden_dp_real(conv_real * r_val), emb_tail_real)
            i_score = torch.matmul(self.hidden_dp_i(conv_i * i_val), emb_tail_i)
            j_score = torch.matmul(self.hidden_dp_j(conv_j * j_val), emb_tail_j)
            k_score = torch.matmul(self.hidden_dp_k(conv_k * k_val), emb_tail_k)
        score = torch.sigmoid(real_score + i_score + j_score + k_score)
        score = score.squeeze()
        return score

    def forward_head_and_loss(self, h_idx, r_idx, targets):
        return self.loss(self.forward_head_batch(h_idx=h_idx, r_idx=r_idx), targets)

    def forward_tail_and_loss(self, r_idx, e2_idx, targets):
        return self.loss(self.forward_tail_batch(r_idx=r_idx, e2_idx=e2_idx), targets)

    def init(self):
        nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        nn.init.xavier_normal_(self.emb_ent_i.weight.data)
        nn.init.xavier_normal_(self.emb_ent_j.weight.data)
        nn.init.xavier_normal_(self.emb_ent_k.weight.data)

        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_i.weight.data)
        nn.init.xavier_normal_(self.emb_rel_j.weight.data)
        nn.init.xavier_normal_(self.emb_rel_k.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data,
                                self.emb_ent_i.weight.data,
                                self.emb_ent_j.weight.data,
                                self.emb_ent_k.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data,
                             self.emb_rel_i.weight.data,
                             self.emb_rel_j.weight.data,
                             self.emb_rel_k.weight.data), 1)

        return entity_emb, rel_emb
