import torch
import torch.nn.functional as F
from torch import nn

from toolbox.nn.ComplexEmbedding import ComplexEmbedding, ComplexDropout, ComplexBatchNorm1d, ComplexScoringAll, ComplexMult


class CoreRotatE(nn.Module):
    def __init__(self, gamma, embedding_range):
        super(CoreRotatE, self).__init__()
        self.gamma = gamma
        self.embedding_range = embedding_range

    def forward(self, head, relation, tail, mode='tail-batch'):
        """
        d_e == d_r + d_r
        head:     matrix of shape B x 1 x d_e
        relation: matrix of shape B x 1 x d_r
        tail:     matrix of shape B x E x d_e
        """
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, gamma=12.0, hidden_dropout=0.2):
        super(RotatE, self).__init__()
        self.embedding_dim = embedding_dim
        self.E = nn.Embedding(num_entities, 2 * embedding_dim)
        self.R = nn.Embedding(num_relations, embedding_dim)

        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / embedding_dim]), requires_grad=False)

        self.core = CoreRotatE(self.gamma, self.embedding_range)

        self.dropout = nn.Dropout(hidden_dropout)
        self.b = nn.Parameter(torch.zeros(num_entities))
        self.m = nn.PReLU()

        self.loss = nn.BCELoss()

    def init(self):
        nn.init.uniform_(
            tensor=self.E.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.R.weight.data,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def forward(self, h_idx, r_idx):
        B = h_idx.size(0)

        h = self.E(h_idx).view(B, 1, -1)  # B x 1 x d_e
        r = self.R(r_idx).view(B, 1, -1)  # B x 1 x d_r
        E = self.dropout(self.E.weight.repeat(B, 1, 1))  # B x E x d_e

        x = self.core(h, r, E)

        x = x + self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x  # batch_size x E


def rotate_mul_with_unit_norm(Q_1, Q_2):
    a_h, b_h = Q_1  # = {a_h + b_h i : a_r, b_r \in R^k}
    a_r, b_r = Q_2  # = {a_r + b_r i : a_r, b_r \in R^k}

    # Normalize the relation to eliminate the scaling effect
    denominator = torch.sqrt(a_r ** 2 + b_r ** 2)
    p = a_r / denominator
    q = b_r / denominator
    #  Q'=E Hamilton product R
    r_val = a_h * p - b_h * q
    i_val = a_h * q + b_h * p
    return r_val, i_val


def rotate_mul(Q_1, Q_2):
    a_h, b_h = Q_1  # = {a_h + b_h i : a_r, b_r \in R^k}
    a_r, b_r = Q_2  # = {a_r + b_r i : a_r, b_r \in R^k}
    r_val = a_h * a_r - b_h * b_r
    i_val = a_h * b_r + b_h * a_r
    return r_val, i_val



class RotateMult2(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3):
        super(RotateMult2, self).__init__()
        self.name = 'QMult'
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.flag_hamilton_mul_norm = norm_flag
        self.E = ComplexEmbedding(self.num_entities, self.embedding_dim, 2)
        self.R = ComplexEmbedding(self.num_relations, self.embedding_dim, 2)
        self.E_dropout = ComplexDropout([input_dropout, input_dropout])
        self.R_dropout = ComplexDropout([input_dropout, input_dropout])
        self.E_bn = ComplexBatchNorm1d(self.embedding_dim, 2)

        self.mul = ComplexMult(norm_flag)

        self.bce = nn.BCELoss()
        self.b1 = nn.Parameter(torch.zeros(num_entities))
        self.b2 = nn.Parameter(torch.zeros(num_entities))
        self.scoring_all = ComplexScoringAll()

    def forward(self, h_idx, r_idx):
        return self.forward_head_batch(h_idx.view(-1), r_idx.view(-1))

    def forward_head_batch(self, h_idx, r_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        h = self.E(h_idx)
        r = self.R(r_idx)

        t = self.mul(h, r)

        if self.flag_hamilton_mul_norm:
            score_a, score_b = self.scoring_all(t, self.E.get_embeddings())  # a + b i
        else:
            score_a, score_b = self.scoring_all(self.E_dropout(t), self.E_dropout(self.E_bn(self.E.get_embeddings())))
        score_a = score_a + self.b1.expand_as(score_a)
        score_b = score_b + self.b2.expand_as(score_b)

        y_a = torch.sigmoid(score_a)
        y_b = torch.sigmoid(score_b)

        return y_a, y_b

    def loss(self, target, y):
        y_a, y_b = target
        return self.bce(y_a, y) + self.bce(y_b, y)

    def init(self):
        self.E.init()
        self.R.init()


class RotateMult(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3):
        super(RotateMult, self).__init__()
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
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(r_idx)
        emb_rel_i = self.emb_rel_i(r_idx)

        if self.flag_hamilton_mul_norm:
            # (2) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val = rotate_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i),
                Q_2=(emb_rel_real, emb_rel_i))
            # (3) Inner product of (2) with all entities.
            real_score = torch.mm(r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(i_val, self.emb_ent_i.weight.transpose(1, 0))
        else:
            # (2)
            # (2.1) Apply BN + Dropout on (1.2)-relations.
            # (2.2) Apply quaternion multiplication on (1.1) and (2.1).
            r_val, i_val = rotate_mul(
                Q_1=(emb_head_real, emb_head_i),
                Q_2=(self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                     self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)),))
            # (3)
            # (3.1) Dropout on (2)-result of quaternion multiplication.
            # (3.2) Apply BN + DP on ALL entities.
            # (3.3) Inner product
            real_score = torch.mm(self.hidden_dp_real(r_val),
                                  self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)).transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(i_val),
                               self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight)).transpose(1, 0))
        score = real_score + i_score
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
        # (1.2)  Reshape Quaternion embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx).view(-1, self.embedding_dim, 1)
        emb_tail_i = self.emb_ent_i(e2_idx).view(-1, self.embedding_dim, 1)
        if self.flag_hamilton_mul_norm:
            # (2) Reshape (1.1)-relations.
            emb_rel_real = emb_rel_real.view(-1, 1, self.embedding_dim)
            emb_rel_i = emb_rel_i.view(-1, 1, self.embedding_dim)
            # (3) Quaternion multiplication of ALL entities and unit normalized (2).
            r_val, i_val = rotate_mul_with_unit_norm(
                Q_1=(self.emb_ent_real.weight, self.emb_ent_i.weight),
                Q_2=(emb_rel_real, emb_rel_i))
            # (4) Inner product of (3) with (1.2).
            real_score = torch.matmul(r_val, emb_tail_real)
            i_score = torch.matmul(i_val, emb_tail_i)
        else:
            # (2) BN + Dropout + Reshape (1.1)-relations.
            emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)).view(-1, 1, self.embedding_dim)
            emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)).view(-1, 1, self.embedding_dim)

            # (3)
            # (3.1) BN + Dropout on ALL entities.
            # (3.2) Quaternion multiplication of (3.1) and (2).
            r_val, i_val = rotate_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)),
                     self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight))),
                Q_2=(emb_rel_real, emb_rel_i))

            # (4)
            # (4.1) Dropout on (3).
            # (4.2) Inner product on (4.1) with (1.2).
            real_score = torch.matmul(self.hidden_dp_real(r_val), emb_tail_real)
            i_score = torch.matmul(self.hidden_dp_i(i_val), emb_tail_i)
        score = torch.sigmoid(real_score + i_score)
        score = score.squeeze()
        return score

    def forward_head_and_loss(self, h_idx, r_idx, targets):
        return self.loss(self.forward_head_batch(h_idx=h_idx, r_idx=r_idx), targets)

    def forward_tail_and_loss(self, r_idx, e2_idx, targets):
        return self.loss(self.forward_tail_batch(r_idx=r_idx, e2_idx=e2_idx), targets)

    def init(self):
        nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        nn.init.xavier_normal_(self.emb_ent_i.weight.data)
        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_i.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)

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
        self.emb_rel_real = nn.Embedding(self.num_relations, self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.num_relations, self.embedding_dim)  # imaginary i
        # Dropouts
        self.input_dp_ent_real = nn.Dropout(input_dropout)
        self.input_dp_ent_i = nn.Dropout(input_dropout)
        self.input_dp_rel_real = nn.Dropout(input_dropout)
        self.input_dp_rel_i = nn.Dropout(input_dropout)
        self.hidden_dp_real = nn.Dropout(hidden_dropout)
        self.hidden_dp_i = nn.Dropout(hidden_dropout)
        # Batch Normalization
        self.bn_ent_real = nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = nn.BatchNorm1d(self.embedding_dim)

        self.bn_rel_real = nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = nn.BatchNorm1d(self.embedding_dim)

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
        emb_ent_real, emb_ent_imag_i = Q_1
        emb_rel_real, emb_rel_imag_i = Q_2
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim)], 2)

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
        # (1.2) Quaternion embeddings of relations
        emb_rel_real = self.emb_rel_real(r_idx)
        emb_rel_i = self.emb_rel_i(r_idx)

        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_head_real, emb_head_i),
                                        Q_2=(emb_rel_real, emb_rel_i))
        conv_real, conv_imag_i = Q_3
        if self.flag_hamilton_mul_norm:
            # (3) Quaternion multiplication of (1.1) and unit normalized (1.2).
            r_val, i_val = rotate_mul_with_unit_norm(
                Q_1=(emb_head_real, emb_head_i),
                Q_2=(emb_rel_real, emb_rel_i))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Inner product of (4.1) with ALL entities.
            real_score = torch.mm(conv_real * r_val, self.emb_ent_real.weight.transpose(1, 0))
            i_score = torch.mm(conv_imag_i * i_val, self.emb_ent_i.weight.transpose(1, 0))
        else:
            # (3)
            # (3.1) Apply BN + Dropout on (1.2).
            # (3.2) Apply quaternion multiplication on (1.1) and (3.1).
            r_val, i_val = rotate_mul(
                Q_1=(emb_head_real, emb_head_i),
                Q_2=(self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)),
                     self.input_dp_rel_i(self.bn_rel_i(emb_rel_i))))
            # (4)
            # (4.1) Hadamard product of (2) with (3).
            # (4.2) Dropout on (4.1).
            # (4.3) Apply BN + DP on ALL entities.
            # (4.4) Inner product
            real_score = torch.mm(self.hidden_dp_real(conv_real * r_val),
                                  self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)).transpose(1, 0))
            i_score = torch.mm(self.hidden_dp_i(conv_imag_i * i_val),
                               self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight)).transpose(1, 0))
        score = real_score + i_score
        return torch.sigmoid(score)

    def forward_tail_batch(self, r_idx, e2_idx):
        """
        Completed.
        """
        # (1)
        # (1.1) Quaternion embeddings of relations.
        emb_rel_real = self.emb_rel_real(r_idx)
        emb_rel_i = self.emb_rel_i(r_idx)
        # (1.2) Quaternion embeddings of tail entities.
        emb_tail_real = self.emb_ent_real(e2_idx)
        emb_tail_i = self.emb_ent_i(e2_idx)
        # (2) Apply convolution operation on (1.1) and (1.2).
        Q_3 = self.residual_convolution(Q_1=(emb_rel_real, emb_rel_i),
                                        Q_2=(emb_tail_real, emb_tail_i))
        conv_real, conv_i, conv_j, conv_k = Q_3
        # (3)
        # (3.1) Reshape (1.2) tail entities.
        emb_tail_real = emb_tail_real.view(-1, self.embedding_dim, 1)
        emb_tail_i = emb_tail_i.view(-1, self.embedding_dim, 1)
        # (3.2) Reshape (2) output of convolution.
        conv_real = conv_real.view(-1, 1, self.embedding_dim)
        conv_i = conv_i.view(-1, 1, self.embedding_dim)
        if self.flag_hamilton_mul_norm:
            # (4) Reshape (1.1)-relations to broadcast quaternion multiplication.
            emb_rel_real = emb_rel_real.view(-1, 1, self.embedding_dim)
            emb_rel_i = emb_rel_i.view(-1, 1, self.embedding_dim)
            # (5) Quaternion multiplication of ALL entities and unit normalized (4).
            r_val, i_val = rotate_mul_with_unit_norm(
                Q_1=(self.emb_ent_real.weight, self.emb_ent_i.weight),
                Q_2=(emb_rel_real, emb_rel_i))
            # (6)
            # (6.1) Hadamard product of (3.2)-reshaped conv. with (5).
            # (6.2) Inner product of (6.1) and (3.1)-reshaped tails.
            real_score = torch.matmul(conv_real * r_val, emb_tail_real)
            i_score = torch.matmul(conv_i * i_val, emb_tail_i)
        else:
            # (4) BN + Dropout + Reshape (1.1) relations.
            emb_rel_real = self.input_dp_rel_real(self.bn_rel_real(emb_rel_real)).view(-1, 1, self.embedding_dim)
            emb_rel_i = self.input_dp_rel_i(self.bn_rel_i(emb_rel_i)).view(-1, 1, self.embedding_dim)
            # (5)
            # (5.1) BN + Dropout on ALL entities.
            # (5.2) Quaternion multiplication of (5.1) and (4).
            r_val, i_val = rotate_mul(
                Q_1=(self.input_dp_ent_real(self.bn_ent_real(self.emb_ent_real.weight)),
                     self.input_dp_ent_i(self.bn_ent_i(self.emb_ent_i.weight))),
                Q_2=(emb_rel_real, emb_rel_i))
            # (6)
            # (6.1) Hadamard product of (3.2) and (5).
            # (6.2) Dropout on (6.1).
            # (6.3) Inner product on (6.2) with (3.1).
            real_score = torch.matmul(self.hidden_dp_real(conv_real * r_val), emb_tail_real)
            i_score = torch.matmul(self.hidden_dp_i(conv_i * i_val), emb_tail_i)
        score = torch.sigmoid(real_score + i_score)
        score = score.squeeze()
        return score

    def forward_head_and_loss(self, h_idx, r_idx, targets):
        return self.loss(self.forward_head_batch(h_idx=h_idx, r_idx=r_idx), targets)

    def forward_tail_and_loss(self, rel_idx, e2_idx, targets):
        return self.loss(self.forward_tail_batch(rel_idx=rel_idx, e2_idx=e2_idx), targets)

    def init(self):
        nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        nn.init.xavier_normal_(self.emb_ent_i.weight.data)

        nn.init.xavier_normal_(self.emb_rel_real.weight.data)
        nn.init.xavier_normal_(self.emb_rel_i.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)

        return entity_emb, rel_emb


def negative_sample_loss(model,
                         positive_sample, negative_sample, subsampling_weight, mode,
                         single_mode="single"):
    negative_score = model((positive_sample, negative_sample), mode=mode)
    negative_score = F.logsigmoid(-negative_score).mean(dim=1)

    positive_score = model(positive_sample, mode=single_mode)
    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

    positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
    negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
    return (positive_sample_loss + negative_sample_loss) / 2


def train_step(model, optimizer,
               positive_sample, negative_sample, subsampling_weight, mode,
               align_positive_sample, align_negative_sample, align_subsampling_weight, align_mode,
               device="cuda"):
    model.train()
    optimizer.zero_grad()

    positive_sample = positive_sample.to(device)
    negative_sample = negative_sample.to(device)
    subsampling_weight = subsampling_weight.to(device)
    if align_mode is not None:
        align_positive_sample = align_positive_sample.to(device)
        align_negative_sample = align_negative_sample.to(device)
        align_subsampling_weight = align_subsampling_weight.to(device)

    raw_loss = model.loss(model,
                          positive_sample, negative_sample, subsampling_weight,
                          mode, "single")
    if align_mode is not None:
        align_loss = model.loss(model,
                                align_positive_sample, align_negative_sample, align_subsampling_weight,
                                align_mode, "align-single")
    else:
        align_loss = raw_loss

    loss = (raw_loss + align_loss) / 2
    loss.backward()
    optimizer.step()

    return loss.item(), raw_loss.item(), align_loss.item()
