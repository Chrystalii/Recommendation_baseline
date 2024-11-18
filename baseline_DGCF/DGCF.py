import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class DGCF(nn.Module):
    def __init__(self, args, data_config):
        super(DGCF, self).__init__()
        # argument settings
        self.device = args.device
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 1
        self.all_h_list = torch.LongTensor(data_config['all_h_list']).to(self.device)
        self.all_t_list = torch.LongTensor(data_config['all_t_list']).to(self.device)

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.n_factors = args.n_factors
        self.n_iterations = args.n_iterations
        self.n_layers = args.n_layers
        self.pick_level = args.pick_scale
        self.cor_flag = args.cor_flag
        if args.pick == 1:
            self.is_pick = True
        else:
            self.is_pick = False

        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.n_factors = args.n_factors
        self.n_iterations = args.n_iterations
        self.n_layers = args.n_layers
        self.pick_level = args.pick_scale
        self.cor_flag = args.cor_flag
        self.cor_weight = args.corDecay

        if args.pick == 1:
            self.is_pick = True
        else:
            self.is_pick = False

        # assign different values with different factors (channels).
        self.A_values = torch.randn(self.n_factors, len(self.all_h_list), requires_grad=True)

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)

        self.user_all_embeddings = nn.Embedding(self.n_users, self.emb_dim)
        self.item_all_embeddings = nn.Embedding(self.n_items, self.emb_dim)

        self._init_matrix()
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        user_all_embeddings, item_all_embeddings = self._create_star_routing_embed_with_P(self.is_pick)
        self.user_all_embeddings.weight.data.copy_(user_all_embeddings)
        self.item_all_embeddings.weight.data.copy_(item_all_embeddings)

    def _init_matrix(self):
        num_edge = len(self.all_h_list)
        edge_ids = torch.arange(num_edge).to(self.device)
        edge2head = torch.stack((self.all_h_list, edge_ids), dim=0).to(self.device)
        head2edge = torch.stack((edge_ids, self.all_h_list), dim=0).to(self.device)
        tail2edge = torch.stack((edge_ids, self.all_t_list), dim=0).to(self.device)
        val_one = torch.ones_like(self.all_h_list).float()
        num_node = self.n_users + self.n_items
        self.edge2head_mat = self._build_sparse_tensor(
            edge2head, val_one, (num_node, num_edge)
        )
        self.head2edge_mat = self._build_sparse_tensor(
            head2edge, val_one, (num_edge, num_node)
        )
        self.tail2edge_mat = self._build_sparse_tensor(
            tail2edge, val_one, (num_edge, num_node)
        )

    def _build_sparse_tensor(self, indices, values, size):
        # Construct the sparse matrix with indices, values and size.
        return torch.sparse_coo_tensor(indices, values, size).to(self.device)

    def _create_star_routing_embed_with_P(self, pick_=False):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings.unsqueeze(1)]
        # initialize with every factor value as 1
        A_values = torch.ones((len(self.all_h_list), self.n_factors)).to(self.device)
        A_values.requires_grad_(True)
        for k in range(self.n_layers):
            layer_embeddings = []
            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-length list of embeddings
            # [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = torch.chunk(ego_embeddings, self.n_factors, 1)
            for t in range(0, self.n_iterations):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self.build_matrix(A_values=A_values)
                for i in range(0, self.n_factors):
                    # update the embeddings via simplified graph convolution layer
                    edge_weight = factor_edge_weight[i]
                    # (num_edge, 1)
                    edge_val = torch.sparse.mm(
                        self.tail2edge_mat, ego_layer_embeddings[i].to(self.device)
                    )
                    # (num_edge, dim / n_factors)
                    edge_val = edge_val * edge_weight
                    # (num_edge, dim / n_factors)
                    factor_embeddings = torch.sparse.mm(self.edge2head_mat, edge_val)
                    # (num_node, num_edge) (num_edge, dim) -> (num_node, dim)

                    iter_embeddings.append(factor_embeddings)

                    if t == self.n_iterations - 1:
                        layer_embeddings = iter_embeddings

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    head_factor_embeddings = torch.index_select(
                        factor_embeddings, dim=0, index=self.all_h_list
                    )
                    tail_factor_embeddings = torch.index_select(
                        ego_layer_embeddings[i].to(self.device), dim=0, index=self.all_t_list
                    )

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    # to adapt to torch version
                    head_factor_embeddings = F.normalize(
                        head_factor_embeddings, p=2, dim=1
                    )
                    tail_factor_embeddings = F.normalize(
                        tail_factor_embeddings, p=2, dim=1
                    )
                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [num_edge, 1]
                    A_factor_values = torch.sum(
                        head_factor_embeddings * torch.tanh(tail_factor_embeddings),
                        dim=1,
                        keepdim=True,
                    )
                    # update the attentive weights
                    A_iter_values.append(A_factor_values)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                # (num_edge, n_factors)
                # add all layer-wise attentive weights up.
                A_values = A_values + A_iter_values

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = torch.cat(layer_embeddings, dim=1)

            ego_embeddings = side_embeddings
            # concatenate outputs of all layers
            all_embeddings += [ego_embeddings.unsqueeze(1)]

        all_embeddings = [embeddings.to(self.device) for embeddings in all_embeddings]
        res_embeddings = torch.cat(all_embeddings, dim=1)
        res_embeddings = torch.mean(res_embeddings, dim=1, keepdim=False)
        # all_embeddings = torch.cat(all_embeddings, dim=1)
        # (num_node, n_layer + 1, embedding_size)
        # all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        # (num_node, embedding_size)

        u_g_embeddings = res_embeddings[: self.n_users, :]
        i_g_embeddings = res_embeddings[self.n_users:, :]

        return u_g_embeddings, i_g_embeddings

    def build_matrix(self, A_values):
        norm_A_values = F.softmax(A_values, dim=1)
        factor_edge_weight = []
        for i in range(self.n_factors):
            tp_values = norm_A_values[:, i].unsqueeze(1)
            # (num_edge, 1)
            d_values = torch.sparse.mm(self.edge2head_mat, tp_values)
            # (num_node, num_edge) (num_edge, 1) -> (num_node, 1)
            d_values = torch.clamp(d_values, min=1e-8)
            try:
                assert not torch.isnan(d_values).any()
            except AssertionError:
                self.logger.info("d_values", torch.min(d_values), torch.max(d_values))

            d_values = 1.0 / torch.sqrt(d_values)
            head_term = torch.sparse.mm(self.head2edge_mat, d_values)
            # (num_edge, num_node) (num_node, 1) -> (num_edge, 1)

            tail_term = torch.sparse.mm(self.tail2edge_mat, d_values)
            edge_weight = tp_values * head_term * tail_term
            factor_edge_weight.append(edge_weight)
        return factor_edge_weight

    def forward(self, users, pos_items, neg_items, cor_users, cor_items):
        u_embeddings = self.user_all_embeddings(users)
        pos_embeddings = self.item_all_embeddings(pos_items)
        neg_embeddings = self.item_all_embeddings(neg_items)

        mf_loss = self.create_bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)

        # cul regularized
        u_ego_embeddings = self.user_embedding(users)
        pos_ego_embeddings = self.item_embedding(pos_items)
        neg_ego_embeddings = self.item_embedding(neg_items)
        emb_loss = self.create_emb_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        if self.cor_weight < 1e-9:
            cor_loss = 0
        else:
            cor_u_embeddings = self.user_all_embeddings(cor_users)
            cor_i_embeddings = self.item_all_embeddings(cor_items)
            cor_loss = self.cor_weight * self.create_cor_loss(cor_u_embeddings, cor_i_embeddings)
        loss = mf_loss + emb_loss + cor_loss
        return mf_loss, emb_loss, cor_loss, loss

    def create_bpr_loss(self, user_embeddings, pos_item_embeddings, neg_item_embeddings):
        pos_scores = torch.mul(user_embeddings, pos_item_embeddings).sum(dim=1)
        neg_scores = torch.mul(user_embeddings, neg_item_embeddings).sum(dim=1)
        mf_loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))
        return mf_loss

    def create_emb_loss(self, user_embeddings, pos_item_embeddings, neg_item_embeddings):
        regularizer = (torch.norm(user_embeddings, p=2) ** 2 +
                       torch.norm(pos_item_embeddings, p=2) ** 2 +
                       torch.norm(neg_item_embeddings, p=2) ** 2)
        emb_loss = self.decay * regularizer / self.batch_size
        return emb_loss

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        cor_loss = None

        ui_embeddings = torch.cat((cor_u_embeddings, cor_i_embeddings), dim=0)
        ui_factor_embeddings = torch.chunk(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors - 1):
            x = ui_factor_embeddings[i]
            # (M + N, emb_size / n_factor)
            y = ui_factor_embeddings[i + 1]
            # (M + N, emb_size / n_factor)
            if i == 0:
                cor_loss = self._create_distance_correlation(x, y)
            else:
                cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= (self.n_factors + 1.0) * self.n_factors / 2

        return cor_loss

    def _create_distance_correlation(self, X1, X2):
        def _create_centered_distance(X):
            """
            X: (batch_size, dim)
            return: X - E(X)
            """
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            r = torch.sum(X * X, dim=1, keepdim=True)
            # (N, 1)
            # (x^2 - 2xy + y^2) -> l2 distance between all vectors
            value = r - 2 * torch.mm(X, X.T) + r.T
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            D = torch.sqrt(value + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            # matrix - average over row - average over col + average over matrix
            D = (
                    D
                    - torch.mean(D, dim=0, keepdim=True)
                    - torch.mean(D, dim=1, keepdim=True)
                    + torch.mean(D)
            )
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = float(D1.size(0))
            value = torch.sum(D1 * D2) / (n_samples * n_samples)
            zero_value = torch.zeros_like(value)
            value = torch.where(value > 0.0, value, zero_value)
            dcov = torch.sqrt(value + 1e-8)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        value = dcov_11 * dcov_22
        zero_value = torch.zeros_like(value)
        value = torch.where(value > 0.0, value, zero_value)
        dcor = dcov_12 / (torch.sqrt(value) + 1e-10)
        return dcor

    def get_user_info(self, user_batch):
        user_embeds = self.user_all_embeddings(user_batch)
        item_embeds = self.item_all_embeddings.weight
        return torch.matmul(user_embeds, item_embeds.T)

    def test_model(self, args, train_data, test_data, topK):
        self.eval()
        n_test = test_data.shape[0]
        recalls, hit_ratios, ndcgs = [], [], []
        with torch.no_grad():
            for start_idx in range(0, n_test, args.batch_size):
                end_idx = min(start_idx + args.batch_size, n_test)

                X_tr = train_data[start_idx: end_idx]
                X_te = test_data[start_idx: end_idx]
                X_tr = torch.Tensor(X_tr.toarray()).to(self.device)
                X_te = torch.Tensor(X_te.toarray())

                users = [i for i in range(start_idx, end_idx)]
                users = torch.tensor(users).to(self.device)
                X_tr_logits = self.get_user_info(users)

                X_tr_logits[torch.nonzero(X_tr, as_tuple=True)] = float('-inf')
                X_tr_logits = X_tr_logits.cpu()

                recall, hit_ratio, ndcg = utils.compute_matrics(X_tr_logits, X_te, topK)
                recalls.append(recall)
                hit_ratios.append(hit_ratio)
                ndcgs.append(ndcg)

        recalls, hit_ratios, ndcgs = torch.cat(recalls), torch.cat(hit_ratios), torch.cat(ndcgs)

        return recalls.mean(), hit_ratios.mean(), ndcgs.mean()