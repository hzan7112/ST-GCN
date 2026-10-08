import torch
import torch.nn as nn
import torch.nn.functional as F


RADIAL_BRANCHES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
    [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21],
    [2, 22], [22, 23], [23, 24], [5, 25], [25, 26], [26, 27],
    [27, 28], [28, 29], [29, 30], [30, 31], [31, 32]
]


def sanitize_edge_list(edge_list):
    return [[int(e[0]), int(e[1])] for e in edge_list]


class FirstOrderNonlinearGCNLayer(nn.Module):
    def __init__(self, hidden_dim, use_initial_anchor=True, use_residual=True):
        super().__init__()
        self.use_initial_anchor = use_initial_anchor
        self.use_residual = use_residual

        self.gcn_linear = nn.Linear(hidden_dim, hidden_dim)

        if use_initial_anchor:
            self.anchor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.anchor_linear = None
            self.alpha_raw = None

        if use_residual:
            self.beta_raw = nn.Parameter(torch.tensor(0.0))
        else:
            self.beta_raw = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gcn_linear.weight)
        nn.init.zeros_(self.gcn_linear.bias)
        if self.anchor_linear is not None:
            nn.init.xavier_uniform_(self.anchor_linear.weight)

    def forward(self, H, H0, A_norm):
        Z = self.gcn_linear(torch.matmul(A_norm, H))

        if self.use_initial_anchor:
            Z = Z + torch.sigmoid(self.alpha_raw) * self.anchor_linear(H0)

        H_out = F.relu(Z)

        if self.use_residual:
            H_out = H_out + torch.sigmoid(self.beta_raw) * H

        return H_out, Z


class DeepFirstOrderGCN(nn.Module):
    def __init__(
        self,
        in_features=2,
        hidden_dim=64,
        num_layers=9,
        K=None,
        edge_list=None,
        num_nodes=33,
        node_relu_dim=48,
        edge_relu_dim=48,
        node_emb_dim=16,
        edge_emb_dim=16,
        use_residual=True,
        use_initial_anchor=True,
        use_jk=True,
        include_input_in_jk=True,
        use_linear_skip=True,
    ):
        super().__init__()

        edge_list = sanitize_edge_list(edge_list if edge_list is not None else RADIAL_BRANCHES)

        if K is not None:
            try:
                num_layers = max(num_layers, int(K))
            except Exception:
                pass

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = 1
        self.requested_K = K
        self.num_nodes = num_nodes
        self.edge_list = edge_list
        self.num_edges = len(edge_list)

        self.node_relu_dim = node_relu_dim
        self.edge_relu_dim = edge_relu_dim
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim

        self.use_residual = use_residual
        self.use_initial_anchor = use_initial_anchor
        self.use_jk = use_jk
        self.include_input_in_jk = include_input_in_jk
        self.use_linear_skip = use_linear_skip

        self.src_indices = [e[0] for e in edge_list]
        self.dst_indices = [e[1] for e in edge_list]

        self.register_buffer("src_indices_tensor", torch.tensor(self.src_indices, dtype=torch.long))
        self.register_buffer("dst_indices_tensor", torch.tensor(self.dst_indices, dtype=torch.long))
        self.register_buffer("A_norm", self._build_adj(edge_list, num_nodes))

        self.input_embed = nn.Linear(in_features, hidden_dim)

        self.gcn_layers = nn.ModuleList([
            FirstOrderNonlinearGCNLayer(
                hidden_dim=hidden_dim,
                use_initial_anchor=use_initial_anchor,
                use_residual=use_residual,
            )
            for _ in range(num_layers)
        ])

        if use_jk:
            jk_blocks = num_layers + 1 if include_input_in_jk else num_layers
            self.readout_dim = hidden_dim * jk_blocks
        else:
            self.readout_dim = hidden_dim

        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.edge_emb = nn.Embedding(self.num_edges, edge_emb_dim)

        node_input_dim = self.readout_dim + node_emb_dim
        edge_input_dim = self.readout_dim * 3 + edge_emb_dim

        self.node_hidden = nn.Linear(node_input_dim, node_relu_dim)
        self.node_out = nn.Linear(node_relu_dim, 1)
        self.node_skip = nn.Linear(node_input_dim, 1) if use_linear_skip else None

        self.edge_hidden = nn.Linear(edge_input_dim, edge_relu_dim)
        self.edge_out = nn.Linear(edge_relu_dim, 1)
        self.edge_skip = nn.Linear(edge_input_dim, 1) if use_linear_skip else None

        self.reset_parameters()

    @staticmethod
    def _build_adj(edge_list, num_nodes):
        A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        for u, v in edge_list:
            A[u, v] = 1.0
            A[v, u] = 1.0

        A_hat = A + torch.eye(num_nodes, dtype=torch.float32)
        deg = A_hat.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)

        return torch.diag(deg_inv_sqrt) @ A_hat @ torch.diag(deg_inv_sqrt)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_embed.weight)
        nn.init.zeros_(self.input_embed.bias)

        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.edge_emb.weight, mean=0.0, std=0.02)

        for layer in [self.node_hidden, self.node_out, self.edge_hidden, self.edge_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if self.node_skip is not None:
            nn.init.xavier_uniform_(self.node_skip.weight)
            nn.init.zeros_(self.node_skip.bias)

        if self.edge_skip is not None:
            nn.init.xavier_uniform_(self.edge_skip.weight)
            nn.init.zeros_(self.edge_skip.bias)

    def build_jk_feature(self, H0, H_list):
        if not self.use_jk:
            return H_list[-1]
        if self.include_input_in_jk:
            return torch.cat([H0] + H_list, dim=-1)
        return torch.cat(H_list, dim=-1)

    def forward(self, X):
        batch_size = X.size(0)
        device = X.device
        A_norm = self.A_norm.to(device)

        H0 = self.input_embed(X)
        H = H0
        H_list, gcn_Z_list = [], []

        for layer in self.gcn_layers:
            H, Z = layer(H, H0, A_norm)
            H_list.append(H)
            gcn_Z_list.append(Z)

        H_readout = self.build_jk_feature(H0, H_list)

        node_ids = torch.arange(self.num_nodes, device=device)
        node_emb = self.node_emb(node_ids).unsqueeze(0).expand(batch_size, -1, -1)
        H_node = torch.cat([H_readout, node_emb], dim=-1)

        Z_node = self.node_hidden(H_node)
        V_pred = self.node_out(F.relu(Z_node)).squeeze(-1)

        if self.node_skip is not None:
            V_pred = V_pred + self.node_skip(H_node).squeeze(-1)

        H_src = H_readout[:, self.src_indices, :]
        H_dst = H_readout[:, self.dst_indices, :]
        H_diff = H_src - H_dst

        edge_ids = torch.arange(self.num_edges, device=device)
        edge_emb = self.edge_emb(edge_ids).unsqueeze(0).expand(batch_size, -1, -1)

        H_edge = torch.cat([H_src, H_dst, H_diff, edge_emb], dim=-1)

        Z_edge = self.edge_hidden(H_edge)
        I_pred = self.edge_out(F.relu(Z_edge)).squeeze(-1)

        if self.edge_skip is not None:
            I_pred = I_pred + self.edge_skip(H_edge).squeeze(-1)

        return V_pred, I_pred, gcn_Z_list, Z_node, Z_edge

    def get_frozen_adj_norm(self):
        return self.A_norm.detach().cpu().numpy()

    def get_frozen_adj_powers(self):
        return self.A_norm.detach().unsqueeze(0).cpu().numpy()

    def get_frozen_node_emb(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            ids = torch.arange(self.num_nodes, device=self.node_emb.weight.device)
            out = self.node_emb(ids).cpu().numpy()
        if was_training:
            self.train()
        return out

    def get_frozen_edge_emb(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            ids = torch.arange(self.num_edges, device=self.edge_emb.weight.device)
            out = self.edge_emb(ids).cpu().numpy()
        if was_training:
            self.train()
        return out

    def get_binary_count(self):
        gcn_binary = self.num_layers * self.num_nodes * self.hidden_dim
        node_head_binary = self.num_nodes * self.node_relu_dim
        edge_head_binary = self.num_edges * self.edge_relu_dim

        return {
            "gcn_binary": gcn_binary,
            "node_head_binary": node_head_binary,
            "edge_head_binary": edge_head_binary,
            "total_binary": gcn_binary + node_head_binary + edge_head_binary,
        }


StandardGCN = DeepFirstOrderGCN
GCNModel = DeepFirstOrderGCN
MultiOrderNonlinearGCN = DeepFirstOrderGCN