import torch
import torch.nn as nn
from model.SGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)



class SGCRN_RD(nn.Module):
    def __init__(self, args, precomputed_embeddings):
        super(SGCRN_RD, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        if precomputed_embeddings is not None:
            embeddings_tensor = torch.as_tensor(precomputed_embeddings, dtype=torch.float32)
            self.node_embeddings = nn.Parameter(embeddings_tensor, requires_grad=False)
        else:
            self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)

        # Two Encoders for Residual Learning
        self.encoder1 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                 args.embed_dim, args.num_layers)
        self.encoder2 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                 args.embed_dim, args.num_layers)

        # CNN-based Predictors
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))
        #self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, 12))
        self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, min(12, self.hidden_dim)))
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))

    def forward(self, source):
        #  Encoder for Major Trend
        init_state1 = self.encoder1.init_hidden(source.shape[0])
        output1, _ = self.encoder1(source, init_state1, self.node_embeddings)
        output1 = output1[:, -1:, :, :]
        output1 = self.end_conv1(output1)

        #  Compute Residual
        source1 = self.end_conv2(output1.permute(0, 3, 2, 1))  # Reshape before Conv2D
        source2 = source - source1  # Residual Component
        #source2 = (1 - alpha) * (source - source1) + alpha * target  # combining residual and trend
        # Step 3: Second Encoder for Residual Learning
        init_state2 = self.encoder2.init_hidden(source2.shape[0])
        output2, _ = self.encoder2(source2, init_state2, self.node_embeddings)
        output2 = output2[:, -1:, :, :]
        output2 = self.end_conv3(output2)

        return output1 + output2

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.SGCRNCell import AGCRNCell

# # Learnable Trend Extraction using Dilated Convolution
# class LearnableTrend(nn.Module):
#     def __init__(self, c_in, kernel_size=3, dilation=2):
#         super(LearnableTrend, self).__init__()
#         self.conv = nn.Conv1d(c_in, c_in, kernel_size=kernel_size, padding='same', dilation=dilation)

#     def forward(self, x):
#         return self.conv(x)  # Learnable trend extraction

# # Chebyshev Polynomial-based Graph Convolution
# class ChebyshevGraphConv(nn.Module):
#     def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
#         super(ChebyshevGraphConv, self).__init__()
#         self.cheb_k = cheb_k
#         self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
#         self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

#     def forward(self, x, node_embeddings):
#         node_num = node_embeddings.shape[0]
#         supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
#         support_set = [torch.eye(node_num).to(supports.device), supports]
#         for k in range(2, self.cheb_k):
#             support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
#         supports = torch.stack(support_set, dim=0)
#         weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
#         bias = torch.matmul(node_embeddings, self.bias_pool)
#         x_g = torch.einsum("knm,bmc->bknc", supports, x)
#         x_g = x_g.permute(0, 2, 1, 3)
#         x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
#         return x_gconv

# class AVWDCRNN(nn.Module):
#     def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
#         super(AVWDCRNN, self).__init__()
#         assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
#         self.node_num = node_num
#         self.input_dim = dim_in
#         self.num_layers = num_layers
#         self.dcrnn_cells = nn.ModuleList()
#         self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
#         for _ in range(1, num_layers):
#             self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

#     def forward(self, x, init_state, node_embeddings):
#         assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
#         seq_length = x.shape[1]
#         current_inputs = x
#         output_hidden = []
#         for i in range(self.num_layers):
#             state = init_state[i]
#             inner_states = []
#             for t in range(seq_length):
#                 state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
#                 inner_states.append(state)
#             output_hidden.append(state)
#             current_inputs = torch.stack(inner_states, dim=1)
#         return current_inputs, output_hidden

#     def init_hidden(self, batch_size):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
#         return torch.stack(init_states, dim=0)

# class SGCRN_RD(nn.Module):
#     def __init__(self, args, precomputed_embeddings):
#         super(SGCRN_RD, self).__init__()
#         self.num_nodes = args.num_nodes
#         self.input_dim = args.input_dim
#         self.hidden_dim = args.rnn_units
#         self.output_dim = args.output_dim
#         self.horizon = args.horizon
#         self.num_layers = args.num_layers
        
#         self.node_embeddings = nn.Parameter(torch.tensor(precomputed_embeddings, dtype=torch.float32), requires_grad=False)
        
#         # Trend Extraction
#         self.learnable_trend = LearnableTrend(args.input_dim)
        
#         # Graph-based Residual Processing using Chebyshev Approximation
#         self.residual_cheb = ChebyshevGraphConv(args.input_dim, args.input_dim, args.cheb_k, args.embed_dim)
        
#         # Two Encoders for Residual Learning
#         self.encoder1 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
#                                  args.embed_dim, args.num_layers)
#         self.encoder2 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
#                                  args.embed_dim, args.num_layers)
        
#         # Adaptive Weighting for Trend & Residual
#         self.alpha_layer = nn.Linear(args.input_dim, args.input_dim)
        
#         # CNN-based Predictors
#         self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))
#         self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, min(12, self.hidden_dim)))
#         self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))

#     def forward(self, source):
#         # Decompose input into trend and residual
#         trend = self.learnable_trend(source)
#         residual = source - trend
        
#         # Adaptive weighting
#         alpha = torch.sigmoid(self.alpha_layer(trend))
#         beta = 1 - alpha
#         weighted_trend = alpha * trend
#         weighted_residual = beta * residual
        
#         # Process trend with AVWDCRNN
#         init_state1 = self.encoder1.init_hidden(weighted_trend.shape[0])
#         output1, _ = self.encoder1(weighted_trend, init_state1, self.node_embeddings)
#         output1 = output1[:, -1:, :, :]
#         output1 = self.end_conv1(output1)
        
#         # Process residual with Chebyshev Approximation + AVWDCRNN
#         residuals_cheb = self.residual_cheb(weighted_residual, self.node_embeddings)
#         init_state2 = self.encoder2.init_hidden(residuals_cheb.shape[0])
#         output2, _ = self.encoder2(residuals_cheb, init_state2, self.node_embeddings)
#         output2 = output2[:, -1:, :, :]
#         output2 = self.end_conv3(output2)
        
#         return output1 + output2
        
    def update_embeddings(self, new_embeddings):
        """ Update node embeddings dynamically """
        self.node_embeddings.data = torch.tensor(new_embeddings, dtype=torch.float32)
        print("Node embeddings updated!")
        
    def set_embedding_trainable(self, trainable):
        """
        Dynamically update the trainability of node_embeddings.
        Args:
            trainable (bool): Whether to allow gradients to update node_embeddings.
        """
        self.node_embeddings.requires_grad = trainable
        if trainable:
            print("Node embeddings are now trainable.")
        else:
            print("Node embeddings are now frozen.")