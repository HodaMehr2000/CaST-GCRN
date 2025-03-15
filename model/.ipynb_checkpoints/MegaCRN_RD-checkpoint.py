'''
Here's a clear, structured English explanation:

 1. **Encoder for Feature Extraction:
- The input data is first passed through an encoder module.  
- This encoder captures both spatial and temporal patterns from the input data.
- The result is a **latent representation**, a condensed, meaningful encoding of the original data.

 2. Memory Gate:
- A memory gate acts like a knowledge bank, containing several learned patterns (memory slots).
- The model uses the output of the encoder to ask the memory, "Have we seen a similar pattern before?"
- Based on the similarity between the encoder’s output and memory slots, the model selects the most relevant pattern and retrieves it. This retrieved vector is the "memory response."

 3. Combining Memory Response and Encoder Output:
- The model then **combines** the latent representation from the encoder with the memory response.
- This combination allows the model to leverage historical patterns, improving the quality of its predictions.

 4. Residual Branch:
- Since the main encoder might miss some subtle or detailed patterns, a **second branch** (the residual branch) separately processes the small, leftover signals (residual information).
- This allows the model to handle details or short-term variations better.

 5. Final Integration and Prediction:
- Finally, outputs from the main branch (enriched by the memory gate) and the residual branch are combined to produce the final prediction.
- The combined prediction captures both long-term patterns and short-term details more effectively.

---

### **In even simpler terms:**
Imagine your model has a notebook (**memory**) where it writes down important patterns it sees. When the model needs to predict something, it first asks itself: "Have I seen something like this before?" It checks its notebook, retrieves the best matching pattern, and uses that information along with the current data to make a more accurate forecast.

This method can significantly enhance the accuracy and robustness of predictions, especially when patterns repeat or have long-term dependencies.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # x shape: [B, T, N, D]
        # init_state shape: [num_layers, B, N, hidden_dim]
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
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class SGCRN_RD(nn.Module):
    def __init__(self, args, precomputed_embeddings):
        super(SGCRN_RD, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        # Initialize node embeddings (smart or random)
        if precomputed_embeddings is not None:
            embeddings_tensor = torch.as_tensor(precomputed_embeddings, dtype=torch.float32)
            self.node_embeddings = nn.Parameter(embeddings_tensor, requires_grad=False)
        else:
            self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)

        # ---------------- Memory Gate Module ----------------
        self.mem_num = args.mem_num if hasattr(args, 'mem_num') else 20
        self.mem_dim = args.mem_dim if hasattr(args, 'mem_dim') else 64
        self.memory = self.construct_memory()
        # Since we are blending by a weighted sum (not concatenation),
        # the blended vector remains of size hidden_dim.
        self.memory_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Learnable weight to balance memory vs. direct encoder output.
        self.memory_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        # -----------------------------------------------------

        # Two encoders (one for trend branch, one for residual branch)
        self.encoder1 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units,
                                  args.cheb_k, args.embed_dim, args.num_layers)
        self.encoder2 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units,
                                  args.cheb_k, args.embed_dim, args.num_layers)

        # CNN-based Predictors (for final forecasting)
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))
        self.end_conv2 = nn.Conv2d(self.hidden_dim, args.horizon * self.output_dim, kernel_size=(1, min(12, self.hidden_dim)))
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        # Memory bank: shape (mem_num, mem_dim)
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        # We define Wq with shape (mem_dim, hidden_dim) so that when transposed,
        # it projects the encoder output (hidden_dim) to mem_dim.
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.mem_dim, self.hidden_dim), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t):
        # h_t: [B, N, hidden_dim]
        # Multiply h_t with Wq^T to get query: [B, N, mem_dim]
        query = torch.matmul(h_t, self.memory['Wq'].T)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].T), dim=-1)  # [B, N, mem_num]
        h_att = torch.matmul(att_score, self.memory['Memory'])  # [B, N, mem_dim]
        return h_att

    def forward(self, source):
        # ---------- Trend Branch with Memory Gate ----------
        init_state1 = self.encoder1.init_hidden(source.shape[0])
        output1, _ = self.encoder1(source, init_state1, self.node_embeddings)  # [B, T, N, hidden_dim]
        trend_hidden = output1[:, -1, :, :]  # [B, N, hidden_dim]
        h_att = self.query_memory(trend_hidden)  # [B, N, mem_dim]

        # Blend memory with encoder output using the learnable weight
        # Weighted sum: result remains of dimension [B, N, hidden_dim]
        h_enriched = self.memory_weight * h_att + (1 - self.memory_weight) * trend_hidden
        h_enriched = self.memory_proj(h_enriched)  # [B, N, hidden_dim]
        trend_output = h_enriched.unsqueeze(1)  # [B, 1, N, hidden_dim]
        trend_output = self.end_conv1(trend_output)

        # ---------- Residual Branch ----------
        # Permute output1 to shape [B, hidden_dim, N, T] for the convolution
        source1 = self.end_conv2(output1.permute(0, 3, 2, 1))
        residual = source - source1  # Residual component (assumes source shape is compatible)
        init_state2 = self.encoder2.init_hidden(residual.shape[0])
        output2, _ = self.encoder2(residual, init_state2, self.node_embeddings)  # [B, T, N, hidden_dim]
        residual_hidden = output2[:, -1, :, :]  # [B, N, hidden_dim]
        residual_output = residual_hidden.unsqueeze(1)  # [B, 1, N, hidden_dim]
        residual_output = self.end_conv3(residual_output)

        # ---------- Final Output ----------
        return trend_output + residual_output

    def update_embeddings(self, new_embeddings):
        self.node_embeddings.data = torch.tensor(new_embeddings, dtype=torch.float32)
        print("Node embeddings updated!")
        
    def set_embedding_trainable(self, trainable):
        self.node_embeddings.requires_grad = trainable
        print(f"Node embeddings trainable: {trainable}")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.AGCRNCell import AGCRNCell

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

#         # Initialize node embeddings
#         if precomputed_embeddings is not None:
#             embeddings_tensor = torch.as_tensor(precomputed_embeddings, dtype=torch.float32)
#             self.node_embeddings = nn.Parameter(embeddings_tensor, requires_grad=False)
#         else:
#             self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)

#         # ---------------- Memory Gate Module ----------------
#         self.mem_num = args.mem_num if hasattr(args, 'mem_num') else 20
#         self.mem_dim = args.mem_dim if hasattr(args, 'mem_dim') else 64
#         self.memory = self.construct_memory()
#         # We'll project from hidden_dim to hidden_dim (after blending memory)
#         self.memory_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
#         # New transformation: project memory response from mem_dim to hidden_dim.
#         self.memory_transform = nn.Linear(self.mem_dim, self.hidden_dim)
#         # Dynamic Memory Utilization Factor (λ) starting low (new: use float 0.1)
#         self.memory_lambda = nn.Parameter(torch.tensor(0.1), requires_grad=False)
#         # -----------------------------------------------------

#         # Two encoders (one for trend branch, one for residual branch)
#         self.encoder1 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim, args.num_layers)
#         self.encoder2 = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim, args.num_layers)

#         # CNN-based Predictors
#         self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))
#         self.end_conv2 = nn.Conv2d(self.hidden_dim, args.horizon * self.output_dim, kernel_size=(1, min(12, self.hidden_dim)))
#         self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim))

#     def construct_memory(self):
#         memory_dict = nn.ParameterDict()
#         memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
#         memory_dict['Wq'] = nn.Parameter(torch.randn(self.hidden_dim, self.mem_dim), requires_grad=True)
#         for param in memory_dict.values():
#             nn.init.xavier_normal_(param)
#         return memory_dict

#     def query_memory(self, h_t):
#         # h_t: [B, N, hidden_dim]
#         query = torch.matmul(h_t, self.memory['Wq'])  # [B, N, mem_dim]
#         att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].T), dim=-1)  # [B, N, mem_num]
#         h_att = torch.matmul(att_score, self.memory['Memory'])  # [B, N, mem_dim]
#         return h_att

#     def forward(self, source):
#         # ---------- Trend Branch with Memory Gate ----------
#         init_state1 = self.encoder1.init_hidden(source.shape[0])
#         output1, _ = self.encoder1(source, init_state1, self.node_embeddings)  # [B, T, N, hidden_dim]
#         trend_hidden = output1[:, -1, :, :]  # [B, N, hidden_dim]

#         # Query memory and apply dynamic λ scaling
#         h_att = self.query_memory(trend_hidden)  # [B, N, mem_dim]
#         # Project memory response to hidden dimension
#         h_att_trans = self.memory_transform(h_att)  # [B, N, hidden_dim]
#         # Blend memory with encoder output using weighted sum
#         h_enriched = self.memory_lambda * h_att_trans + (1 - self.memory_lambda) * trend_hidden  # [B, N, hidden_dim]
#         h_enriched = self.memory_proj(h_enriched)  # [B, N, hidden_dim]
#         trend_output = h_enriched.unsqueeze(1)  # [B, 1, N, hidden_dim]
#         trend_output = self.end_conv1(trend_output)

#         # ---------- Residual Branch ----------
#         # Permute output1 to shape [B, hidden_dim, N, T] for conv2
#         source1 = self.end_conv2(output1.permute(0, 3, 2, 1))
#         residual = source - source1  # Residual component (ensure source shape is compatible)
#         init_state2 = self.encoder2.init_hidden(residual.shape[0])
#         output2, _ = self.encoder2(residual, init_state2, self.node_embeddings)  # [B, T, N, hidden_dim]
#         residual_hidden = output2[:, -1, :, :]  # [B, N, hidden_dim]
#         residual_output = residual_hidden.unsqueeze(1)  # [B, 1, N, hidden_dim]
#         residual_output = self.end_conv3(residual_output)

#         # ---------- Final Output ----------
#         return trend_output + residual_output

#     def increase_memory_usage(self, epoch, total_epochs):
#         """
#         Dynamically increases memory_lambda.
#         - Starts from 0.1 (10% memory usage).
#         - Gradually increases to 1.0 (100% memory usage).
#         """
#         new_lambda = min(1.0, 0.1 + (epoch / total_epochs) * 0.9)
#         with torch.no_grad():
#             self.memory_lambda.copy_(torch.tensor(new_lambda, dtype=torch.float32, device=self.memory_lambda.device))


#     def update_embeddings(self, new_embeddings):
#         self.node_embeddings.data = torch.tensor(new_embeddings, dtype=torch.float32)
#         print("Node embeddings updated!")
        
#     def set_embedding_trainable(self, trainable):
#         self.node_embeddings.requires_grad = trainable
#         print(f"Node embeddings trainable: {trainable}")
