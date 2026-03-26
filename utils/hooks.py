import torch

def get_hook_q(name, storage):
    def hook(module, input, output):
        # Saves queries for a specific layer name
        storage[f"q_{name}"] = output.detach()
    return hook

def get_hook_k(name, storage):
    def hook(module, input, output):
        # Saves keys for a specific layer name
        storage[f"k_{name}"] = output.detach()
    return hook

# # This version extracts magnitude of the innovations
# def get_res_hook(layer_idx):
#     def hook(module, input, output):
#         # input[0] is the RAW residual stream from the previous layer
#         # It has NOT been normalized by this layer's internal RMSNorm yet.
#         x_raw_in = input[0].detach()
        
#         # output[0] is the RAW residual stream after this layer's 
#         # Attention and MLP have been added to the x_raw_in.
#         x_raw_out = output[0].detach()

#         # Magnitude
#         saved[f"mag_{layer_idx}"] = torch.norm(x_raw_in[0, -1])
        
#         # Innovation
#         saved[f"delta_x_{layer_idx}"] = x_raw_out - x_raw_in
#     return hook

## This version uses cosine similarity of innovation & residual stream
def get_res_hook(layer_idx, storage):
    def hook(module, input, output):
        x_in = input[0].detach()
        x_out = output[0].detach()
        delta_x = x_out - x_in
        
        # Calculate Cosine Similarity
        # We focus on the last token [0, -1, :]
        num = torch.sum(x_in[0, -1] * delta_x[0, -1])
        den = (torch.norm(x_in[0, -1]) * torch.norm(delta_x[0, -1])) + 1e-8
        cos_sim = num / den
        
        storage[f"mag_{layer_idx}"] = torch.norm(x_in[0, -1])
        storage[f"delta_x_{layer_idx}"] = cos_sim.unsqueeze(0)
    return hook

