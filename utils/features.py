import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


# Compute Accumulated Evidence for the final token at a given layer
def get_p_tot_log(model, saved, key_suffix=""):
    # Find the right tensors
    q_key = f"q_{key_suffix}" if key_suffix else "q"
    k_key = f"k_{key_suffix}" if key_suffix else "k"
    
    # Setup dimensions
    batch_size, seq_len, _ = saved[q_key].shape
    
    n_heads_q = model.config.num_attention_heads
    n_heads_k = model.config.num_key_value_heads 
    head_dim = model.config.hidden_size // n_heads_q

    # Reshape to (batch, num_heads, seq, head_dim)
    Q = saved[q_key].view(batch_size, seq_len, n_heads_q, head_dim).transpose(1, 2)
    K = saved[k_key].view(batch_size, seq_len, n_heads_k, head_dim).transpose(1, 2)

    # Apply RoPE (before expanding K)
    position_ids = torch.arange(seq_len, device=model.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(K, position_ids)

    Q, K = apply_rotary_pos_emb(Q, K, cos, sin, unsqueeze_dim=1)

    # Expand K to match Q (GQA broadcast)
    # We repeat each of the 4 K-heads 8 times to reach 32
    K = K.repeat_interleave(n_heads_q // n_heads_k, dim=1)

    # Final Scaled Dot Product Logits
    d_k = head_dim ** 0.5
    logits = torch.matmul(Q, K.transpose(-1, -2)) / d_k

    # Apply Causal Mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(model.device)
    logits.masked_fill_(mask, float('-inf'))

    # Evidence Sum 
    # Using logsumexp for stability
    p_tot_log = torch.logsumexp(logits, dim=-1)
    p_tot = torch.exp(p_tot_log)

    return p_tot_log, p_tot

##################################################################
##################################################################

def get_head_magnitudes(model, outputs, layer_idx):
    """
    Extracts the L2 norm of each attention head output at a specific layer.
    """
    # Get the hidden states for the target layer
    # hidden_states is a tuple of (num_layers + 1)
    # Layer 0 is the embedding; Layer 1 is the first block output.
    h_state = outputs.hidden_states[layer_idx] # Shape: (batch, seq, 2048)
    
    # Focus on the final token (the one we are predicting from)
    h_T = h_state[0, -1]
    
    # Reshape into (num_heads, head_dim) -> (32, 64)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    h_heads = h_T.view(num_heads, head_dim)
    
    # Compute L2 norm for each head
    head_mags = torch.norm(h_heads, dim=1) # Shape: (32,)
    
    return head_mags

##################################################################
##################################################################

# Get top k logits
def get_logit_feats(outputs, k=64):
    # Get raw logits for the final token
    logits = outputs.logits[0, -1] # Shape: (vocab_size,)
    
    # Extract Top-K Raw Logits
    top_logits, top_idx = torch.topk(logits, k)
    
    # Calculate Stats on the Raw Logits
    probs = torch.softmax(logits, dim=-1)
#     p_entropy = -(probs * torch.log(probs + 1e-9)).sum()
    top_probs, _ = torch.topk(probs, k)
    p_entropy = -(top_probs * torch.log(top_probs + 1e-9)).sum()
    # The margin is more stable as a raw logit difference (Logit1 - Logit2)
    l_margin = top_logits[0] - top_logits[1]
    
    # Bundle: [Top-K Logits (K), Entropy (1), Margin (1)] 
    stats = torch.tensor([p_entropy.item(), l_margin.item()]).to(logits.device)
    
    return top_probs, top_logits, stats

##################################################################
##################################################################

# Extract features from the model (These will be used to train our classifier.)
def extract_features(model, outputs, saved, milestones):
    all_feats = []

    # 1. Accmulated Evidence for selected layers (per head) (32+32 = 64 dims)
    for m in milestones:
        p_log, _ = get_p_tot_log(model, saved, key_suffix=f"lyr_{m}")
        all_feats.append(p_log[0, :, -1])
    
    # 2. Magnitudes (per head) for selected layers  (32+32 = 64 dims)
    for m in milestones:  
        head_mags = get_head_magnitudes(model, outputs, m)
        all_feats.append(head_mags)

    # 3. Whole-Net Trajectory (22 Mags + 22 Innovs = 44 dims)
    for i in range(22):
        mag = saved[f"mag_{i}"].unsqueeze(0)
#         innov = torch.norm(saved[f"delta_x_{i}"][0, -1]).unsqueeze(0)
        innov = saved[f"delta_x_{i}"]
        all_feats.append(mag)
        all_feats.append(innov)

    # 4. Surface Logits (32 dims)
    _, logits, stats = get_logit_feats(outputs)
    all_feats.append(logits)
    all_feats.append(stats)

    return torch.cat([f.flatten() for f in all_feats])

##################################################################
##################################################################


