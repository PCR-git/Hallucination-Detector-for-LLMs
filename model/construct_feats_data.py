import torch
import pandas as pd
from tqdm import tqdm

from utils import get_res_hook, get_hook_q, get_hook_k, extract_features

################################################################################
################################################################################

def stream_openai_lambada(file_path):
    # Load the entire parquet file into memory
    df = pd.read_parquet(file_path)
    
    # Iterate through the 'text' column
    for full_text in df['text']:
        # Find the last space to split context from target
        last_space_idx = full_text.rfind(" ")
        context = full_text[:last_space_idx]
        target_word = full_text[last_space_idx:]
        
        yield context, target_word

################################################################################
################################################################################
        
def generate_hallucination_dataset(model, tokenizer, generator, saved, limit=4000, milestones=[11, 21], save_path="hallucination_data.pt"):
    """
    Runs forward passes to extract internal features and labels for hallucination detection.
    """
    X = []
    y = []
    
    for i, (context, target_word) in enumerate(tqdm(generator, total=limit)):
        if i >= limit:
            break
            
        saved.clear()
        handles = []

        # Register hooks for residual streams and milestone Q/K projections
        for idx, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(get_res_hook(idx, saved)))
            
            if idx in milestones:
                handles.append(layer.self_attn.q_proj.register_forward_hook(get_hook_q(f"lyr_{idx}", saved)))
                handles.append(layer.self_attn.k_proj.register_forward_hook(get_hook_k(f"lyr_{idx}", saved)))

        # Tokenization and alignment
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        full_enc = tokenizer.encode(context + target_word, add_special_tokens=False)
        context_enc = tokenizer.encode(context, add_special_tokens=False)
        
        if len(full_enc) <= len(context_enc):
            for h in handles: h.remove()
            continue
            
        target_token_id = full_enc[len(context_enc)]

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Feature vector construction
        feat_vec = extract_features(model, outputs, saved, milestones)

        # Labeling based on Top-5 accuracy
        logits = outputs.logits[0, -1]
        top_idx = torch.topk(logits, 5).indices
        label = 1.0 if target_token_id in top_idx else 0.0

        X.append(feat_vec.cpu())
        y.append(label)

        # Remove hooks to prevent memory leaks and interference
        for h in handles:
            h.remove()

    # Tensor conversion and persistence
    X_tensor = torch.stack(X).float()
    y_tensor = torch.tensor(y).float()

    torch.save({
        'X': X_tensor,
        'y': y_tensor,
        'metadata': {
            'samples': len(X),
            'feature_dim': X_tensor.shape[1],
            'milestones': milestones
        }
    }, save_path)
    
    print(f"Dataset saved to {save_path} with {len(X)} samples.")
    
    return X_tensor, y_tensor

################################################################################
################################################################################

