import os
import json
import glob
import random
import torch
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer

from utils import get_hook_q, get_hook_k, get_res_hook
from utils import extract_features
from rag_utils import get_search_results, build_rag_prompt

##################################################################
##################################################################

def generate_trivia_sequence_features(
    model,
    embed_model,
    tokenizer,
    sampled_entries,
    all_snippets,
    index,
    milestones=[11, 21],
    use_rag=False,
    k=3,
    max_prompt_len=1800,
    limit=None
):
    X, y = [], []
    saved = {}

    entries_to_process = sampled_entries[:limit] if limit is not None else sampled_entries
    print(f"Generating {'RAG' if use_rag else 'Zero-Shot'} sequence features...")

    for i, item in enumerate(tqdm(entries_to_process)):
        question = item['Question']
        gold_aliases = item['Answer']['Aliases']

        # -----------------------------
        # 1. BUILD PROMPT
        # -----------------------------
        if use_rag:
            # (Standard RAG prompt building logic remains the same)
            current_k = k
            while current_k > 0:
                context = get_search_results(embed_model, all_snippets, question, index, k=current_k)
                prompt_text = build_rag_prompt(question, context)
                if len(tokenizer.encode(prompt_text)) <= max_prompt_len: break
                current_k -= 1
            if current_k == 0:
                context = [get_search_results(embed_model, all_snippets, question, index, k=1)[0][:500]]
                prompt_text = build_rag_prompt(question, context)
        else:
            prompt_text = f"Question: {question}\nAnswer:"

        # -----------------------------
        # 2. INITIAL GENERATION
        # -----------------------------
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[-1]

        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode for labeling
        answer_raw = tokenizer.decode(gen_outputs[0][prompt_len:], skip_special_tokens=True).strip()
        answer_clean = answer_raw.split('\n\n')[0].strip()
        is_correct = any(alias.lower() in answer_clean.lower() for alias in gold_aliases)

        # -----------------------------
        # 3. TEACHER FORCED HOOK PASS
        # -----------------------------
        # We re-run the model on the FULL sequence [Prompt + Generated Answer]
        # We only care about the answer tokens, but the model needs the prompt for context.
        saved.clear()
        handles = []
        
        # Register hooks for the forward pass
        for idx, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(get_res_hook(idx, saved)))
            if idx in milestones:
                handles.append(layer.self_attn.q_proj.register_forward_hook(get_hook_q(f"lyr_{idx}", saved)))
                handles.append(layer.self_attn.k_proj.register_forward_hook(get_hook_k(f"lyr_{idx}", saved)))

        with torch.no_grad():
            # gen_outputs[0] contains the combined sequence
            # We truncate to actual generated length to avoid processing unused max_new_tokens
            full_seq = gen_outputs[:, :prompt_len + len(tokenizer.encode(answer_clean))]
            outputs = model(full_seq, output_hidden_states=True)

        # -----------------------------
        # 4. EXTRACT SEQUENCE FEATURES
        # -----------------------------
        # We modify extract_features to return a sequence of vectors for the ANSWER tokens
        # Shape: (num_answer_tokens, feature_dim)
        feat_sequence = extract_sequence_features(model, outputs, saved, milestones, prompt_len)

        X.append(feat_sequence.cpu())
        y.append(1.0 if is_correct else 0.0)

        for h in handles: h.remove()

    return X, torch.tensor(y).float()

##################################################################
##################################################################

def generate_sequential_training_data_sequences(
    model, 
    embed_model, 
    tokenizer, 
    sampled_entries, 
    all_snippets, 
    index, 
    save_path, 
    start_idx=0, 
    num_to_process=100
):
    """
    Processes a specific slice of the sampled_entries list for sequence-based features.
    """
    end_idx = start_idx + num_to_process
    current_chunk = sampled_entries[start_idx:end_idx]
    
    if not current_chunk:
        print("No more entries to process in this range!")
        return None, None

    print(f"\n--- Processing Sequence Slice: {start_idx} to {end_idx} ---")

    # 1. Run Zero-Shot on this chunk
    # X_zs is a list of tensors: [ (seq_len_1, D), (seq_len_2, D), ... ]
    X_zs, y_zs = generate_trivia_sequence_features(
        model, embed_model, tokenizer, 
        current_chunk, all_snippets, index, use_rag=False
    )
    
    # 2. Run RAG on the SAME chunk
    X_rag, y_rag = generate_trivia_sequence_features(
        model, embed_model, tokenizer, 
        current_chunk, all_snippets, index, use_rag=True
    )
    
    # 3. Combine lists
    # We use list addition for X because sequence lengths vary
    X_chunk = X_zs + X_rag
    y_chunk = torch.cat([y_zs, y_rag], dim=0)
    
    # 4. Save chunk
    # We save as a dictionary containing the list of tensors
    chunk_filename = f"{save_path}_seq_chunk_{start_idx}_{end_idx}.pt"
    torch.save({
        'X': X_chunk, 
        'y': y_chunk,
        'metadata': {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'samples_per_mode': len(X_zs)
        }
    }, chunk_filename)
    
    print(f"Sequence chunk saved to {chunk_filename}")
    
    return X_chunk, y_chunk

##################################################################
##################################################################

def extract_sequence_features(model, outputs, saved, milestones, prompt_len):
    """
    Wraps the existing extract_features to process every token in the answer.
    """
    sequence_feats = []
    
    # Calculate the actual number of generated tokens
    total_seq_len = outputs.logits.shape[1]
    answer_seq_len = total_seq_len - prompt_len
    
    for t in range(answer_seq_len):
        # Determine the absolute index in the full sequence
        abs_idx = prompt_len + t
        
        # 1. Slice 'outputs' to look like a single-token pass for this step
        # We create a dummy object or a dict that mimics the structure 
        # extract_features expects.
        current_step_outputs = type('obj', (object,), {
            'logits': outputs.logits[:, abs_idx:abs_idx+1, :],
            'hidden_states': tuple(h[:, abs_idx:abs_idx+1, :] for h in outputs.hidden_states)
        })

        # 2. Slice 'saved' dictionary for this specific token
        # Your hooks currently save full sequences [Batch, Seq, Dim]
        step_saved = {}
        for key, tensor in saved.items():
            # We select only the specific token index from the sequence
            step_saved[key] = tensor[:, abs_idx:abs_idx+1, :]

        # 3. Call your original function
        # This returns the 1D feature vector for token 't'
        token_vec = extract_features(model, current_step_outputs, step_saved, milestones)
        
        sequence_feats.append(token_vec)

    # Returns (answer_seq_len, feature_dim)
    return torch.stack(sequence_feats, dim=0)

