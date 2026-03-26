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

# Extract context from TriviaQA JSON
def load_trivia_snippets(data_folder, limit=500):
    snippets = []
    # Path to the JSON metadata
    json_path = os.path.join(data_folder, 'qa', 'wikipedia-dev.json')
    # Path to the folder containing the raw text files
    evidence_folder = os.path.join(data_folder, 'evidence', 'wikipedia')

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Processing {limit} questions to find text files...")
    
    for item in data.get('Data', [])[:limit]:
        if 'EntityPages' in item:
            for page in item['EntityPages']:
                filename = page.get('Filename')
                if filename:
                    # Construct the path to the actual .txt file
                    txt_path = os.path.join(evidence_folder, filename)
                    
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as tf:
                            content = tf.read()
                            # Optional: Take only the first 2000 characters to keep vectors clean
                            snippets.append(content[:2000])
                    else:
                        # Debugging: Print if a file is missing
                        pass 

    return snippets

##################################################################
##################################################################

def get_random_trivia_entries(data_folder, num_samples=5000, seed=2026):
    """
    Modular function to grab N random samples from the TriviaQA training set.
    """
#     json_path = os.path.join(data_folder, 'qa', 'wikipedia-train.json')
    json_path = os.path.join(data_folder, 'qa', 'wikipedia-dev.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_entries = data.get('Data', [])
    
    # Ensure we don't try to sample more than exists
    num_samples = min(num_samples, len(all_entries))
    
#     random.seed(seed)
#     sampled_entries = random.sample(all_entries, num_samples)
    sampled_entries = all_entries[:num_samples]
    
    print(f"Successfully sampled {len(sampled_entries)} entries from a total of {len(all_entries)}.")
    
    return sampled_entries

##################################################################
##################################################################

def generate_trivia_features(
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

    print(f"Generating {'RAG' if use_rag else 'Zero-Shot'} features for {len(entries_to_process)} samples...")

    for i, item in enumerate(tqdm(entries_to_process)):

        question = item['Question']
        gold_aliases = item['Answer']['Aliases']

        saved.clear()
        handles = []

        # -----------------------------
        # 1. BUILD PROMPT (IDENTICAL TO EVAL)
        # -----------------------------
        if use_rag:
            current_k = k
            while current_k > 0:
                context_snippets = get_search_results(
                    embed_model, all_snippets, question, index, k=current_k
                )
                prompt_text = build_rag_prompt(question, context_snippets)

                temp_ids = tokenizer.encode(prompt_text)
                if len(temp_ids) <= max_prompt_len:
                    break

                current_k -= 1

            if current_k == 0:
                context_snippets = [
                    get_search_results(embed_model, all_snippets, question, index, k=1)[0][:500]
                ]
                prompt_text = build_rag_prompt(question, context_snippets)

        else:
            prompt_text = f"Question: {question}\nAnswer:"

        # -----------------------------
        # 2. TOKENIZE
        # -----------------------------
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # -----------------------------
        # 3. GENERATION FIRST (clean path)
        # -----------------------------
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        prompt_len = inputs.input_ids.shape[-1]

        answer_raw = tokenizer.decode(
            gen_outputs[0][prompt_len:],
            skip_special_tokens=True
        ).strip()

        answer_clean = answer_raw.split('\n\n')[0].strip()

        # -----------------------------
        # 4. LABEL
        # -----------------------------
        is_correct = any(alias.lower() in answer_clean.lower() for alias in gold_aliases)

        # -----------------------------
        # 5. REGISTER HOOKS (AFTER generation)
        # -----------------------------
        for idx, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(get_res_hook(idx, saved)))

            if idx in milestones:
                handles.append(layer.self_attn.q_proj.register_forward_hook(get_hook_q(f"lyr_{idx}", saved)))
                handles.append(layer.self_attn.k_proj.register_forward_hook(get_hook_k(f"lyr_{idx}", saved)))

        # -----------------------------
        # 6. FORWARD PASS (features only)
        # -----------------------------
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True
            )

        feat_vec = extract_features(model, outputs, saved, milestones)

        X.append(feat_vec.cpu())
        y.append(1.0 if is_correct else 0.0)

        # -----------------------------
        # 8. CLEANUP
        # -----------------------------
        for h in handles:
            h.remove()

    # -----------------------------
    # FINAL TENSORS
    # -----------------------------
    X_tensor = torch.stack(X).float()
    y_tensor = torch.tensor(y).float()

    return X_tensor, y_tensor

##################################################################
##################################################################

def generate_sequential_training_data(model, embed_model, tokenizer, sampled_entries, all_snippets, index, save_path, start_idx=0, num_to_process=100):
    """
    Processes a specific slice of the sampled_entries list.
    """
    # Slice the entries for this specific run
    end_idx = start_idx + num_to_process
    current_chunk = sampled_entries[start_idx:end_idx]
    
    if not current_chunk:
        print("No more entries to process in this range!")
        return None, None

    print(f"--- Processing Slice: {start_idx} to {end_idx} ---")

    # Run Zero-Shot on this chunk
    X_zs, y_zs = generate_trivia_features(
        model, embed_model, tokenizer, 
        current_chunk, all_snippets, index, use_rag=False
    )
    
    # Run RAG on the SAME chunk
    X_rag, y_rag = generate_trivia_features(
        model, embed_model, tokenizer, 
        current_chunk, all_snippets, index, use_rag=True
    )
    
    # Combine
    X_chunk = torch.cat([X_zs, X_rag], dim=0)
    y_chunk = torch.cat([y_zs, y_rag], dim=0)
    
    # Save this specific chunk (numbered for easy tracking)
    chunk_filename = f"{save_path}_chunk_{start_idx}_{end_idx}.pt"
    torch.save({'X': X_chunk, 'y': y_chunk}, chunk_filename)
    
    print(f"Chunk saved to {chunk_filename}")
    
    return X_chunk, y_chunk

##################################################################
##################################################################

def merge_trivia_chunks(data_dir="data", save_path="data/trivia_hallucination_final.pt"):
    """
    Merge saved files of features
    """
    # Look for files ending in _chunk_START_END
    search_pattern = os.path.join(data_dir, "*_chunk_*_*")
    chunk_files = sorted(glob.glob(search_pattern))
    
    if not chunk_files:
        print(f"No chunks found in {data_dir} matching the pattern.")
        return None, None
    
    all_X, all_y = [], []
    print(f"Found {len(chunk_files)} chunks. Merging...")

    # Load each chunk
    for f in chunk_files:
        try:
            data = torch.load(f)
            # Ensure we are grabbing the right keys
            all_X.append(data['X'])
            all_y.append(data['y'])
            print(f"Successfully loaded {os.path.basename(f)}: {data['X'].shape[0]} samples")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # Concatenate all tensors
    X_final = torch.cat(all_X, dim=0)
    y_final = torch.cat(all_y, dim=0)
    
    # Final Save
    torch.save({
        'X': X_final,
        'y': y_final,
        'metadata': {
            'total_samples': len(X_final),
            'feature_dim': X_final.shape[1],
            'chunk_count': len(chunk_files)
        }
    }, save_path)

    print(f"\n--- Merge Complete ---")
    print(f"Total Samples: {X_final.shape[0]}")
    print(f"Features per Sample: {X_final.shape[1]}")
    print(f"Final file saved at: {save_path}")
    
    return X_final, y_final

