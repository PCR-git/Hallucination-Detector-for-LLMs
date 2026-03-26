import torch
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from utils import extract_features, get_hook_q, get_hook_k, get_res_hook
from rag_utils import get_search_results, build_rag_prompt, score_response

##################################################################
##################################################################

def evaluate_detector_rrag(detector, X_test, y_test, X_test_raw, threshold=0.7):
    """
    Evaluates the trained detector on the held-out test set.
    Separates results by Zero-Shot and RAG modes.
    """
    detector.eval()
    results = {}
    
    with torch.no_grad():
        # Forward pass
        logits = detector(X_test)
        probs = torch.sigmoid(logits)
        
        # Thresholding
        preds = (probs >= threshold).float().cpu().numpy()
        y_true = y_test.cpu().numpy()
        
        # Global Metrics
        results['global_f1_hallucination'] = f1_score(y_true, preds, pos_label=0)
        results['global_accuracy'] = (preds == y_true).mean()

        # Slice by Mode (Final feature is the RAG flag)
        is_rag = (X_test_raw[:, -1] == 1.0)
        is_zs = (X_test_raw[:, -1] == 0.0)

        results['zs_f1_hallucination'] = f1_score(y_true[is_zs], preds[is_zs], pos_label=0)
        results['rag_f1_hallucination'] = f1_score(y_true[is_rag], preds[is_rag], pos_label=0)
        
        # Print the Report
        print(f"\n" + "="*30)
        print(f" DETECTOR EVALUATION (Threshold: {threshold})")
        print(f"="*30)
        print(classification_report(y_true, preds, target_names=['Hallucination (0)', 'Correct (1)']))
        
        print(f"--- Mode Breakdown ---")
        print(f"Zero-Shot F1: {results['zs_f1_hallucination']:.4f}")
        print(f"RAG-Only F1:  {results['rag_f1_hallucination']:.4f}")
        print(f"F1 Delta:     {results['rag_f1_hallucination'] - results['zs_f1_hallucination']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, preds)
        print(f"\n--- Confusion Matrix ---")
        print(f"Caught Hallucinations (TN): {cm[0,0]}")
        print(f"Missed Hallucinations (FN): {cm[0,1]}")
        print(f"False Alarms on Truth (FP): {cm[1,0]}")
        print(f"Verified Truths       (TP): {cm[1,1]}")

    return results

##################################################################
##################################################################    
        
def simulate_system_performance(
    X_test_pair, y_test_pair, detector, X_mean, X_std,
    threshold=0.8, COST_ZS=1.0, COST_RAG=10.0, device="cuda"
):

    detector.eval()
    with torch.no_grad():

        # -----------------------------
        # Split pairs
        # -----------------------------
        X_zs = X_test_pair[:, 0]
        X_rag = X_test_pair[:, 1]

        y_zs = y_test_pair[:, 0]
        y_rag = y_test_pair[:, 1]

        num_questions = X_zs.shape[0]

        # -----------------------------
        # Normalize ONLY features
        # -----------------------------
        X_zs_feat = X_zs[:, :-1]
        X_zs_flag = X_zs[:, -1:]  # keep untouched

        X_zs_feat = (X_zs_feat - X_mean) / X_std

        X_zs_scaled = torch.cat([X_zs_feat, X_zs_flag], dim=1)

        # -----------------------------
        # Detector predictions
        # -----------------------------
        probs = torch.sigmoid(detector(X_zs_scaled.to(device))).cpu().squeeze()

        # -----------------------------
        # Decision
        # -----------------------------
        use_rag = probs < threshold

        # -----------------------------
        # Accuracy
        # -----------------------------
        correct_adaptive = torch.where(use_rag, y_rag, y_zs).sum().item()

        correct_always_zs = y_zs.sum().item()
        correct_always_rag = y_rag.sum().item()

        acc_adaptive = correct_adaptive / num_questions * 100
        acc_zs = correct_always_zs / num_questions * 100
        acc_rag = correct_always_rag / num_questions * 100

        # -----------------------------
        # Cost
        # -----------------------------
        total_cost_adaptive = (
            (~use_rag).sum() * COST_ZS +
            use_rag.sum() * (COST_ZS + COST_RAG)
        ).item()

        cost_always_rag = num_questions * COST_RAG
        savings = (1 - total_cost_adaptive / cost_always_rag) * 100

        # -----------------------------
        # Diagnostics
        # -----------------------------
        rag_helps = ((y_rag == 1) & (y_zs == 0)).float().mean().item()
        rag_hurts = ((y_rag == 0) & (y_zs == 1)).float().mean().item()

        # -----------------------------
        # Report
        # -----------------------------
        print(f"--- Report (Threshold: {threshold}) ---")
        print(f"RAG Trigger Rate:    {use_rag.float().mean():.2%}")
        print(f"Accuracy (Adaptive): {acc_adaptive:.2f}%")
        print(f"Accuracy (Always ZS): {acc_zs:.2f}%")
        print(f"Accuracy (Always RAG): {acc_rag:.2f}%")
        print(f"Total Savings:       {savings:.2f}% vs. Always-RAG")
        print()
        print(f"RAG helps: {rag_helps:.2%}")
        print(f"RAG hurts: {rag_hurts:.2%}")

    return {
        "acc_adaptive": acc_adaptive,
        "acc_zs": acc_zs,
        "acc_rag": acc_rag,
        "savings": savings
    }

##################################################################
##################################################################    

def evaluate_adaptive_rag(
    model,
    detector,
    embed_model,
    tokenizer,
    data_folder,
    all_snippets,
    index,
    X_mean,
    X_std,
    milestones,
    threshold=0.5,
    k=3,
    num_questions=10,
    show_text=True,
    start_idx=5000
):

    # -----------------------------
    # SETUP & PREPROCESSING
    # -----------------------------
    X_mean = X_mean.to(model.device)
    X_std  = X_std.to(model.device)

    # Prune features if they contain the full logit distribution (238 -> 183 dims)
    if X_mean.shape[0] > 200:
        X_mean = torch.cat([X_mean[:182], X_mean[237:]])
        X_std  = torch.cat([X_std[:182],  X_std[237:]])

    json_path = os.path.join(data_folder, 'qa', 'wikipedia-dev.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    end_idx = start_idx + num_questions
    questions_data = data.get('Data', [])[start_idx:end_idx]

    detector.eval()

    # Metrics Tracking
    zs_scores, rag_scores, adaptive_scores = [], [], []
    rag_triggered = 0
    rag_help = 0
    rag_hurt = 0
    rag_rejected = 0

    detector_preds, detector_targets = [], []

    # Time Tracking
    total_zs_time = 0.0
    total_rag_time = 0.0
    adaptive_total_time = 0.0

    # -----------------------------
    # HELPER FUNCTIONS
    # -----------------------------
    def forward_with_feats(prompt):
        saved = {}
        handles = []
        # Register Hooks
        for i in milestones:
            layer = model.model.layers[i]
            name = f"lyr_{i}"
            handles.append(layer.self_attn.q_proj.register_forward_hook(get_hook_q(name, saved)))
            handles.append(layer.self_attn.k_proj.register_forward_hook(get_hook_k(name, saved)))
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(get_res_hook(i, saved)))

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        t_delta = time.time() - t0

        for h in handles:
            h.remove()

        feats = extract_features(model, outputs, saved, milestones)
        return feats, t_delta

    def generate_answer(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        t_delta = time.time() - t0
        prompt_len = inputs.input_ids.shape[-1]
        answer = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip().split("\n\n")[0]
        return answer, t_delta

    def build_safe_rag_prompt(question):
        current_k = k
        while current_k > 0:
            context = get_search_results(embed_model, all_snippets, question, index, k=current_k)
            prompt = build_rag_prompt(question, context)
            if len(tokenizer.encode(prompt)) <= 1800:
                return prompt
            current_k -= 1
        snippet = get_search_results(embed_model, all_snippets, question, index, k=1)[0][:500]
        return build_rag_prompt(question, [snippet])

    # -----------------------------
    # MAIN EVALUATION LOOP
    # -----------------------------
    for i, item in enumerate(tqdm(questions_data)):
        question = item['Question']
        gold_aliases = item['Answer']['Aliases']

        # --- STEP A: ZERO-SHOT PATH ---
        prompt_zs = f"Question: {question}\nAnswer:"
        feats_zs, t_zs_fwd = forward_with_feats(prompt_zs)
        
        # Preprocess features for detector
        feats_zs = feats_zs.to(model.device)
        feats_zs = torch.cat([feats_zs[:182], feats_zs[237:]])
        
        # Normalize only features (all but last)
        feats_feat = (feats_zs[:-1] - X_mean) / X_std
        # Force the RAG flag to 0.0 for Zero-Shot
        feats_input_zs = torch.cat([feats_feat, torch.tensor([0.0]).to(model.device)]).unsqueeze(0)

        with torch.no_grad():
            prob_zs = torch.sigmoid(detector(feats_input_zs)).item()

        answer_zs, t_zs_gen = generate_answer(prompt_zs)
        zs_correct = score_response(answer_zs, gold_aliases)

        total_zs_time += (t_zs_fwd + t_zs_gen)
        zs_scores.append(zs_correct)

        # --- STEP B: RAG PATH (Run for stats) ---
        rag_prompt = build_safe_rag_prompt(question)
        feats_rag, t_rag_fwd = forward_with_feats(rag_prompt)
        
        feats_rag = feats_rag.to(model.device)
        feats_rag = torch.cat([feats_rag[:182], feats_rag[237:]])
        
        # Normalize only features (all but last)
        feats_feat_rag = (feats_rag[:-1] - X_mean) / X_std
        # Force the RAG flag to 1.0 for RAG
        feats_input_rag = torch.cat([feats_feat_rag, torch.tensor([1.0]).to(model.device)]).unsqueeze(0)

        with torch.no_grad():
            prob_rag = torch.sigmoid(detector(feats_input_rag)).item()

        answer_rag, t_rag_gen = generate_answer(rag_prompt)
        rag_correct = score_response(answer_rag, gold_aliases)

        # total_rag_time calculation includes forward + generation to compare total system cost
        total_rag_time += (t_rag_fwd + t_rag_gen)
        rag_scores.append(rag_correct)

        # --- STEP C: ADAPTIVE DECISION ---
        use_rag = False
        if prob_zs < threshold:
            rag_triggered += 1
            if prob_rag > prob_zs:
                use_rag = True
            else:
                rag_rejected += 1

        final_answer = answer_rag if use_rag else answer_zs
        adaptive_scores.append(rag_correct if use_rag else zs_correct)

        # Effect Tracking
        if rag_correct and not zs_correct: rag_help += 1
        elif zs_correct and not rag_correct: rag_hurt += 1

        detector_preds.append(int(prob_zs < threshold)) # predicted hallucination (trigger)
        detector_targets.append(int(not zs_correct))    # actual hallucination

        if show_text:
            print(f"\nQ{i+start_idx+1}: {question}")
            print(f"GOLD: {', '.join(gold_aliases[:5])}")
            print(f"ZS Conf: {prob_zs:.3f} | RAG Conf: {prob_rag:.3f}")
            print(f"Decision: {'RAG' if use_rag else 'Zero-Shot'}")
            print(f"ZS Output:  {answer_zs} | {'✅' if zs_correct else '❌'}")
            print(f"RAG Output: {answer_rag} | {'✅' if rag_correct else '❌'}")

        torch.cuda.empty_cache()

    # -----------------------------
    # FINAL REPORTING (V2 Time Methodology)
    # -----------------------------
    num = len(questions_data)
    zs_acc = np.mean(zs_scores) * 100
    rag_acc = np.mean(rag_scores) * 100
    adaptive_acc = np.mean(adaptive_scores) * 100

    # V2 Style Time Accounting
    avg_zs = total_zs_time / num
    avg_rag = total_rag_time / num
    
    adaptive_total_time = sum([
        avg_zs + avg_rag if p == 1 else avg_zs
        for p in detector_preds
    ])

    efficiency_gain = (1 - (adaptive_total_time / (total_zs_time + total_rag_time))) * 100

    # Modular output dictionaries
    accuracy_results = {
        'Zero-Shot': zs_acc,
        'Always RAG': rag_acc,
        'Reflective RAG': adaptive_acc
    }

    time_results = {
        'ZS Baseline': total_zs_time,
        'Always RAG': (total_zs_time + total_rag_time),
        'Reflective': adaptive_total_time
    }

    print("\n" + "="*60)
    print("ACCURACY REPORT")
    print(f"Zero-Shot      : {zs_acc:.2f}%")
    print(f"Always RAG     : {rag_acc:.2f}%")
    print(f"Reflective RAG : {adaptive_acc:.2f}%")

    print("\nCOMPUTE & TIME COST")
    print(f"Total Time (Always Zero-Shot) : {total_zs_time:.2f}s")
    print(f"Total Time (Always RAG)       : {(total_zs_time + total_rag_time):.2f}s")
    print(f"Total Time (Reflective RAG)   : {adaptive_total_time:.2f}s")
    print(f"Efficiency Gain vs Always RAG: {efficiency_gain:.2f}%")

    return accuracy_results, time_results, efficiency_gain

##################################################################
##################################################################   

def plot_reflective_rag_report(accuracy_data, time_data, efficiency_gain=None):
    """
    Plots a dual-axis report for accuracy and computational cost.
    """
    # Set style and figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Accuracy Plot ---
    labels_acc = list(accuracy_data.keys())
    values_acc = list(accuracy_data.values())
    # Colors: Blue (ZS), Red (Always RAG), Green (Reflective)
    colors_acc = ['#3498db', '#e74c3c', '#2ecc71'] 
    
    bars1 = ax1.bar(labels_acc, values_acc, color=colors_acc, alpha=0.85, edgecolor='black')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, max(values_acc) + 10)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    # --- Compute & Time Cost Plot ---
    labels_time = list(time_data.keys())
    values_time = list(time_data.values())
    colors_time = ['#7f8c8d', '#c0392b', '#27ae60'] 
    
    bars2 = ax2.bar(labels_time, values_time, color=colors_acc, alpha=0.85, edgecolor='black')
    ax2.set_title('Compute & Time Cost', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Total Time (seconds)', fontsize=12)
    ax2.set_ylim(0, max(values_time) + 50)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add efficiency gain as a text box if provided
    if efficiency_gain:
        ax2.text(0.95, 0.95, f'Efficiency Gain:\n{efficiency_gain:.1f}%', 
                 transform=ax2.transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                 fontweight='bold', color='#27ae60')

    # Add labels on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.2f}s', 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('reflective_rag_results.png')
    plt.show()
    