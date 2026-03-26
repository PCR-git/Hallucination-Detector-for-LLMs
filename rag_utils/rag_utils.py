import os
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer

##################################################################
##################################################################

def get_search_results(embed_model, all_snippets, query, index, k=3):
    # Vectorize the new query
    query_vector = embed_model.encode([query]).astype('float32')
    
    # Search the index
    distances, indices = index.search(query_vector, k)
    
    # Pull the actual text strings
    retrieved_context = [all_snippets[i] for i in indices[0]]
    
    return retrieved_context

##################################################################
##################################################################

def build_rag_prompt(question, context_list):
    # Combine the snippets into one string
    context_str = "\n\n".join([f"Source {i+1}: {text}" for i, text in enumerate(context_list)])
    
    # Create the template
    prompt = f"""Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    CONTEXT:
    {context_str}

    QUESTION: 
    {question}

    ANSWER:"""
    
    return prompt

##################################################################
##################################################################

def run_zero_shot_baseline(data_folder, tokenizer, model, num_questions=5):
    # Load the metadata to get the questions
    json_path = os.path.join(data_folder, 'qa', 'wikipedia-dev.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions_data = data.get('Data', [])[:num_questions]
    
    print(f"--- Running {num_questions} Zero-Shot Tests ---")
    
    for i, item in enumerate(questions_data):
        question = item['Question']
        
        # We use a Q&A format to help the base model focus
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        outputs = model.generate(
            **inputs, 
            max_new_tokens=30, 
            do_sample=False, # Keep it deterministic
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Slice to get only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Clean up: stop at the first newline or period
        clean_answer = raw_answer.split('\n')[0].split('.')[0]
        
        print(f"\n[{i+1}] Q: {question}")
        print(f"    A: {clean_answer}")

##################################################################
##################################################################

def score_response(llm_output, gold_aliases):
    """
    Checks if any of the correct aliases appear in the LLM's generated text.
    """
    # Standardize to lowercase for fairness
    prediction = llm_output.lower()
    
    # Check each alias
    for alias in gold_aliases:
        if alias.lower() in prediction:
            return 1.0  # Perfect match found in the sentence
            
    return 0.0  # No correct alias found

##################################################################
##################################################################

def evaluate_zero_shot_full_text(model, embed_model, tokenizer, data_folder, all_snippets, index, num_questions=10, use_rag=False, k=3, show_text=True):
    json_path = os.path.join(data_folder, 'qa', 'wikipedia-dev.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions_data = data.get('Data', [])[:num_questions]
    scores = []
    
    if show_text:
        print(f"--- Full Text Evaluation ({num_questions} Questions) ---\n")

    for i, item in enumerate(tqdm(questions_data)):
        question = item['Question']
        gold_aliases = item['Answer']['Aliases']
        
        # Generation
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Determine the Prompt String
        if use_rag:
            current_k = k
            while current_k > 0:
                # Retrieve current_k snippets
                context_snippets = get_search_results(embed_model, all_snippets, question, index, k=current_k)
                prompt_text = build_rag_prompt(question, context_snippets)
                
                # Check token length
                temp_ids = tokenizer.encode(prompt_text)
                if len(temp_ids) <= 1800:
                    break
                
                # If too long, reduce k and try again
                current_k -= 1
                print(f"TRIMMING: Question {i+1} too long, reducing k to {current_k}")
            
            # Final fallback: if even k=1 is too long, truncate the text of the first snippet
            if current_k == 0:
                context_snippets = [get_search_results(embed_model, all_snippets, question, index, k=1)[0][:500]]
                prompt_text = build_rag_prompt(question, context_snippets)
        else:
            prompt_text = f"Question: {question}\nAnswer:"
        
        # Tokenize the chosen prompt
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, # Increased slightly to capture full sentences
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        prompt_length = inputs.input_ids.shape[-1]
        answer_raw = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # Clean to the first actual thought (stopping at double newlines)
        answer_clean = answer_raw.split('\n\n')[0].strip()
        
        # Scoring
        is_correct = any(alias.lower() in answer_clean.lower() for alias in gold_aliases)
        score = 1.0 if is_correct else 0.0
        scores.append(score)
        
        if show_text:
            # Display Results
            print(f"TEST #{i+1}")
            print(f"QUESTION: {question}")
            print(f"MODEL OUTPUT: {answer_clean}")
            print(f"GOLD ALIASES: {', '.join(gold_aliases)}")
            print(f"MATCH: {'✅ YES' if is_correct else '❌ NO'}")
            print("-" * 50)

    accuracy = (sum(scores) / len(scores)) * 100
    print(f"FINAL ACCURACY: {accuracy:.2f}%")

##################################################################
##################################################################    
        