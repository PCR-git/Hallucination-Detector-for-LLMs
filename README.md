# Internal Signal Probing for Hallucination Detection & Reflective RAG

This repository implements a method for estimating the correctness of language model outputs using internal transformer signals, and applies it to an adaptive retrieval framework ("Reflective RAG").

---

## Overview

We extract features from a model’s internal activations (attention, residual stream, and logits) during a forward pass, and train a small classifier to predict whether the model’s output is correct. This estimate can be interpreted as a post-hoc confidence score.

We use this signal to guide retrieval: if the model is not confident in its zero-shot output, we trigger retrieval and generate a RAG response. When both are available, we select the response with higher predicted correctness.

## Repository Structure
```
Reflective RAG
├── data/ # Dataset utilities and sample TriviaQA subset
├── evals/ # Evaluation and plotting functions
├── model/ # Classifier ("hallucination detector") and training code
├── utils/ # Feature extraction and hooks into TinyLlama
├── rag_utils/ # Retrieval and RAG pipeline
│
├── Hallucination_Detector_V2.ipynb # Feature analysis and ablations (LAMBADA)
├── Reflective_RAG_V2.ipynb # Reflective RAG experiments (TriviaQA)
│
├── snippets_map.json # RAG corpus
├── trivia_kb.index # FAISS index
│
└── Hallucination_Detection_from_Transformer_Activations.pdf # Method description and results
```

## Data

The original experiments used:
- LAMBADA (OpenAI processed version)
- TriviaQA (Wikipedia subset)

Due to environment constraints, full datasets are not included. The repository contains a sample (~5k questions) from TriviaQA’s dev set for small-scale experiments.

## Models

The following models were used (not included in the repository):
- TinyLlama-1.1B-Chat-v1.0 (LLM)
- all-MiniLM-L6-v2 (embedding model for retrieval)

Both were downloaded locally and placed in the project root during experiments.

## Notebooks

**Hallucination_Detector_V2.ipynb**
- Trains the classifier on LAMBADA
- Performs ablations on feature groups and layers
- Visualizes internal signals for correct vs incorrect predictions

**Reflective_RAG_V2.ipynb**
- Trains and evaluates on TriviaQA
- Implements and tests Reflective RAG

## Method Summary

1. Run a forward pass of the LLM  
2. Extract internal features  
3. Train a classifier to estimate the probability of a correct output
4. Use this estimate to decide whether to trigger retrieval  

## Notes

- Models and full datasets are not included  
- Paths may need adjustment for local setup  
- This code is intended for research experiments, not production use  
