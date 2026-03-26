import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from model import train_hallucination_detector

################################################################################
################################################################################

def evaluate_detector(model, X_test, y_test, threshold=0.5, name="Detector"):
    """
    Evaluates the hallucination detector and prints a report.
    """
    model.eval()
    with torch.no_grad():
        # Forward pass (Logits)
        logits = model(X_test)
        
        # Convert to probabilities and binary classes
        probs = torch.sigmoid(logits).squeeze()
        predicted_classes = (probs > threshold).float()
        
        # Calculate Accuracy
        y_test_squeezed = y_test.squeeze()
        correct = (predicted_classes == y_test_squeezed).float()
        accuracy = correct.mean().item()
        
        # Prepare for Scikit-Learn metrics
        y_true = y_test_squeezed.cpu().numpy()
        y_pred = predicted_classes.cpu().numpy()
        
        print(f"\n" + "="*30)
        print(f" REPORT: {name}")
        print("="*30)
        print(f"Test Accuracy: {accuracy:.2%}")
        print(f"Decision Threshold: {threshold}")
        
        print("\n--- Classification Stats ---")
        # target_names assumes 0 is Hallucination, 1 is Correct
        print(classification_report(y_true, y_pred, target_names=['Hallucination', 'Correct']))

        # Breakdown the Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        # Handle cases where the model might only predict one class
        if cm.shape == (2, 2):
            print("--- Confusion Matrix Breakdown ---")
            print(f"Actual Hallucinations: {cm[0][0]:4d} caught | {cm[0][1]:4d} missed")
            print(f"Actual Correct:        {cm[1][0]:4d} alarms | {cm[1][1]:4d} verified")
        else:
            print("Confusion Matrix (Single-class prediction detected):")
            print(cm)
            
        return accuracy, cm

################################################################################
################################################################################

def run_kfold_evaluation(X_raw, y_raw, model, k_folds=5, threshold=0.5, seed=2026, input_dim=238, hidden_dim=32, epochs=200, batch_size=32, device="cuda", pos_weight=10):
    """
    Performs k-fold cross-validation 
    """

    # Setup reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_size = len(X_raw)
    indices = torch.randperm(dataset_size)
    fold_size = dataset_size // k_folds
    fold_results = []

    print(f" Starting {k_folds}-Fold Cross Validation...")

    for fold in range(k_folds):
        print(f"\n--- FOLD {fold + 1}/{k_folds} ---")
        
        # Define indices for this fold
        start, end = fold * fold_size, (fold + 1) * fold_size
        test_idx = indices[start:end]
        train_idx = torch.cat((indices[:start], indices[end:]))
        
        # Slice the data
        X_train_fold_raw = X_raw[train_idx]
        y_train = y_raw[train_idx].to(device)
        
        X_test_fold_raw = X_raw[test_idx]
        y_test = y_raw[test_idx].to(device)

        # Normalize based only on training fold data
        X_mean = X_train_fold_raw.mean(dim=0)
        X_std = X_train_fold_raw.std(dim=0) + 1e-4
        
        X_train = ((X_train_fold_raw - X_mean) / X_std).to(device)
        X_test = ((X_test_fold_raw - X_mean) / X_std).to(device)
        
        # Initialize fresh model and optimizer
        detector = model(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        
        # Train model
        history = train_hallucination_detector(
            detector, 
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            pos_weight=pos_weight
        )
        
        # Plot log loss
        plt.plot(np.log(history))
        plt.grid()
        plt.show()
        
        # Evaluate
        acc, cm = evaluate_detector(detector, X_test, y_test, threshold=threshold, name=f"Fold {fold+1}")
        
        # Calculate F1 for Hallucination (Class 0)
        detector.eval()
        with torch.no_grad():
            test_logits = detector(X_test)
            test_preds = (torch.sigmoid(test_logits) > threshold).float().cpu().numpy()
            test_true = y_test.cpu().numpy()
            f1_hall = f1_score(test_true, test_preds, pos_label=0)
            
        fold_results.append({
            'accuracy': acc,
            'f1_hallucination': f1_hall,
            'cm': cm
        })

    # Aggregate Final Statistics
    avg_acc = np.mean([f['accuracy'] for f in fold_results])
    avg_f1 = np.mean([f['f1_hallucination'] for f in fold_results])
    std_f1 = np.std([f['f1_hallucination'] for f in fold_results])

    print("\n" + "="*40)
    print(f" FINAL CV RESULTS ({input_dim} Features)")
    print("="*40)
    print(f"Average Accuracy: {avg_acc:.2%}")
    print(f"Average F1 (Hallucination): {avg_f1:.4f} (+/- {std_f1:.4f})")
    
    return fold_results

################################################################################
################################################################################

def run_kfold_evaluation_V2(
    X_raw, y_raw, model,
    k_folds=5, threshold=0.5, seed=2026,
    input_dim=238, hidden_dim=32,
    epochs=200, batch_size=32,
    device="cuda", pos_weight=10
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_size = len(X_raw)
    indices = torch.randperm(dataset_size)
    fold_size = dataset_size // k_folds
    fold_results = []

    print(f" Starting {k_folds}-Fold Cross Validation...")

    for fold in range(k_folds):
        print(f"\n--- FOLD {fold + 1}/{k_folds} ---")

        start, end = fold * fold_size, (fold + 1) * fold_size
        test_idx = indices[start:end]
        train_idx = torch.cat((indices[:start], indices[end:]))

        X_train_raw = X_raw[train_idx]
        y_train = y_raw[train_idx].to(device)

        X_test_raw = X_raw[test_idx]
        y_test = y_raw[test_idx].to(device)

        # -----------------------------
        # Split features + flag
        # -----------------------------
        X_train_feat = X_train_raw[:, :-1]
        X_train_flag = X_train_raw[:, -1:]

        X_test_feat = X_test_raw[:, :-1]
        X_test_flag = X_test_raw[:, -1:]

        # -----------------------------
        # Normalize ONLY features
        # -----------------------------
        X_mean = X_train_feat.mean(dim=0)
        X_std  = X_train_feat.std(dim=0) + 1e-6

        X_train_feat = (X_train_feat - X_mean) / X_std
        X_test_feat  = (X_test_feat  - X_mean) / X_std

        # -----------------------------
        # Reattach flag
        # -----------------------------
        X_train = torch.cat([X_train_feat, X_train_flag], dim=1).to(device)
        X_test  = torch.cat([X_test_feat,  X_test_flag],  dim=1).to(device)

        detector = model(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

        history = train_hallucination_detector(
            detector,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            pos_weight=pos_weight
        )

        plt.plot(np.log(history))
        plt.grid()
        plt.show()

        acc, cm = evaluate_detector(detector, X_test, y_test, threshold=threshold, name=f"Fold {fold+1}")

        detector.eval()
        with torch.no_grad():
            logits = detector(X_test)
            preds = (torch.sigmoid(logits) > threshold).float().cpu().numpy()
            f1_hall = f1_score(y_test.cpu().numpy(), preds, pos_label=0)

        fold_results.append({
            'accuracy': acc,
            'f1_hallucination': f1_hall,
            'cm': cm
        })

    return fold_results

################################################################################
################################################################################

def run_kfold_evaluation_multisplit(
    X_raw, y_raw, model,
    k_folds=5,
    threshold=0.5,
    input_dim=239,
    hidden_dim=64,
    batch_size=32,
    epochs=100,
    pos_weight=2.0,
    device="cuda"
):

    N = len(X_raw)
    fold_size = N // k_folds

    fold_results = []

    print(f"Running {k_folds}-fold CV...")

    for fold in range(k_folds):
        print(f"\n--- FOLD {fold + 1}/{k_folds} ---")

        # NO RANDOM SHUFFLE HERE
        start = fold * fold_size
        end   = (fold + 1) * fold_size

        test_idx  = torch.arange(start, end)
        train_idx = torch.cat([torch.arange(0, start), torch.arange(end, N)])

        # -----------------------------
        # Raw splits
        # -----------------------------
        X_train_raw = X_raw[train_idx]
        y_train = y_raw[train_idx].to(device)

        X_test_raw = X_raw[test_idx]
        y_test = y_raw[test_idx].to(device)

        # -----------------------------
        # Split features + flag
        # -----------------------------
        X_train_feat = X_train_raw[:, :-1]
        X_train_flag = X_train_raw[:, -1:]

        X_test_feat = X_test_raw[:, :-1]
        X_test_flag = X_test_raw[:, -1:]   # keep untouched

        # -----------------------------
        # Normalize ONLY features
        # -----------------------------
        X_mean = X_train_feat.mean(dim=0)
        X_std  = X_train_feat.std(dim=0) + 1e-6

        X_train_feat = (X_train_feat - X_mean) / X_std
        X_test_feat  = (X_test_feat  - X_mean) / X_std

        # -----------------------------
        # Reattach flag
        # -----------------------------
        X_train = torch.cat([X_train_feat, X_train_flag], dim=1).to(device)
        X_test  = torch.cat([X_test_feat,  X_test_flag],  dim=1).to(device)

        # -----------------------------
        # Dynamic input dim
        # -----------------------------
        input_dim_actual = X_train.shape[1]

        detector = model(input_dim=input_dim_actual, hidden_dim=hidden_dim).to(device)

        train_hallucination_detector(
            detector,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            pos_weight=pos_weight
        )

        detector.eval()

        with torch.no_grad():

            logits = detector(X_test)
            probs = torch.sigmoid(logits)

            preds = (probs > threshold).float()

            # -----------------------------
            # Use RAW FLAG (important)
            # -----------------------------
            flags = X_test_flag.squeeze().cpu()

            is_zs  = flags == 0
            is_rag = flags == 1

            def get_f1(mask):
                if mask.sum() == 0:
                    return 0.0
                return f1_score(
                    y_test[mask].cpu().numpy(),
                    preds[mask].cpu().numpy(),
                    pos_label=0
                )

            fold_results.append({
                'combined_f1': f1_score(
                    y_test.cpu().numpy(),
                    preds.cpu().numpy(),
                    pos_label=0
                ),
                'zero_shot_f1': get_f1(is_zs),
                'rag_f1': get_f1(is_rag)
            })

            print(
                f"ZS F1: {fold_results[-1]['zero_shot_f1']:.4f} | "
                f"RAG F1: {fold_results[-1]['rag_f1']:.4f}"
            )

    return fold_results

################################################################################
################################################################################

