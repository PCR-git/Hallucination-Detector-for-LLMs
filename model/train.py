import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

################################################################################
################################################################################

def train_hallucination_detector(detector, X_train, y_train, epochs=200, batch_size=32, lr=1e-3, weight_decay=0, pos_weight=None):
    """
    Trains the MLP detector and returns the loss history.
    """
    detector.train()
    optimizer = optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup Criterion with optional weighting for class imbalance
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(X_train.device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    losses = []
    pbar = tqdm(range(epochs), desc="Training Progress")
    
    for epoch in pbar:
        epoch_loss = 0.0

        # Shuffle indices for this epoch
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for i in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_x = X_shuffled[i : i + batch_size]
            batch_y = y_shuffled[i : i + batch_size].unsqueeze(1)
            
            batch_x = torch.clamp(batch_x, min=-5.0, max=5.0)

            # Forward pass
            outputs = detector(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=0.1)
            optimizer.step()

            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / (len(X_train) / batch_size)
        losses.append(avg_loss)
        
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
    return losses

