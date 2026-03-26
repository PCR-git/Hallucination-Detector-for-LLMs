import torch
import torch.nn as nn

################################################################################
################################################################################

class HallucinationDetector(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

################################################################################
################################################################################

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.net(x)

################################################################################
################################################################################

class RNNHallucinationDetector(nn.Module):
    """
    A recurrent sequence classifier for detecting hallucinations based on 
    internal transformer activations collected during generation.
    """
    def __init__(self, input_dim=183, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        
        # GRU layer to process the temporal trajectory of hidden states.
        # bidirectional=True
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head for binary prediction (Hallucination vs. Correct).
        # Input dimension is hidden_dim * 2 to account for bidirectionality.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths=None):
        """
        Forward pass for a batch of sequences.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).
            lengths (torch.Tensor, optional): Actual lengths of each sequence 
                                              before padding for accurate RNN processing.
        Returns:
            torch.Tensor: Logits for the hallucination classification.
        """
        # Optional: Handle variable length sequences with packing
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        rnn_out, _ = self.rnn(x)

        # Unpack if necessary
        if lengths is not None:
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # Extract the final hidden state as the sequence representation.
        # In a bidirectional GRU, the last temporal output captures 
        # the accumulated state of the entire generation trajectory.
        final_state = rnn_out[:, -1, :] 
        
        return self.classifier(final_state)

