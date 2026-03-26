import torch
import numpy as np
import matplotlib.pyplot as plt

################################################################################
################################################################################

def plot_trajectory_comparison(X, y):
    # Indices 128-171: [Mag_0, Innov_0, Mag_1, Innov_1, ..., Mag_21, Innov_21]
    X_traj = X[:, 128:172].cpu().numpy()
    mags = X_traj[:, 0::2]
    innovs = X_traj[:, 1::2]
    
    # Masks for grouping
    h_mask = (y == 0.0).cpu().numpy()
    c_mask = (y == 1.0).cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    layers = np.arange(22)

    for ax, data, title, label_y in zip(
        [ax1, ax2], 
        [mags, innovs], 
        ['Structural Magnitude', 'Internal Innovation (Δx)'],
        ['Standardized Mag', 'Standardized Δx']
    ):
        # Calculate stats
        h_mean, c_mean = data[h_mask].mean(0), data[c_mask].mean(0)
        h_sem, c_sem = data[h_mask].std(0)/np.sqrt(sum(h_mask)), data[c_mask].std(0)/np.sqrt(sum(c_mask))
        
        print(c_mean)
        
        # Plot Hallucinations
        ax.plot(layers, h_mean, label='Hallucination', color='#e74c3c', marker='o')
        ax.fill_between(layers, h_mean - h_sem, h_mean + h_sem, color='#e74c3c', alpha=0.2)
        
        # Plot Correct
        ax.plot(layers, c_mean, label='Correct', color='#2ecc71', marker='s')
        ax.fill_between(layers, c_mean - c_sem, c_mean + c_sem, color='#2ecc71', alpha=0.2)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel(label_y)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

################################################################################
################################################################################

def plot_trajectory_grid_scaled(X, y):
    plt.rcParams.update({'font.size': 16})
    
    # Indices 128-171: [Mag_0, Innov_0, Mag_1, Innov_1, ..., Mag_21, Innov_21]
    X_traj = X[:, 128:172].cpu().numpy()
    mags = X_traj[:, 0::2]
    innovs = X_traj[:, 1::2]
    
    h_mask = (y == 0.0).cpu().numpy()
    c_mask = (y == 1.0).cpu().numpy()
    layers = np.arange(22)

    # sharey='row' ensures the Left and Right plots always have the same vertical scale
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey='row')
    
    titles = [
        ["Correct: Magnitudes", "Hallucination: Magnitudes"],
        ["Correct: Innovations (CosSim)", "Hallucination: Innovations (CosSim)"]
    ]
    
    data_map = [mags, innovs]
    
    for row in range(2):
        if row == 0:
            axes[row, 0].set_ylim(-4, 6)  # Example limits for Magnitudes
        else:
            axes[row, 0].set_ylim(-4, 6)    # Example limits for Innovations (CosSim)
        
        data = data_map[row]
        for col in range(2):
            ax = axes[row, col]
            mask = c_mask if col == 0 else h_mask
            color = '#2ecc71' if col == 0 else '#e74c3c'
            # Lower alpha for 'Correct' because there are more samples
            alpha = 0.02 if col == 0 else 0.1
            
            # Plot every single example path
            ax.plot(layers, data[mask].T, color=color, alpha=alpha, linewidth=0.5)
            
            # Overlay the Mean
            mean_color = '#1b5e20' if col == 0 else '#7b1fa2'
            ax.plot(layers, data[mask].mean(0), color=mean_color, linewidth=4, label='Class Mean')
            
            ax.set_title(titles[row][col], fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.2)
            
            if col == 0:
                ax.set_ylabel("Standardized Mag" if row == 0 else "Cosine Similarity")
#             ax.legend(loc='upper left')

    plt.xlabel("Layer Index")
    plt.tight_layout()
    plt.savefig('trajectory_comparison_scaled.png')
    plt.show()

################################################################################
################################################################################

def plot_evidence_distribution(X, y, idx1, idx2, label):
    plt.rcParams.update({'font.size': 16})
    
    # Features 32:64 are the Evidence for the Last Layer (L21)
    # Based on your map: 0-32 (L11), 32-64 (L21)
    evidence_l21 = X[:, idx1:idx2].cpu().numpy()
    
    h_mask = (y == 0.0).cpu().numpy()
    c_mask = (y == 1.0).cpu().numpy()
    heads = np.arange(32)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    titles = ["Correct: Head " + label, "Hallucination: Head " + label]
    masks = [c_mask, h_mask]
    colors = ['#2ecc71', '#e74c3c']
    alphas = [0.02, 0.1]
    
    for i in range(2):
        
        ax = axes[i]
        data = evidence_l21[masks[i]]

        # Plot individual lines (Each line is one example's 32 heads)
        ax.plot(heads, data.T, color=colors[i], alpha=alphas[i], linewidth=0.5)
        
        # Overlay the Mean
        mean_val = data.mean(0)
        ax.plot(heads, mean_val, color='black', linewidth=3, label=label)
        
        ax.set_title(titles[i], fontsize=15, fontweight='bold')
        ax.set_xlabel("Head Index")
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.set_ylabel("Standardized " + label)
#         ax.legend()

    # Sync the Y-limits based on the global range
    global_min = evidence_l21.min()
    global_max = evidence_l21.max()
    axes[0].set_ylim(global_min - 0.5, global_max + 0.5)

    plt.tight_layout()
    plt.show()




