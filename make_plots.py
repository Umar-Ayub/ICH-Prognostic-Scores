import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load your dataset
file_path = '../ich_data_w_scores.csv'  # Adjust this path to your file location
df = pd.read_csv(file_path)

# Create a directory for images
import os
os.makedirs('../images', exist_ok=True)

# Binarize MRS90 for functional outcome analysis
df['MRS90_binarized'] = df['MRS90'].apply(lambda x: 0 if x <= 3 else 1)

# Define scores
scores = ['oICH_score', 'mICH_score', 'ICH_GS_score', 'LSICH_score', 'ICH_FOS_score', 'Max_ICH_score', 'GCSSCORE', 'NIHSSADM']

# Function to plot ROC curves
def plot_roc_curves(outcome_name):
    plt.figure(figsize=(10, 8))
    y_true = df[outcome_name]
    for score in scores:
        y_scores = df[score]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{score} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {outcome_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'../images/roc_curves_{outcome_name}.png')
    plt.show()

# Plot ROC curves for MORT90
plot_roc_curves('MORT90')

# Plot ROC curves for MRS90_binarized
plot_roc_curves('MRS90_binarized')
