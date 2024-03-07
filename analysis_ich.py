import pandas as pd
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample


df = pd.read_csv('../ich_data_w_scores.csv')


# Create a directory for images
os.makedirs('../images', exist_ok=True)


# Binarize MRS90 for functional outcome analysis
df['MRS90_binarized'] = df['MRS90'].apply(lambda x: 0 if x <= 3 else 1)
df['NIHSSADM_binarized'] = df['NIHSSADM'].apply(lambda x: 0 if x <= 10 else 1)

# Create a DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Score', 'Outcome', 'AUC', 'Optimal Threshold', 'Youden Index', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

def bootstrap_auc(y_true, y_scores, n_bootstraps=1000, ci=95):
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Randomly sample with replacement
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # If there isn't both classes in the resampled data, skip this iteration
            continue
        
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Compute percentiles for the CI
    lower = np.percentile(sorted_scores, (100-ci)/2)
    upper = np.percentile(sorted_scores, 100-(100-ci)/2)
    
    return lower, upper


def plot_roc_curve(y_true, y_scores, score_name, outcome_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, linewidth=2, label=f'{score_name} for {outcome_name}')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {score_name} - {outcome_name}')
    plt.legend(loc="lower right")

def calculate_performance_metrics(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    
    predictions = y_scores >= optimal_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    
    return {
        'AUC': roc_auc_score(y_true, y_scores),
        'Optimal Threshold': optimal_threshold,
        'Youden Index': J[optimal_idx],
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv
    }

scores = ['oICH_score', 'mICH_score', 'ICH_GS_score', 'LSICH_score', 'ICH_FOS_score', 'Max_ICH_score', 'GCSSCORE', 'NIHSSADM', 'NIHSSADM_binarized']
outcomes = {'MORT90': df['MORT90'], 'MRS90': df['MRS90_binarized']}

for score in scores:
    for outcome_name, y_true in outcomes.items():
        metrics = calculate_performance_metrics(y_true, df[score])

        # Calculate bootstrapped confidence intervals for AUC
        auc_lower, auc_upper = bootstrap_auc(y_true, df[score])
        
        
        new_row = pd.DataFrame({
            'Score': [score], 
            'Outcome': [outcome_name], 
            'AUC CI Lower': [auc_lower],
            'AUC CI Upper': [auc_upper],
            **metrics
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

                
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plot_roc_curve(y_true, df[score], score_name=score, outcome_name=outcome_name)
        
        # Add text annotations with metrics
        plt.text(0.6, 0.2, f'AUC: {metrics["AUC"]:.2f}\nSensitivity: {metrics["Sensitivity"]:.2f}\nSpecificity: {metrics["Specificity"]:.2f}\nPPV: {metrics["PPV"]:.2f}\nNPV: {metrics["NPV"]:.2f}', 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save the plot
        plt.savefig(f'../images/ROC_{score}_{outcome_name}.png')
        plt.close()



def calculate_ideal_cutoffs(df, metrics_df=metrics_df):
    outcomes = {
        'MORT90': 'MORT90',
        'MRS90_binarized': 'MRS90_binarized'
    }
    
    summary_table = pd.DataFrame(columns=['Outcome', 'Best Cutoff', 'AUC'])
    
    for outcome_name, outcome_column in outcomes.items():
        if outcome_column not in df.columns:
            continue  # Skip if the column is not present in the dataframe
        
        auc_scores = []
        cutoffs = range(0, max(df['NIHSSADM']) + 1)
        
        for cutoff in cutoffs:
            df['NIHSSADM_binarized'] = df['NIHSSADM'].apply(lambda x: 1 if x > cutoff else 0)
            auc_score = roc_auc_score(df[outcome_column], df['NIHSSADM_binarized'])
            auc_scores.append(auc_score)
        
        best_cutoff = cutoffs[np.argmax(auc_scores)]
        best_auc = max(auc_scores)
        
        new_row = pd.DataFrame({
            'Outcome': [outcome_name],
            'Best Cutoff': [best_cutoff],
            'AUC': [best_auc]
        })
        summary_table = pd.concat([summary_table, new_row], ignore_index=True)


        # plot roc of best cut off
        df['NIHSSADM_binarized'] = df['NIHSSADM'].apply(lambda x: 1 if x > best_cutoff else 0)
        auc_lower, auc_upper = bootstrap_auc(df[outcome_column], df['NIHSSADM_binarized'])
        metrics = calculate_performance_metrics(df[outcome_column], df['NIHSSADM_binarized'])

        new_row = pd.DataFrame({
            'Score': [f"NIHSSADM_{best_cutoff}"], 
            'Outcome': [outcome_name], 
            'AUC CI Lower': [auc_lower],
            'AUC CI Upper': [auc_upper],
            **metrics
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

                
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plot_roc_curve(df[outcome_column], df['NIHSSADM_binarized'], score_name=f"NIHSSADM_{best_cutoff}", outcome_name=outcome_name)
        
        # Add text annotations with metrics
        plt.text(0.6, 0.2, f'AUC: {metrics["AUC"]:.2f}\nSensitivity: {metrics["Sensitivity"]:.2f}\nSpecificity: {metrics["Specificity"]:.2f}\nPPV: {metrics["PPV"]:.2f}\nNPV: {metrics["NPV"]:.2f}', 
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save the plot
        plt.savefig(f'../images/BEST-NIHSS_{best_cutoff}_{outcome_name}.png')
        plt.close()


    return summary_table, metrics_df

summary_table, metrics_df = calculate_ideal_cutoffs(df)
print(summary_table)
summary_table.to_csv('summary_table_NIHSSADM.csv', index=False)
metrics_df.to_csv('prognostic_score_metrics.csv', index=False)
