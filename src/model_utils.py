# In[]
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from config import (
    Basic_columns,
    CBC_WBC_columns,
    CBC_DC_columns,
    CBC_RBC_columns,
    CBC_PLT_columns,
    CPD_RBC_columns,
    CPD_PLT_columns,
    CPD_WBC_columns
)

def get_chosen_columns(var):
    """
    Given a variable group list (e.g., ['CBC', 'DC', 'basic']),
    return the combined list of column names to be selected from the dataset.
    """
    chosen_col = []

    if 'basic' in var:
        chosen_col += Basic_columns

    if 'CBC' in var and 'CBC(RBC)' not in var and 'CBC(PLT)' not in var and 'CBC(WBC)' not in var:
        chosen_col += CBC_WBC_columns + CBC_RBC_columns + CBC_PLT_columns + CBC_DC_columns

    if 'CBC(RBC)' in var:
        chosen_col += CBC_RBC_columns

    if 'CBC(PLT)' in var:
        chosen_col += CBC_PLT_columns

    if 'CBC(WBC)' in var:
        chosen_col += CBC_WBC_columns

    if 'CPD' in var and 'CPD(RBC)' not in var and 'CPD(PLT)' not in var and 'CPD(WBC)' not in var:
        chosen_col += CPD_WBC_columns + CPD_RBC_columns + CPD_PLT_columns

    if 'CPD(RBC)' in var:
        chosen_col += CPD_RBC_columns

    if 'CPD(PLT)' in var:
        chosen_col += CPD_PLT_columns

    if 'CPD(WBC)' in var:
        chosen_col += CPD_WBC_columns


    if not chosen_col:
        print(f'Warning: Variable group {var} is not defined.')

    return chosen_col

def performance (y_test, predictions, y_prob):
    """
    Compute performance metrics based on true labels and predicted probabilities.

    Parameters:
    - y_test: ground truth labels
    - predictions: predicted class labels
    - y_prob: predicted probabilities

    Returns:
    - conf_matrix: confusion matrix
    - metrics: list of performance metrics including AUROC, AUPRC, accuracy, F1 score, sensitivity (recall), 
               specificity, precision (PPV), and NPV
    """

    conf_matrix = confusion_matrix(y_test, predictions)
    true_positive= conf_matrix[1,1]
    true_negative= conf_matrix[0,0]
    false_positive= conf_matrix[0,1]
    false_negative= conf_matrix[1,0]
    total= true_positive+true_negative+false_positive+false_negative
    precision= true_positive/(true_positive+false_positive)  #positive predictive value
    recall= true_positive/(true_positive+false_negative)   # sensitivity
    specificity= true_negative/(true_negative+false_positive)
    negative_predictive_value= true_negative/(true_negative+false_negative)
    accuracy= (true_positive+true_negative)/total
    F1_score=(2*precision*recall)/(precision+recall)
    AUROC= roc_auc_score (y_test, y_prob)
    AUPRC= average_precision_score (y_test, y_prob)
    return conf_matrix, [{
        'AUROC': AUROC,
        'AUPRC': AUPRC,
        'accuracy': accuracy,
        'F1_score': F1_score,
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'NPV': negative_predictive_value
    }]

def evaluate_model(prob, y, threshold):
    """
    Evaluate a classification model using a specified threshold.

    Parameters:
    - prob: predicted probabilities from the model
    - y: ground truth labels
    - threshold: decision threshold for binary classification

    Returns:
    - con_matrix: confusion matrix
    - df_model_perf: performance metrics as a pandas DataFrame
    - predictions: binary predictions
    """
    predictions = [1 if p > threshold else 0 for p in prob]
    con_matrix, model_perf = performance(y, predictions, prob)
    df_model_perf = pd.DataFrame(model_perf)
    return con_matrix, df_model_perf, predictions


def imputer(df, numerical_impute_method, train=True, num_impute_values=None):
    """
    Perform imputation on missing values for numerical columns.

    Parameters:
    - df: input DataFrame
    - numerical_impute_method: method to impute ('mean', 'median', 'zero', 'knn', 'mice')
    - train: whether this is the training phase or validation phase
    - num_impute_values: values or imputer used during training to apply during validation

    Returns:
    - df_impute: DataFrame after imputation
    - num_impute_values: imputer object or imputation values (used in validation)
    """
    
    numerical_columns = [col for col in df.columns if col != 'Gender']
    df_impute = df.copy()

    if train:
        if numerical_impute_method == 'mean':
            num_impute_values = df[numerical_columns].mean() if numerical_columns else None
        elif numerical_impute_method == 'median':
            num_impute_values = df[numerical_columns].median() if numerical_columns else None
        elif numerical_impute_method == 'zero':
            num_impute_values = 0
        elif numerical_impute_method == 'knn':
            knn_imputer = KNNImputer()
            df_impute[numerical_columns] = knn_imputer.fit_transform(df_impute[numerical_columns])
            num_impute_values = knn_imputer
        elif numerical_impute_method == 'mice':
            mice_imputer = IterativeImputer(max_iter=30)
            df_impute[numerical_columns] = mice_imputer.fit_transform(df_impute[numerical_columns])
            num_impute_values = mice_imputer
        else:
            print('Not specified impute method for numerical variables')

        if numerical_columns and numerical_impute_method not in ['knn', 'mice']:
            df_impute.loc[:, numerical_columns] = df_impute[numerical_columns].fillna(num_impute_values)

        return df_impute, num_impute_values

    else:
        # Apply imputer during validation
        if isinstance(num_impute_values, (IterativeImputer, KNNImputer)):
            df_impute[numerical_columns] = num_impute_values.transform(df_impute[numerical_columns])
        else:
            df_impute.loc[:, numerical_columns] = df_impute[numerical_columns].fillna(num_impute_values)

        return df_impute

def scaling(x, scaler, train=True):
    """
    Scale numerical features using StandardScaler.

    Parameters:
    - x: input DataFrame
    - scaler: StandardScaler object
    - train: whether this is the training phase or not

    Returns:
    - x_scaled: DataFrame with scaled values
    - scaler: fitted scaler object (used during validation)
    """
    columns_to_scale = [col for col in x.columns if col != 'Gender']

    if train:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x[columns_to_scale])
        x_scaled = pd.DataFrame(x_scaled, columns=columns_to_scale, index=x.index)
    else:
        x_scaled = scaler.transform(x[columns_to_scale])
        x_scaled = pd.DataFrame(x_scaled, columns=columns_to_scale, index=x.index)

    if 'Gender' in x.columns:
        x_scaled['Gender'] = x['Gender']

    return x_scaled, scaler



def print_model_summary(df, model_names, title=None, metrics=None, show_std=['AUROC', 'AUPRC']):
    """
    Print model performance metrics in a horizontally aligned table using DataFrame.

    Parameters:
    - df: DataFrame of metrics (indexed by model_name)
    - model_names: list of model names to include
    - title: optional title to print
    - metrics: list of metrics to include (default = all)
    - show_std: metrics to display with ± std
    """
    default_metrics = ['AUROC', 'AUPRC', 'accuracy', 'F1_score', 'recall', 'specificity', 'precision', 'NPV']
    metrics = metrics if metrics is not None else default_metrics

    summary_rows = []

    for name in model_names:
        model_df = df.loc[df.index == name]
        mean_vals = model_df.mean()
        std_vals = model_df.std()

        row = {}
        row['comparison_label'] = name
        for metric in metrics:
            if metric in show_std:
                row[metric] = f"{mean_vals[metric]:.3f} ± {std_vals[metric]:.3f}"
            else:
                row[metric] = f"{mean_vals[metric]:.3f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if title:
        print(f"\n==== {title} ====")
    print(summary_df.to_string(index=False))


def compute_ecce_r(y_true, y_prob):
    """
    Compute the Expected Cumulative Calibration Error Range (ECCE-R) metric.

    ECCE-R is a non-parametric calibration metric that measures the range of 
    cumulative differences between the true labels and predicted probabilities. 
    A smaller ECCE-R indicates better calibration.

    Parameters:
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).
        
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns:
    -------
    ecce_r : float
        The ECCE-R score, defined as the range (max - min) of the cumulative
        differences between the sorted true labels and predicted probabilities.
    """
    # Sort by predicted probabilities
    sorted_indices = np.argsort(y_prob)
    sorted_y_true = np.array(y_true)[sorted_indices]
    sorted_y_prob = np.array(y_prob)[sorted_indices]
    
    # Compute cumulative error
    cum_diff = np.cumsum(sorted_y_true - sorted_y_prob) / len(y_true)
    
    # ECCE-R = Maximum cumulative error - Minimum cumulative error
    ecce_r = np.max(cum_diff) - np.min(cum_diff)
    return ecce_r


def print_baseline_performance(perf_dict):
    """
    Print a formatted table showing performance of the Mentzer rule across cohorts.

    Parameters:
    - perf_dict: dictionary with cohort name as key and performance list (from performance()) as value
    """
    metrics = ['AUROC', 'AUPRC', 'accuracy', 'F1_score', 'recall', 'specificity', 'precision', 'NPV']
    
    print("\n=== Baseline model Performance ===")
    print("{:<12} ".format("Cohort") + "  ".join([f"{m:<12}" for m in metrics]))
    print("-" * (12 + 15 * len(metrics)))

    for cohort, perf in perf_dict.items():
        row = f"{cohort:<12} "
        for m in metrics:
            value = perf[0][m]
            row += f"{value:<14.3f}"
        print(row)

# %%
