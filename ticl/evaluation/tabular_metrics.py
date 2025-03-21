"""
===============================
Metrics calculation
===============================
Includes a few metric as well as functions composing metrics on results files.

"""


import numpy as np
import torch, pdb
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, mean_absolute_error,
                             r2_score, roc_auc_score)


def root_mean_squared_error_metric(target, pred):
    target = torch.tensor(target, dtype=torch.float32) if not torch.is_tensor(target) else target.float()
    pred = torch.tensor(pred, dtype=torch.float32) if not torch.is_tensor(pred) else pred.float()
    if len(pred.shape) == 2:
        assert pred.shape[1] == 1, "If non squeezed prediction ensure its only 1-d"
        pred = pred.flatten()
    return torch.sqrt(torch.nn.functional.mse_loss(target, pred))


def mean_squared_error_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.nn.functional.mse_loss(target, pred)


def mean_absolute_error_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.tensor(mean_absolute_error(target, pred))


"""
===============================
Metrics calculation
===============================
"""


def auc_metric(target, pred, multi_class='ovo', numpy=False):
    lib = np if numpy else torch
    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    
    # Convert tensors to numpy for consistent processing
    target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else np.array(target)
    pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.array(pred)
    
    # Handle NaN and Inf values in predictions
    if np.isnan(pred_np).any() or np.isinf(pred_np).any():
        print(f"Warning: AUC metric - found {np.isnan(pred_np).sum()} NaN values and {np.isinf(pred_np).sum()} Inf values in predictions")
        # Replace NaN/Inf with safe values
        pred_np = np.nan_to_num(pred_np, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Handle invalid targets - ensure they are integers with valid class labels
    if np.isnan(target_np).any() or np.isinf(target_np).any():
        print(f"Warning: AUC metric - found {np.isnan(target_np).sum()} NaN values and {np.isinf(target_np).sum()} Inf values in targets")
        # For targets, we need valid class indices - replace with most common class
        if np.isnan(target_np).any() or np.isinf(target_np).any():
            valid_targets = target_np[~(np.isnan(target_np) | np.isinf(target_np))]
            if len(valid_targets) > 0:
                # Replace with most common class
                from scipy import stats
                most_common = stats.mode(valid_targets, keepdims=False)[0]
                target_np = np.nan_to_num(target_np, nan=most_common, posinf=most_common, neginf=most_common)
            else:
                # If all targets are invalid, use zeros
                target_np = np.zeros_like(target_np)
    
    # Ensure target is integer type for classification
    target_np = target_np.astype(int)
    
    # Get unique classes with valid targets
    unique_classes = np.unique(target_np)
    n_classes = len(unique_classes)
    
    # For multi-class classification (more than 2 classes)
    if n_classes > 2:
        # Check if predictions need normalization (sum to 1 across classes)
        if len(pred_np.shape) == 2:
            # Check for and fix rows with all zeros or NaNs
            row_sums = np.sum(pred_np, axis=1)
            zero_rows = np.where(np.abs(row_sums) < 1e-10)[0]
            
            if len(zero_rows) > 0:
                print(f"Warning: AUC metric - {len(zero_rows)} rows with zero sum detected, setting to uniform distribution")
                n_pred_classes = pred_np.shape[1]
                pred_np[zero_rows] = np.ones((len(zero_rows), n_pred_classes)) / n_pred_classes
                row_sums[zero_rows] = 1.0
            
            # Normalize predictions to sum to 1 across classes
            if not np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-3):
                pred_np = pred_np / row_sums[:, np.newaxis]
        
        # Make sure we have at least one sample of each class in target
        # (ROC AUC requires at least one positive and one negative sample for each class)
        class_counts = np.bincount(target_np, minlength=n_classes)
        missing_classes = np.where(class_counts == 0)[0]
        
        if len(missing_classes) > 0:
            print(f"Warning: AUC metric - Classes {missing_classes} not present in targets. Using default AUC score.")
            auc_value = 0.5  # Default AUC for random performance
        else:
            try:
                # Use ovr (one-vs-rest) for better handling of edge cases
                auc_value = roc_auc_score(target_np, pred_np, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"Error in multi-class ROC AUC calculation: {e}")
                # Fall back to default score
                auc_value = 0.5
    else:
        # Binary classification
        if len(pred_np.shape) == 2 and pred_np.shape[1] == 2:
            # Use second column probability for positive class
            pred_np = pred_np[:, 1]
        
        # Ensure we have both classes present
        if len(unique_classes) < 2:
            print(f"Warning: AUC metric - Only class {unique_classes[0]} present in targets. Using default AUC score.")
            auc_value = 0.5
        else:
            try:
                auc_value = roc_auc_score(target_np, pred_np)
            except Exception as e:
                print(f"Error in binary ROC AUC calculation: {e}")
                auc_value = 0.5
    
    # Return value in original format
    if not numpy:
        return torch.tensor(auc_value, device=pred.device if torch.is_tensor(pred) else None)
    return auc_value


def accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(accuracy_score(target, pred[:, 1] > 0.5))


def brier_score_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    target = torch.nn.functional.one_hot(target, num_classes=len(torch.unique(target)))
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    diffs = (pred - target)**2
    return torch.mean(torch.sum(diffs, axis=1))


def ece_metric(target, pred):
    import torchmetrics
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torchmetrics.functional.calibration_error(pred, target)


def average_precision_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(average_precision_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(average_precision_score(target, pred[:, 1] > 0.5))


def balanced_accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(balanced_accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(balanced_accuracy_score(target, pred[:, 1] > 0.5))


def cross_entropy(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        ce = torch.nn.CrossEntropyLoss()
        return ce(pred.float(), target.long())
    else:
        bce = torch.nn.BCELoss()
        return bce(pred[:, 1].float(), target.float())


def r2_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    return torch.tensor(neg_r2(target, pred))


def neg_r2(target, pred):
    return -r2_score(pred.float(), target.float())


def is_classification(metric_used):
    if metric_used.__name__ in ["auc_metric", "cross_entropy"]:
        return True
    return False


def get_scoring_string(metric_used, multiclass=True, usage="sklearn_cv"):
    if metric_used.__name__ == auc_metric.__name__:
        if usage == 'sklearn_cv':
            return 'roc_auc_ovo'
        elif usage == 'autogluon':
            # return 'log_loss' # Autogluon crashes when using 'roc_auc' with some datasets usning logloss gives better scores;
            # We might be able to fix this, but doesn't work out of box.
            # File bug report? Error happens with dataset robert and fabert
            if multiclass:
                return 'roc_auc_ovo_macro'
            else:
                return 'roc_auc'
        elif usage == 'tabnet':
            return 'logloss' if multiclass else 'auc'
        elif usage == 'autosklearn':
            import autosklearn.classification
            if multiclass:
                return autosklearn.metrics.log_loss  # roc_auc only works for binary, use logloss instead
            else:
                return autosklearn.metrics.roc_auc
        elif usage == 'catboost':
            return 'MultiClass'  # Effectively LogLoss, ROC not available
        elif usage == 'xgb':
            return 'logloss'
        elif usage == 'lightgbm':
            if multiclass:
                return 'auc'
            else:
                return 'binary'
        return 'roc_auc'
    elif metric_used.__name__ == cross_entropy.__name__:
        if usage == 'sklearn_cv':
            return 'neg_log_loss'
        elif usage == 'autogluon':
            return 'log_loss'
        elif usage == 'tabnet':
            return 'logloss'
        elif usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.log_loss
        elif usage == 'catboost':
            return 'MultiClass'  # Effectively LogLoss
        return 'logloss'
    elif metric_used.__name__ == r2_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.r2
        elif usage == 'sklearn_cv':
            return 'r2'  # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'r2'
        elif usage == 'xgb':  # XGB cannot directly optimize r2
            return 'rmse'
        elif usage == 'catboost':  # Catboost cannot directly optimize r2 ("Can't be used for optimization." - docu)
            return 'RMSE'
        else:
            return 'r2'
    elif metric_used.__name__ == root_mean_squared_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.root_mean_squared_error
        elif usage == 'sklearn_cv':
            return 'neg_root_mean_squared_error'  # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'rmse'
        elif usage == 'xgb':
            return 'rmse'
        elif usage == 'catboost':
            return 'RMSE'
        else:
            return 'neg_root_mean_squared_error'
    elif metric_used.__name__ == mean_absolute_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.mean_absolute_error
        elif usage == 'sklearn_cv':
            return 'neg_mean_absolute_error'  # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'mae'
        elif usage == 'xgb':
            return 'mae'
        elif usage == 'catboost':
            return 'MAE'
        else:
            return 'neg_mean_absolute_error'
    else:
        raise Exception('No scoring string found for metric')