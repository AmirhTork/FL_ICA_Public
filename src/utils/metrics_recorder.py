import torch
import pandas as pd

class MetricsRecorder:
    def __init__(self):
        self.rows = []
        self.metrics_history = {'accuracy': [], 'f1_weighted': [], 'roc_auc': []}
        self.all_labels = []
        self.all_logits = []

    def record(self, round_num: int, metrics: dict, client_loss_mean: float = None):
        self.metrics_history['accuracy'].append(metrics['accuracy'])
        self.metrics_history['f1_weighted'].append(metrics['f1_weighted'])


        if 'logits' in metrics and 'preds' in metrics:
            try:
                from sklearn.metrics import roc_auc_score
                # Convert logits to probabilities
                probs = torch.softmax(metrics['logits'], dim=1).cpu().numpy()
                auc = roc_auc_score(metrics['labels'], probs[:,1])
            except Exception:
                auc = None
            self.metrics_history['roc_auc'].append(auc)
        else:
            auc = None

        row = {
            'round': round_num,
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'roc_auc': auc
        }
        if client_loss_mean is not None:
            row['client_loss'] = client_loss_mean
        self.rows.append(row)


        if 'labels' in metrics and 'logits' in metrics:
            self.all_labels.append(metrics['labels'])
            self.all_logits.append(metrics['logits'].detach().cpu())

    def get_results_df(self):
        return pd.DataFrame(self.rows)

    def get_history(self):
        return self.metrics_history
 
