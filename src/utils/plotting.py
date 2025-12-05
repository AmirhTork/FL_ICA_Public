import matplotlib.pyplot as plt

def plot_metrics(history, title='Metrics over Rounds', show=True, save_path=None):
    rounds = range(1, len(history['accuracy']) + 1)
    plt.figure(figsize=(7,5))
    plt.plot(rounds, history['accuracy'], marker='o', label='Accuracy')
    plt.plot(rounds, history['f1_weighted'], marker='x', label='Weighted F1')
    if 'roc_auc' in history:
        plt.plot(rounds, history['roc_auc'], marker='s', label='ROC AUC')
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close() 
