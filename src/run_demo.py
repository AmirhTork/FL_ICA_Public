from utils.dataset_utils import synthetic_binary_classification
from federated.safe_federated_learning import safe_federated_learning
from federated.attacked_federated_learning import attacked_federated_learning
from utils.plotting import plot_metrics
import numpy as np
 
# ---------- config ----------
NUM_CLIENTS = 4
NUM_ROUNDS = 9
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01

ATTACKER_IDS = [0,1]
SCALING_FACTOR = 50.0
ATTACK_ROUNDS = [3, 4, 5]
# ---------------------------

def run_demo():
    X, y = synthetic_binary_classification(
        n_samples=2000, n_features=20, seed=1, imbalance=0.5
    )

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    split = int(0.8 * len(y))
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

    print("\n========== SAFE TRAINING ==========\n")
    clean_model, safe_history = safe_federated_learning(
        X_train, y_train, X_test, y_test,
        NUM_ROUNDS, NUM_CLIENTS, BATCH_SIZE, LOCAL_EPOCHS, LEARNING_RATE
    )
    plot_metrics(safe_history, title="Safe Federated Training")

    print("\n========== ATTACK TRAINING ==========\n")
    attacked_model, attack_history = attacked_federated_learning(
        clean_model,
        X_train, y_train, X_test, y_test,
        NUM_ROUNDS, NUM_CLIENTS, BATCH_SIZE, LOCAL_EPOCHS, LEARNING_RATE,
        ATTACKER_IDS, ATTACK_ROUNDS, SCALING_FACTOR
    )
    plot_metrics(attack_history, title="Attacked Federated Training")

if __name__ == "__main__":
    run_demo()

