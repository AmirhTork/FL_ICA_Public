import copy
from torch.utils.data import DataLoader
 
from aggregator.ica import ICA
from utils.evaluate import evaluate_model
from utils.metrics_recorder import MetricsRecorder
from models.simple_model import SimpleModel
from training.local_training import local_training
from utils.split_data import split_data
from attacks.model_replacement import model_replacement_attack

def attacked_federated_learning(
        clean_global_model,
        X_train, y_train, X_test, y_test,
        num_rounds, num_clients, batch_size, local_epochs, lr,
        attacker_ids, attack_rounds, scaling_factor
    ):
    input_size = X_train.shape[1]
    num_classes = 2

    global_model = SimpleModel(input_size, num_classes)
    global_model.load_state_dict(copy.deepcopy(clean_global_model.state_dict()))

    client_models = [SimpleModel(input_size, num_classes) for _ in range(num_clients)]
    datasets = split_data(X_train, y_train, num_clients)

    aggregator = ICA()
    recorder = MetricsRecorder()

    for rnd in range(1, num_rounds + 1):
        print(f"Round {rnd}/{num_rounds}")

        for cm in client_models:
            cm.load_state_dict(copy.deepcopy(global_model.state_dict()))

        client_losses = []

        for cid, ds in enumerate(datasets):
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            losses = local_training(client_models[cid], loader, local_epochs, lr)
            client_losses.append(float(sum(losses) / len(losses)))

            if (cid in attacker_ids) and (rnd - 1 in attack_rounds):
                print(f"   -> Client {cid} performs model replacement")
                replaced = model_replacement_attack(
                    global_model.state_dict(),
                    client_models[cid].state_dict(),
                    scaling_factor
                )
                client_models[cid].load_state_dict(replaced)

        aggregator.run(global_model, client_models)

        metrics = evaluate_model(global_model, X_test, y_test)
        recorder.record(rnd, metrics, float(sum(client_losses)/len(client_losses)))

        print(f" Acc: {metrics['accuracy']:.4f}, F1_w: {metrics['f1_weighted']:.4f}")

    return global_model, recorder.get_history()
