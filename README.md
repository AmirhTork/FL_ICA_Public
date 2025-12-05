# FL-ICA-Public

**Federated Learning with Impact-Calculated Aggregation (ICA)** — public demo (PyTorch)  


Important Note:
This repository contains a simplified prototype of the defense framework for demonstration purposes.
The results, datasets, hyperparameters, and full implementation used in the manuscript under review are not included for confidentiality and double-blind review compliance.
A complete release will be provided after acceptance.


-----------------------------------


🚀 Overview

This repository provides a clean, modular, fully executable demonstration pipeline for Federated Learning (FL) with support for:

ICA algorithm (placeholder ICA module)

Synthetic dataset generation (no external datasets required)

Local model training for multiple clients

Simulation of benign and adversarial federated rounds

Minimal model replacement attack module for demonstration

Evaluation, metrics tracking, and lightweight visualization


The goal of this repository is to showcase:

Implementation quality

Federated learning engineering workflow

Code structure, modularity, and research software design


without releasing the proprietary aggregation algorithm or experimental data from our paper.


-----------------------------------


🧩 Repository Structure

    FL_ICA_public/
    │
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    │
    └── src/
        ├── run_demo.py                   # Entry point for running the demo
        │
        ├── aggregator/
        │   └── ica.py                    # Public placeholder ICA module
        │
        ├── attacks/
        │   └── model_replacement.py      # Minimal demonstration of a model replacement attack
        │
        ├── federated/
        │   ├── safe_federated_learning.py      # Safe FL rounds to train global model
        │   └── attacked_federated_learning.py  # FL rounds under attack
        │
        ├── models/
        │   └── simple_model.py           # Lightweight MLP used for demo
        │
        ├── training/
        │   └── local_training.py         # Local client SGD training loop
        │
        └── utils/
            ├── dataset_utils.py          # Synthetic dataset generator
            ├── evaluate.py               # Evaluation utilities
            ├── metrics_recorder.py       # Stores metrics during FL rounds
            ├── plotting.py               # Simple plots for demo
            └── split_data.py             # Client data partitioning

-----------------------------------


🧪 What This Demo Does (Technically)

✔ Generates a synthetic classification dataset

No real-world or paper‑related dataset is included.

The demo uses controllable random features + noise.

✔ Spawns multiple virtual FL clients

Each with independent, locally trained models.

✔ Runs FedAvg with a pluggable aggregator

The ICA file included here is a minimal placeholder

It only demonstrates the interface and workflow

No paper-specific logic or innovation is exposed

✔ Can simulate an adversarial client

Using a toy "model replacement" demonstration attack to show how the pipeline supports adversarial analysis.

✔ Tracks metrics

Accuracy, loss, divergence, and other lightweight indicators.

✔ Visualizes demo results

All plots are synthetic and for demonstration only.


-----------------------------------


⚙️ Installation

git clone https://github.com/AmirhTork/FL_ICA_public.git
cd FL_ICA_public
pip install -r requirements.txt


-----------------------------------


▶️ Running the Demo

python src/run_demo.py

This will:

Generate a synthetic dataset

Create N federated clients

Run a few benign federated rounds

Optionally simulate one adversarial round

Output evaluation logs and simple plots

All results are synthetic and random.


-----------------------------------

🧱 ICA Aggregator (Public Placeholder)

The real ICA aggregation method developed in my research is not included.

Instead, aggregator/ica.py provides:

the architecture

the interfaces

the expected data flow

an extremely simplified placeholder version


This allows the pipeline to run while keeping the scientific contribution private.


-----------------------------------


📄 License


This demo version is released for evaluation and academic review only.
Commercial or derivative research use of the ICA method is not permitted.

-----------------------------------
