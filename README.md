# FL-ICA-Public
**Federated Learning with Impact-Calculated Aggregation (ICA) — Public Demonstration (PyTorch)**

---

## Important Notice
This repository contains a **simplified prototype** of the defense architecture developed in our research.

**Confidentiality and double-blind review compliance:**
- Results, datasets, hyperparameters, and the full implementation from the manuscript under review are **not included**.
- A **complete release** with the final ICA method and experimental data will be provided **after acceptance**.

This demo is intended solely to illustrate the **engineering workflow, modular code design, and federated learning pipeline**.

---

## 1. Overview
This repository provides a clean, modular, and fully executable demonstration pipeline for **Federated Learning (FL)** with the following capabilities:

- Placeholder implementation of the **ICA algorithm** (public demo only)
- **Synthetic dataset generation**; no external datasets included
- Local training of multiple client models
- Simulation of **benign and adversarial federated rounds**
- Minimal **model replacement attack module** for demonstration purposes
- Evaluation, metrics tracking, and lightweight visualization

**Purpose of this demo:**
- Showcase **implementation quality and reproducible engineering workflow**
- Demonstrate **modular code structure** suitable for research software
- Maintain **protection of proprietary algorithms and experimental data**

---

## 2. Repository Structure
```
FL-ICA-Public/
│
├── .gitignore
├── README.md
├── requirements.txt
│
└── src/
   ├── run_demo.py # Entry point for the demo
   │
   ├── aggregator/
   │ └── ica.py # Placeholder ICA module
   │
   ├── attacks/
   │ └── model_replacement.py # Minimal demo of a model replacement attack
   │
   ├── federated/
   │ ├── safe_federated_learning.py # FL rounds without attacks
   │ └── attacked_federated_learning.py # FL rounds under adversarial conditions
   │
   ├── models/
   │ └── simple_model.py # Lightweight MLP model for demo
   │
   ├── training/
   │ └── local_training.py # Local client training loop
   │
   └── utils/
      ├── dataset_utils.py # Synthetic dataset generator
      ├── evaluate.py # Evaluation utilities
      ├── metrics_recorder.py # Metrics storage during FL rounds
      ├── plotting.py # Lightweight visualization
      └── split_data.py # Client data partitioning
```
---

## 3. Technical Description

The demo performs the following tasks:

1. **Synthetic Dataset Generation**
   - Classification datasets are fully synthetic with controllable features and noise
   - No real-world or paper-specific datasets are included

2. **Federated Client Simulation**
   - Multiple virtual clients are spawned
   - Each client trains an independent local model

3. **Federated Learning Execution**
   - **FedAvg** or pluggable aggregation can be run
   - ICA module included here is a **minimal placeholder**
   - No proprietary logic or research innovation is exposed

4. **Adversarial Scenario Simulation**
   - Toy "model replacement" attack demonstrates how the pipeline supports adversarial analysis

5. **Metrics Tracking and Visualization**
   - Accuracy, loss, divergence, and other indicators are recorded
   - Visualizations are generated using **synthetic data only**

---

## 4. Installation

```bash
git clone https://github.com/AmirhTork/FL_ICA_public.git
cd FL_ICA_public

# Create virtual environment
python -m venv venv
# Activate environment (Windows)
venv\Scripts\activate
# OR (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 5. Running the Demo

```bash
python src/run_demo.py
```
This script will:

- Generate a synthetic dataset
- Create N federated clients
- Execute a few benign FL rounds
- Optionally simulate one adversarial round
- Output evaluation logs and basic plots

> **Note:** All results are synthetic and intended solely for demonstration purposes.

---

## 6. ICA Aggregator (Public Placeholder)

The actual **Impact-Calculated Aggregation (ICA)** method from our research is **not included**.

`aggregator/ica.py` provides:

- Module structure and interfaces
- Expected data flow
- Minimal placeholder logic to allow pipeline execution

> This approach preserves the integrity of the scientific contribution while enabling reproducible demonstrations.

---

## 7. License

This demo is released for **evaluation and academic review only**.

- **Commercial or derivative research use of the ICA method is prohibited.**

