# NFL Defensive Coverage Classification

This project utilizes player tracking data to classify NFL defensive coverages into **Man** or **Zone** schemes. By comparing three distinct neural architectures, the study identifies which models best capture the spatial and temporal nuances of defensive play.

---

##  Project Overview
Defensive coverage identification is a cornerstone of modern football analytics. This project leverages raw tracking data (X/Y coordinates, velocity, and orientation) to predict defensive intent. 

### Data Source
> **Note on Data Acquisition:** The datasets used in this project are sourced from the **[2025 NFL Big Data Bowl on Kaggle](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025)**. This includes player-level tracking data, play metadata, and scouting labels.



### Model Performance Summary

* **Transformer:** The top performer with a **test accuracy of 0.8438**. It achieved a **peak pre-snap accuracy of 0.9365**, demonstrating the power of the self-attention mechanism in capturing player-to-player relationships.
* **Multi-Layer Perceptron (MLP):** A robust baseline that pools player features for a global classification.
* **Neural Additive Model (NAM):** An interpretable architecture that evaluates the contribution of individual features (like velocity or positioning) independently.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `bdb_cleaning_functions_01.py` | Core preprocessing: coordinate rotation, left-to-right normalization, and velocity component calculation. |
| `bdb_dataloading_02.py` | Data ingestion, snap-alignment, and Gaussian-weighted **data augmentation**. |
| `bdb_training_..._03.py` | Training scripts featuring **Bayesian Optimization** via Optuna for hyperparameter tuning. |
| `bdb_preds_..._04.py` | Inference scripts that load the optimized model weights and generate test set predictions. |
| `bdb_evaluation_..._05.py` | Visualization scripts to analyze accuracy relative to the ball snap ($T=0$). |

---

## Data Pipeline & Features
The pipeline transforms raw CSV tracking data into standardized PyTorch tensors.
* **Standardization:** All plays are normalized to a left-to-right direction.
* **Features:** Inputs include `x_clean`, `y_clean`, `v_x`, `v_y`, and a `defense` toggle.
* **Temporal Focus:** The model filters for frames between -15.0 and +5.0 seconds relative to the snap, with augmentation centered on the pre-snap window.

---

## Results & Evaluation
The project evaluates accuracy across the "life" of a play. Results indicate that defensive schemes are most predictable in the **pre-snap phase** (approx. 1 second before the ball moves), where player alignments provide the strongest signals for the Transformer's attention heads.



---

##  Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* CUDA (optional, for GPU acceleration)


### Workflow
1.  **Prepare Data:** Run `bdb_dataloading_02.py` to generate `.pt` tensors from the Kaggle CSV files.
2.  **Optimize & Train:** Execute the `bdb_training_..._03.py` scripts. This will use **Bayesian Optimization** to find the best hyperparameters.
3.  **Generate Predictions:** Run `bdb_preds_..._04.py` to produce a CSV of results.
4.  **Evaluate:** Use the `bdb_evaluation_..._05.py` scripts to generate performance visualizations.
