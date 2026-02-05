import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
import optuna
from optuna.trial import Trial
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import os
pd.options.mode.chained_assignment = None

class ManZoneMLP(nn.Module):
    def __init__(self, feature_len=5, hidden_dim=64, num_layers=2, dropout=0.1, output_dim=2):
        super(ManZoneMLP, self).__init__()
        layers = [nn.Linear(feature_len, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, num_players, feature_len)
        x = x.mean(dim=1)  # Pool over players
        return self.network(x)


def get_dataloaders(week_eval, batch_size, device):
    """Load data and create dataloaders for a specific week"""
    base_path = "C:/python/nfl-big-data-bowl-2025"
    
    train_features = torch.load(f"{base_path}/features_training_week{week_eval}preds.pt").to(device)
    train_targets = torch.load(f"{base_path}/targets_training_week{week_eval}preds.pt").to(device)
    
    val_features = torch.load(f"{base_path}/features_val_week{week_eval}preds.pt").to(device)
    val_targets = torch.load(f"{base_path}/targets_val_week{week_eval}preds.pt").to(device)
    
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, device, 
                       num_epochs, early_stopping_patience=5, checkpoint_path=None):
    """Train and evaluate the model, with early stopping"""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for val_features_batch, val_targets_batch in val_loader:
                val_features_batch = val_features_batch.to(device)
                val_targets_batch = val_targets_batch.to(device)
                
                val_outputs = model(val_features_batch)
                val_loss = loss_fn(val_outputs, val_targets_batch)
                
                val_running_loss += val_loss.item() * val_features_batch.size(0)
                
                _, predicted = torch.max(val_outputs, 1)
                correct += (predicted == val_targets_batch).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            # Save the best model
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered. Best version saved at {checkpoint_path}")
                break
    
    # Return the best validation metrics
    return best_val_loss, val_accuracy, train_losses, val_losses, val_accuracies

def objective(trial: Trial, week_eval, device, feature_len=5, output_dim=2, 
              max_epochs=None, base_path="C:/python/nfl-big-data-bowl-2025"):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameters to optimize
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)  
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])  
    num_layers = trial.suggest_int("num_layers", 1, 3)  
    dropout = trial.suggest_float("dropout", 0.0, 0.3)  

    # Adaptive trial duration - use fewer epochs for early trials
    if max_epochs is None:
        # Dynamic epochs based on trial number
        if trial.number < 10:
            actual_epochs = 10  # Quick initial exploration
        elif trial.number < 15:
            actual_epochs = 20  # Medium-length for refinement
        else:
            actual_epochs = 30  # Full duration for promising areas
    else:
        actual_epochs = max_epochs
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(week_eval, batch_size, device)
    
    # Initialize model with hyperparameters
    model = ManZoneMLP(
        feature_len=feature_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    # Setup early stopping
    early_stopping_patience = trial.suggest_int("early_stopping_patience", 3, 7)
    
    # Train and evaluate the model - store trial models in a separate trials directory
    trials_dir = f"{base_path}/trials"
    os.makedirs(trials_dir, exist_ok=True)
    checkpoint_path = f"{trials_dir}/trial_{trial.number}_week{week_eval}_mlp.pth"
    
    print(f"\nTrial #{trial.number}: Running with {actual_epochs} epochs")
    best_val_loss, val_accuracy, _, _, _ = train_and_evaluate(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        actual_epochs, early_stopping_patience, checkpoint_path
    )
    
    # Report intermediate values - helps with pruning
    trial.report(val_accuracy, actual_epochs)
    
    # Save the best validation accuracy for this trial
    trial.set_user_attr("best_val_accuracy", val_accuracy)
    trial.set_user_attr("epochs_used", actual_epochs)
    
    return val_accuracy  # We want to maximize accuracy

def run_bayesian_optimization(week_eval, min_trials=5, max_trials=20, convergence_threshold=0.005,
                              patience=5, study_name=None, save_path=None):
    """Run Bayesian optimization for hyperparameter tuning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if study_name is None:
        study_name = f"man_zone_mlp_week{week_eval}"
    
    if save_path is None:
        # Use the original base path
        save_path = "C:/python/nfl-big-data-bowl-2025"
        
    # Create a subdirectory for optimization artifacts without affecting main model paths
    optim_save_path = f"{save_path}/optim_results_mlp"
    
    # Create directory for study results if it doesn't exist
    os.makedirs(optim_save_path, exist_ok=True)
    
    # Create or load study
    storage_name = f"sqlite:///{optim_save_path}/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",  # We want to maximize validation accuracy
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    print(f"Starting adaptive Bayesian optimization (min_trials={min_trials}, max_trials={max_trials})")

    best_value = study.best_value if len(study.trials)> 0 else -np.inf
    no_improvement_count = 0

    while len(study.trials) < max_trials:
        previous_best_value = best_value

        study.optimize(lambda trial: objective(trial, week_eval, device), n_trials=1, timeout=None, show_progress_bar=False)

        current_best_value = study.best_value
        improvement = current_best_value - previous_best_value

        if improvement > convergence_threshold:
            best_value = current_best_value
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print(f"Trial {len(study.trials)}: best value = {current_best_value:.5f}, improvement = {improvement:.5f}, no_improvement_count = {no_improvement_count}")

        if len(study.trials) >= min_trials and no_improvement_count >= patience:
            print(f"Stopping early due to convergence after {len(study.trials)} trials.")
            break

    best_trial = study.best_trial
    print(f"\nBest trial (#{best_trial.number}):")
    print(f"Value (Validation Accuracy): {best_trial.value:.4f}")
    print("Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("Saving study with joblib...")
    joblib.dump(study, f"{optim_save_path}/{study_name}_study.pkl")
    print("Study saved.")

    print("Generating optimization history plot...")
    fig1 = plot_optimization_history(study)
    fig1.write_html(f"{optim_save_path}/{study_name}_optimization_history.html")
    print("Optimization history plot saved.")

    print("Generating parameter importances plot...")
    fig2 = plot_param_importances(study)
    fig2.write_html(f"{optim_save_path}/{study_name}_param_importances.html")
    print("Parameter importances plot saved.")

    # Final training with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    best_params = best_trial.params

    train_loader, val_loader = get_dataloaders(week_eval, best_params["batch_size"], device)

    model = ManZoneMLP(
        feature_len=5,
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        output_dim=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    final_checkpoint_path = f"C:/python/nfl-big-data-bowl-2025/best_model_week{week_eval}_mlp.pth"
    _, final_val_accuracy, train_losses, val_losses, val_accuracies = train_and_evaluate(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        num_epochs=30, early_stopping_patience=best_params["early_stopping_patience"],
        checkpoint_path=final_checkpoint_path
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig(f"{optim_save_path}/week{week_eval}_bayesian_training_curves.png")

    print(f"\nFinal model saved to {final_checkpoint_path}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")

    return study, model

if __name__ == "__main__":
    # Specify which weeks to train on
    weeks_train = [1]
    
    # Run Bayesian optimization for each week
    for week_eval in weeks_train:
        print(f"\n{'#' * 20} WEEK {week_eval} - BAYESIAN OPTIMIZATION {'#' * 20}\n")
        
        # Use adaptive convergence-based optimization instead of fixed trials
        study, best_model = run_bayesian_optimization(
            week_eval=week_eval,
            min_trials=5,     # Minimum trials before checking convergence
            max_trials=20,     # Maximum trials as a failsafe
            convergence_threshold=0.005,  # 0.5% improvement threshold
            patience=5         # Stop after 5 trials with minimal improvement
        )
        
        print(f"\n{'#' * 20} WEEK {week_eval} - OPTIMIZATION COMPLETE {'#' * 20}\n")