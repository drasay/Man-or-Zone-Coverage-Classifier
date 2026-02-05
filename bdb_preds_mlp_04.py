import torch
import pandas as pd
import joblib
from bdb_training_mlp_03 import ManZoneMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
week_eval = 1

# Load Saved Tensors
features = torch.load("C:/python/nfl-big-data-bowl-2025/features_test_preds.pt").to(device)
targets = torch.load("C:/python/nfl-big-data-bowl-2025/targets_test_preds.pt")  # Not used for inference but useful for accuracy check
frame_ids = torch.load("C:/python/nfl-big-data-bowl-2025/frame_ids_test_preds.pt")

# Load Model
best_params = joblib.load(f"C:/python/nfl-big-data-bowl-2025/optim_results_mlp/man_zone_mlp_week{week_eval}_study.pkl").best_trial.params

model = ManZoneMLP(
        feature_len=5,
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        output_dim=2
    ).to(device)

model.load_state_dict(torch.load(f"C:/python/nfl-big-data-bowl-2025/best_model_week{week_eval}_mlp.pth", weights_only=True))
model.eval()

# Inference
results = []
with torch.no_grad():
    for i in range(features.size(0)):
        x = features[i].unsqueeze(0)  # add batch dimension
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        zone_prob, man_prob = probs[0], probs[1]
        pred = int(man_prob > zone_prob)
        actual = int(targets[i].item()) if targets is not None else None

        results.append({
            "frameUniqueId": frame_ids[i],
            "zone_prob": zone_prob,
            "man_prob": man_prob,
            "pred": pred,
            "actual": actual
        })

# Save Predictions
df_results = pd.DataFrame(results)
df_results.to_csv(f"C:/python/nfl-big-data-bowl-2025/week_{week_eval}_preds_mlp.csv", index=False)
print("Inference complete and saved.")
