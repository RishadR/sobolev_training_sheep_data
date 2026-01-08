import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import RobustScaler

from custom_sobolev_training.data_gen import (
    generate_derivatives,
    CentralDifference,
    generate_dataloaders,
)
from custom_sobolev_training.training import sobolev_train
from custom_sobolev_training.complex_differences import LabelBinnedDifference

data = pd.read_csv("./data/combined_LLPSA2.csv")
data["experiment_id"] = pd.factorize(data["experiment_id"])[0]
data_np = data.to_numpy()
features_idx = list(range(10))
label_idx = 22
group_idx = 20
x_scaler = RobustScaler()
y_scaler = RobustScaler()
data_np[:, features_idx] = x_scaler.fit_transform(data_np[:, features_idx])
data_np[:, label_idx] = (y_scaler.fit_transform(data_np[:, label_idx].reshape(-1, 1)).flatten())

print(f"Y Scaler - Scale: {y_scaler.scale_}")
print(f"Y Scaler - Center: {y_scaler.center_}")

derivatives, original_data, kept_indices = generate_derivatives(
    data_np,
    feature_indices=features_idx,
    label_index=label_idx,
    group_index=group_idx,
    # derivative_type=CentralDifference(skip=1),
    derivative_type=LabelBinnedDifference(bin_width=0.05),
)
print(f"Original data Length: {data_np.shape[0]}")
print(f"Derivatives Length: {derivatives.shape[0]}")
print(f"Points Dropped: {data_np.shape[0] - original_data.shape[0]}")

dataset = np.concatenate((original_data, derivatives), axis=1)
modified_features_idx = features_idx + list(
    range(original_data.shape[1], dataset.shape[1])
)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

train_loader, val_loader = generate_dataloaders(
    dataset, modified_features_idx, [label_idx], group_idx, [0], 64, True, device
)

def evaluator(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            num_features = features.shape[1] // 2
            actual_features = features[:, :num_features]
            predictions = model(actual_features)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predictions.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    mae = np.mean(np.abs(all_labels - all_preds))
    return {"MAE": float(mae)}

model = torch.nn.Sequential(
    torch.nn.Linear(len(modified_features_idx) // 2, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
num_epochs = 50
alpha = 1e-4
sobolev_train(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    evaluator,
    num_epochs,
    alpha,
    None
)
