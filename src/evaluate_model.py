import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, X, y, scaler):
    if scaler is None:
        X_scaled = X
    else:
        X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze().numpy()
        y_pred_labels = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y, y_pred_labels)
    return accuracy, y_pred_labels
