import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, brier_score_loss, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt
from shap import Explainer

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, seq_length=30):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_length], self.labels[idx+self.seq_length]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train_lstm(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_loss = float('inf')
    early_stop_count = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = evaluate_loss(model, val_loader, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                break
    return model

def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels in loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_lstm(model, test_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            preds.extend(outputs.squeeze().numpy())
            trues.extend(labels.squeeze().numpy())
    auc = roc_auc_score(trues, preds)
    precision, recall, _ = precision_recall_curve(trues, preds)
    pr_auc = np.trapz(recall, precision)
    brier = brier_score_loss(trues, preds)
    prob_true, prob_pred = calibration_curve(trues, preds, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    return auc, pr_auc, brier, ece

def cross_validate_lstm(data, labels, input_size, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(data):
        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        train_dataset = TimeSeriesDataset(train_data, train_labels)
        val_dataset = TimeSeriesDataset(val_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        model = LSTMModel(input_size)
        model = train_lstm(model, train_loader, val_loader)
        auc, _, _, _ = evaluate_lstm(model, val_loader)
        scores.append(auc)
    return np.mean(scores)

def interpret_model(model, data, feature_names):
    explainer = Explainer(model, data)
    shap_values = explainer(data)
    importances = np.mean(np.abs(shap_values.values), axis=0)
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return ranked

def plot_roc_curve(trues, preds, output_file):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(trues, preds)
    plt.plot(fpr, tpr)
    plt.savefig(output_file)
    plt.close()

def plot_calibration_curve(trues, preds, output_file, n_bins=10):
    prob_true, prob_pred = calibration_curve(trues, preds, n_bins=n_bins)
    plt.plot(prob_pred, prob_true)
    plt.savefig(output_file)
    plt.close()
