import os
import gc
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class AdultDataset(Dataset):
    def __init__(self, data, catcols, target):
        self.data = data
        self.catcols = catcols
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = []
        for c in self.catcols:
            lbl_enc = LabelEncoder()
            lbl_enc.fit(self.data[c])
            emb_dim = int(min(np.ceil((len(lbl_enc.classes_)) / 2), 50))
            emb = nn.Embedding(len(lbl_enc.classes_), emb_dim)
            emb.weight.data.copy_(torch.tensor(lbl_enc.transform(self.data[c].values[idx])))
            x.append(emb.weight)
        x = torch.cat(x, dim=0).flatten(0, 1)
        y = torch.tensor(self.target[idx])
        return x, y

def create_model(data, catcols):
    inputs = []
    outputs = []
    for c in catcols:
        num_unique_values = len(np.unique(data[c]))
        embed_dim = int(min(np.ceil((num_unique_values) / 2), 50))
        inp = torch.nn.functional.one_hot(torch.tensor(0), num_classes=num_unique_values + 1)
        out = nn.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
        out = nn.SpatialDropout1D(0.3)(out)
        out = nn.Flatten()(out)
        inputs.append(inp)
        outputs.append(out)
    x = nn.Concatenate()(outputs)
    x = nn.BatchNorm1d(x.shape[1])(x)
    x = nn.Linear(x.shape[1], 300)(x)
    x = nn.Dropout(0.3)(x)
    x = nn.BatchNorm1d(300)(x)
    x = nn.Linear(300, 300)(x)
    x = nn.Dropout(0.3)(x)
    x = nn.BatchNorm1d(300)(x)
    y = nn.Linear(300, 2)(x)
    model = nn.Sequential(*inputs, y)
    return model

def run(fold):
    df = pd.read_csv("./adult_folds.csv")
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        lbl_enc = LabelEncoder()
        df.loc[:, col] = lbl_enc.fit_transform(df[col].values)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)
    model = create_model(df_train, features)
    dataset_train = AdultDataset(df_train, features, df_train.target.values)
    dataset_valid = AdultDataset(df_valid, features, df_valid.target.values)
    dataloader_train = DataLoader(dataset_train, batch_size=1024, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1024, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(3):
        model.train()
        total_loss = 0
        for x, y in dataloader_train:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader_train)}')
        model.eval()
        valid_preds = torch.sigmoid(model(dataset_valid.data)).detach().numpy()[:, 1]
        print(roc_auc_score(dataset_valid.target, valid_preds))
        torch.cuda.empty_cache()
if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)
