#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model import CNNModel, TransformerModel, PatchTST, LSTMModel, CNNModel1D, CNNLSTMModel

from sklearn.preprocessing import StandardScaler
import argparse


def loaddata(sampling_hz=20,device='cuda'):
    # Load data from the pickle file
    df = pd.read_pickle('./FallAllD_pkl/FallAllD.pkl')
    df['Target'] = np.where(df['ActivityID'].between(1, 44), 0, 1)
    # Define the number of samples
    num_samples = 4760  # Assuming 238 Hz * 20 seconds
    sampling_hz = sampling_hz#238#40
    sampling_interval = int(np.ceil(238/sampling_hz))

    # Preprocess data for CNN
    X_cnn_acc = df['Acc'] / 10000.0  # Normalize the data
    X_cnn_gyr = df['Gyr'] / 10000.0  # Normalize the data

    X_cnn_acc = X_cnn_acc.apply(lambda x: x[:num_samples:sampling_interval])  #
    X_cnn_gyr = X_cnn_gyr.apply(lambda x: x[:num_samples:sampling_interval])  #

    # Convert to NumPy arrays
    X_cnn_acc = np.array(X_cnn_acc.tolist())
    X_cnn_gyr = np.array(X_cnn_gyr.tolist())
    # Concatenate the arrays
    X_cnn = np.concatenate((X_cnn_acc, X_cnn_gyr), axis=2)
    y_cnn = df['Target']

    feature_num, var_num = X_cnn.shape[1], X_cnn.shape[2]
    inputshape = feature_num, var_num

    # Split the data into training and testing sets for CNN
    X_cnn_train, X_cnn_test, y_cnn_train, y_cnn_test = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)

    num_all = X_cnn.shape[0]
    num_test = X_cnn_test.shape[0]
    num_val = int(num_all*0.1)

    X_cnn_val, y_cnn_val = X_cnn_train[-num_val:], y_cnn_train[-num_val:]
    X_cnn_train, y_cnn_train = X_cnn_train[:-num_val], y_cnn_train[:-num_val]

    # print(y_cnn_test.value_counts())
    # print(y_cnn_val.value_counts())
    # print(y_cnn_train.value_counts())

    # Convert NumPy arrays to PyTorch tensors
    X_cnn_train_tensor = torch.tensor(X_cnn_train, dtype=torch.float32)
    X_cnn_test_tensor = torch.tensor(X_cnn_test, dtype=torch.float32)
    X_cnn_val_tensor = torch.tensor(X_cnn_val, dtype=torch.float32)
    print(X_cnn_train.shape, X_cnn_val.shape, X_cnn_test.shape)

    X_cnn_train_tensor = X_cnn_train_tensor.view(-1, 1, feature_num, 6)  # Set channels to 1
    X_cnn_test_tensor = X_cnn_test_tensor.view(-1, 1, feature_num, 6)
    X_cnn_val_tensor = X_cnn_val_tensor.view(-1, 1, feature_num, 6)

    y_cnn_train_tensor = torch.tensor(y_cnn_train.to_numpy(), dtype=torch.float32).unsqueeze(1)  # Add a dimension for compatibility
    y_cnn_test_tensor = torch.tensor(y_cnn_test.to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_cnn_val_tensor = torch.tensor(y_cnn_val.to_numpy(), dtype=torch.float32).unsqueeze(1)

    # Create TensorDataset
    train_dataset = TensorDataset(X_cnn_train_tensor, y_cnn_train_tensor)
    test_dataset = TensorDataset(X_cnn_test_tensor, y_cnn_test_tensor)
    val_dataset = TensorDataset(X_cnn_val_tensor, y_cnn_val_tensor)

    # Define DataLoaders
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)} Testing samples: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, inputshape

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute False Positive Rate
    fpr = fp / (fp + tn)
    return fpr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def loadmodel(modelname, inputdim, device):
    if modelname == 'transformer':
        cnn_model=TransformerModel(inputdim).to(device)
    elif modelname == 'PatchTST': #problematic
        cnn_model = PatchTST(inputdim, num_layers=1).to(device)
    elif modelname == 'cnn2d':
        cnn_model=CNNModel(inputdim).to(device)
    elif modelname == 'cnn1d':
        cnn_model=CNNModel1D(inputdim).to(device)
    elif modelname == 'lstm':
        cnn_model=LSTMModel(inputdim,hidden_size=32, num_layers=2,bidirectional=False).to(device)
    elif modelname == 'cnnlstm':
        cnn_model = CNNLSTMModel(inputdim,hidden_size=32, num_layers=1,bidirectional=False).to(device)
    else:
        print('Invalid model name!')
        exit()
    return cnn_model

def train(train_loader, inputdim, modelname='cnn1d', num_epochs=50, num_runs=1, num_patience=10,device='cuda'):
    criterion = nn.BCELoss() # Binary Cross Entropy for binary classification
    history = {"train_loss": [], "val_loss": []}

    for run in range(num_runs):
        path = './checkpoint/%s_%d.pth'%(modelname, run)
        # if os.path.exists(path):
            # continue
        train_loss = 0
        ### Init model
        cnn_model = loadmodel(modelname, inputdim, device)
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001) # Adam optimizer

        ### Training
        early_stopping = EarlyStopping(patience=num_patience, verbose=True)
        for epoch in range(num_epochs):
            cnn_model.train(True) # Set model to train mode
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()  # Clear gradients
                outputs = cnn_model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            cnn_model.eval()  # Set model to evaluation mode
            val_loss = 0

            with torch.no_grad():
                for j, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = cnn_model(images)
                    loss = criterion(outputs, labels.float())
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            early_stopping(val_loss, cnn_model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        torch.save(cnn_model.state_dict(), path)

def testing(test_loader, inputdim, modelname='cnn1d', num_runs=1, score_thre=0.95, device='cuda'):
    cnn_model=loadmodel(modelname, inputdim, device)
    ### Testing
    score_thre=score_thre
    results = {'acc': [], 'fpr': [], 'roc': []}

    for run in range(num_runs):
        # load model
        path = './checkpoint/%s_%d.pth'%(modelname, run)
        cnn_model.load_state_dict(torch.load(path))

        # Evaluate models
        cnn_model.eval()
        outputs = []
        y_cnn_test = []
        with torch.no_grad():
            for images, labels in test_loader:
                y_cnn_test.extend(labels)

                images, labels = images.to(device), labels.to(device)
                output = cnn_model(images)
                outputs.extend(output.cpu().numpy().flatten())

        y_cnn_test = np.array(y_cnn_test)
        y_cnn_pred = np.array(outputs)
        y_cnn_pred = (y_cnn_pred > score_thre).astype(int)

        # Calculate accuracy and other metrics for CNN
        cnn_accuracy = accuracy_score(y_true=y_cnn_test, y_pred=y_cnn_pred)
        cnn_fpr = false_positive_rate(y_cnn_test, y_cnn_pred)
        cnn_roc = roc_auc_score(y_true=y_cnn_test, y_score=y_cnn_pred)

        # print("%s Accuracy: %.2f"%(modelname, cnn_accuracy*100))
        # print("%s false positive rate: %.2f"%(modelname, cnn_fpr*100))
        # print("%s AUCROC: %.2f"%(modelname, cnn_roc*100))

        results['acc'].append(cnn_accuracy)
        results['fpr'].append(cnn_fpr)
        results['roc'].append(cnn_roc)

    print('%d runs mean results:'%num_runs)
    print("%s Accuracy: %.2f, %.2f"%(modelname, np.mean(results['acc'])*100, np.std(results['acc'])*100))
    print("%s false positive rate: %.2f, %.2f"%(modelname, np.mean(results['fpr'])*100, np.std(results['fpr'])*100))
    print("%s AUCROC: %.2f, %.2f"%(modelname, np.mean(results['roc'])*100, np.std(results['roc'])*100))

    print('------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--modelname', type=str, required=True, help="model's name")
    parser.add_argument('-r', '--numruns', type=int, default=1)
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    modelname = args.modelname#'transformer'#'cnn'
    num_runs = args.numruns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, inputshape = loaddata(sampling_hz=20, device=device)

    num_epochs=300
    num_patience = 50
    if args.retrain:
        train(train_loader,
                inputdim=inputshape,
                modelname=modelname,
                num_epochs=num_epochs,
                num_runs=num_runs,
                num_patience=num_patience,
                device=device)
    testing(test_loader, inputdim=inputshape, modelname=modelname, num_runs=num_runs, score_thre=0.95,device=device)




### Confusion Matrix
# plt.figure(figsize=(4, 2))
# plt.subplot(1, 2, 1) # 1 rows, 2 columns 1 posn etc.
# cm_cnn = confusion_matrix(y_cnn_test, y_cnn_pred)
# sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title(f'CNN Confusion Matrix\nAccuracy: {cnn_accuracy:.2f}')
# plt.tight_layout()
# plt.show()



