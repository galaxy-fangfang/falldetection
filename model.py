import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import math
class CNNModel1D(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel1D, self).__init__()
        seq_len, n_vars  = input_shape
        self.conv1 = nn.Conv1d(in_channels=n_vars,
                                out_channels=16, 
                                kernel_size=3,   
                                padding=1, stride=2)       # stride 2
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16,
                                out_channels=32, 
                                kernel_size=3,   
                                padding=1, stride=2)       # stride 2
        # Compute the flattened feature size after convolution
        linear_input_size = 32 * ((((seq_len+1) // 2)+1)//2)# 32 channels
        
        # Fully connected layer
        self.fc1 = nn.Linear(linear_input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        x = x.reshape(-1,x.shape[2], x.shape[3]).permute(0,2,1) # [32, 1, 397, 6]->[32, 6, 397]
        x = self.conv1(x)  #[32, 6, 397]->[32, 32, 199]
        x = self.relu(x)    
        x = self.conv2(x)  #[32, 32, 199]->[32, 64, 100]
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1)  # Flatten to feed into FC layer
        x = self.dropout(x)
        x = self.fc1(x)     # Fully connected layer
        x = self.sigmoid(x) # Sigmoid activation for binary classification
        
        return x

# class CNNModel1D(nn.Module):
#     #same model as: https://www.kaggle.com/code/sankalpsinghvishen/p2-cnn-svm-model-graphs-full-dataset
#     def __init__(self, input_shape):
#         super(CNNModel1D, self).__init__()
#         seq_len, n_vars  = input_shape
#         self.conv1 = nn.Conv1d(in_channels=n_vars,
#                                 out_channels=32, 
#                                 kernel_size=3,   
#                                 padding=1, stride=2)       # stride default 1
#         self.relu = nn.ReLU()
#         # Compute the flattened feature size after convolution and pooling
#         linear_input_size = 32 * ((seq_len+1) // 2)# 32 channels
        
#         # Fully connected layer
#         self.fc1 = nn.Linear(linear_input_size, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         # import ipdb;ipdb.set_trace()
#         x = x.reshape(-1,x.shape[2], x.shape[3]).permute(0,2,1) # [32, 1, 397, 6]->[32, 6, 397]
#         x = self.conv1(x)  #[32, 6, 397]->[32, 32, 199]
#         x = self.relu(x)    
#         x = torch.flatten(x, start_dim=1)  # Flatten to feed into FC layer
#         x = self.dropout(x)
#         x = self.fc1(x)     # Fully connected layer
#         x = self.sigmoid(x) # Sigmoid activation for binary classification
        
#         return x

class CNNModel(nn.Module):
    #same model as: https://www.kaggle.com/code/sankalpsinghvishen/p2-cnn-svm-model-graphs-full-dataset
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        input_width, input_height  = input_shape
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=32, 
                                kernel_size=3,   
                                padding=1)       # stride default 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.relu = nn.ReLU()

        # Compute the flattened feature size after convolution and pooling
        linear_input_size = 32 * (input_height // 2) * (input_width // 2)  # 32 channels
        
        # Fully connected layer
        self.fc1 = nn.Linear(linear_input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        x = self.conv1(x)  
        x = self.relu(x)    
        x = self.pool(x)    
        x = torch.flatten(x, start_dim=1)  # Flatten to feed into FC layer
        x = self.fc1(x)     # Fully connected layer
        x = self.sigmoid(x) # Sigmoid activation for binary classification
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_heads=4, num_layers=1):
        super(TransformerModel, self).__init__()
        
        seq_len, self.n_vars = input_size
        # Project input features to hidden_size (embedding layer)
        self.embedding = nn.Linear(self.n_vars, hidden_size)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=hidden_size * 4, 
                                                   dropout=0.1, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer for classification
        self.fc1 = nn.Linear(seq_len*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embedding(x)  # Project input to hidden size: (32, 1, 794, 6)-> (32, 1, 794, 32)
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) #  (32, 1, 794, 32)-> (32, 794, 32)
        x = self.transformer(x)  # Pass through Transformer layers: (32, 794, 32)->(32, 794, 32)
        
        # # Global average pooling over time dimension
        x = torch.flatten(x, start_dim=1) # (32, 794, 32)-> (32, 794 * 32)
        x =  self.dropout(x)
        x = self.fc1(x)  # (32, 794 * 32)->(32, 1)
        x = self.sigmoid(x)  # Binary classification output
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class PatchTST(nn.Module):
    # similar model to: https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py
    def __init__(self, input_size, hidden_size=32, patch_len=16, stride=8, num_heads=4, num_layers=1):
        seq_len, n_var = input_size
        super(PatchTST, self).__init__()
        
        padding=stride
        self.d_model = hidden_size
        # Project input features to hidden_size (embedding layer)
        self.patch_embedding = PatchEmbedding(
            self.d_model, patch_len, stride, padding, 0.1)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=hidden_size * 4, 
                                                   dropout=0.1, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.1)
        self.head_nf = self.d_model * \
                       int((seq_len - patch_len) / stride + 2) # d_model * patch_num
        self.projection = nn.Linear(
                self.head_nf * n_var, 1)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(-1,x.shape[-2], x.shape[-1]) # (32, 1,794, 6)->(32, 794, 6)
        x = x.permute(0, 2, 1) # (32, 794, 6) -> (32, 6, 794)
        x, n_vars = self.patch_embedding(x)  # Project input to hidden size: (32, 6, 794)-> (32*6, 99, 32), num_patch=794/8
        
        x = self.transformer(x)  # Pass through Transformer layers: (32*6, 99, 32)->(32*6, 99, 32)
        x = x.reshape(bs, -1) # [32*6, 99, 32] -> [32, 6* 99* 32]
        x = self.dropout(x)
        x = self.projection(x)  # Fully connected layer
        x = self.sigmoid(x)  # Binary classification output
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, bidirectional=False, num_classes=1):
        super(LSTMModel, self).__init__()
        seq_len, n_var = input_size
        self.lstm = nn.LSTM(input_size=n_var, # The number of expected features in the input x
                            hidden_size=hidden_size, # The number of features in the hidden state h
                            num_layers=num_layers, # Number of recurrent layers.
                            bidirectional=bidirectional,
                            dropout=0.1,
                            batch_first=True) # [batch, sequence, feature] mode is on.
        self.direction_factor = 2 if bidirectional else 1

        self.fc = nn.Linear(hidden_size*self.direction_factor, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        ### x: (batch_size, sequence_length, input_size)
        ### Set initial hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers * self.direction_factor, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * self.direction_factor, x.size(0), self.lstm.hidden_size).to(x.device)

        ### Forward LSTM
        x = x.reshape(-1, x.shape[2], x.shape[3]) # [32, 1, 397, 6]->[32, 397, 6]
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # [32, 397, 6]->[32, 397, 32]
        out = out[:,-1,:] # Decode the hidden state of the last time step
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self, input_shape, hidden_size=32, num_layers=2, bidirectional=False, num_classes=1):
        super(CNNLSTMModel, self).__init__()
        seq_len, n_vars  = input_shape
        self.conv1 = nn.Conv1d(in_channels=n_vars,
                                out_channels=16, 
                                kernel_size=3,   
                                padding=1, stride=2)       # stride 2
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=16, # The number of expected features in the input x
                            hidden_size=hidden_size, # The number of features in the hidden state h
                            num_layers=num_layers, # Number of recurrent layers.
                            bidirectional=bidirectional,
                            dropout=0.1,
                            batch_first=True) # [batch, sequence, feature] mode is on.
        self.direction_factor = 2 if bidirectional else 1

        self.fc = nn.Linear(hidden_size*self.direction_factor, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        ### Conv
        x = x.reshape(-1,x.shape[2], x.shape[3]).permute(0,2,1) # [32, 1, 397, 6]->[32, 6, 397]
        x = self.conv1(x)  #[32, 6, 397]->[32, 16, 199]
        x = self.relu(x) # [32, 16, 199]

        ### Set initial hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers * self.direction_factor, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * self.direction_factor, x.size(0), self.lstm.hidden_size).to(x.device)

        ### Forward LSTM   [32, 16, 199]
        x = x.permute(0,2,1) # [32, 16, 199]->[32, 199, 16]
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # [32, 199, 16]->[32, 199, 32]
        out = out[:,-1,:] # Decode the hidden state of the last time step
        out = self.fc(out)
        return self.sigmoid(out)