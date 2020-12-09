import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score


class RCNN(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 vocab_size, 
                 num_layers = 1,
                 hidden_size=64, 
                 dropout=0.8,
                 word_embeddings=None):

        super(RCNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = dropout,
                            bidirectional = True)

        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(embedding_dim + 2*hidden_size, 128)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape = (seq_len, batch_size)
        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (seq_len, batch_size, embed_size)

        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)
        
        input_features = torch.cat([lstm_out,embedded_sent], 2).permute(1,0,2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
        
        linear_output = self.tanh(
            self.W(input_features)
        )
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)
        
        linear_output = linear_output.permute(0,2,1) # Reshaping fot max_pool
        
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)
        
        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        
        return self.sigmoid(final_out)
        # return final_out