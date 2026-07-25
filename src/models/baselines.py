import torch
import torch.nn as nn
import math

class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BaselineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch, seq_len, hidden_size)
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, output_size, dropout=0.1):
        super(BaselineTransformer, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.output_linear = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_linear(x)
        # Permute for positional encoding: (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        # Permute back: (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)
        
        output = self.transformer_encoder(x)
        output = self.output_linear(output)
        return output
