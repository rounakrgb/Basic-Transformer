import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

#Positional encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  
        self.register_buffer('pe', pe)
        
    def forward(self,x):
        x = x + self.pe[:x.size(0)]

        return x

        
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt
    
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_ff, vocab_size, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.output_layer(tgt)
    
d_model = 128
nhead = 4
dim_ff = 512
num_layers = 2
vocab_size = 5000
seq_len = 10
batch_size = 4

decoder = TransformerDecoder(num_layers, d_model, nhead, dim_ff, vocab_size)
                  
tgt = torch.randint(0, vocab_size, (seq_len, batch_size))  
memory = torch.rand(seq_len, batch_size, d_model)

out = decoder(tgt, memory)
prob = torch.softmax(out,dim= -1)
print(out.shape)  
print(out)                  