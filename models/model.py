import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml  
from typing import Union


with open("models/config/model.yaml") as file:
    config = yaml.safe_load(file)


class InputLayer(nn.Module):
    def __init__(self, v_size: int, emb_dim: int, length: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.emb = nn.Linear(v_size, self.emb_dim, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, length, emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n, V) -> (B, N, D)  | N = n + 1
        z = self.emb(x.type(torch.float))
        z += self.pos_emb
        z = torch.cat([z, self.cls_token.repeat(repeats=(x.size(0),1,1))], dim=1)
        return z


class Attention(nn.Module):
    def __init__(self, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.sqrt_dh = emb_dim ** 0.5
        self.attn_drop = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # (, N, D) -> (, D, N)
        k_T = k.transpose(-1, -2)
        # (, N, D) x (, D, N) -> (, N, N)
        dots = (q @ k_T) / self.sqrt_dh

        attn = F.softmax(dots, dim=-1)
        # (, N, N) x (, N, D) -> (, N, D)
        attn = self.attn_drop(attn)

        out = attn @ v
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim:int, head:int, dropout:float) -> None:
        super().__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)  

        self.attn = Attention(self.head_dim, dropout)

        self.attn_drop = nn.Dropout(dropout)
        self.w_o = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_patch, _ = z.size()
        # (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # (B, h, N, D//h) -> (B, h, N, D//h)
        out = self.attn(q, k ,v)

        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        # (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # (B, N, D) -> (B, N, D) 
        out = self.w_o(out) 
        return out


class EncoderBlock(nn.Module):
    def __init__(self, emb_dim:int, head:int, hidden_dim:int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.msa = MultiHeadSelfAttention(emb_dim, head, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.feedforward = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.msa(self.ln1(z)) + z
        out = self.feedforward(self.ln2(out)) + out 
        return out
    

class OutputLayer(nn.Module):
    def __init__(self, v_size: int, emb_dim: int) -> None:
        super().__init__()
        self.ln = nn.Linear(emb_dim, v_size, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.ln(z)
        return out


class GPT(nn.Module):
    def __init__(self, v_size, emb_dim, length, num_blocks, head, hidden_dim, dropout) -> None:
        super().__init__()
        self.v_size = v_size
        self.emb_dim = emb_dim
        self.length = length
        self.num_blocks = num_blocks
        self.head = head
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.inp = InputLayer(self.v_size, self.emb_dim, self.length)
        self.blocks = nn.Sequential(*[EncoderBlock(self.emb_dim, self.head, self.hidden_dim, self.dropout)for _ in range(self.num_blocks)])
        self.outp = OutputLayer(self.v_size, self.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inp(x)
        out = self.blocks(out)
        out = out[:, -1]
        pred = self.outp(out)
        return pred


def model():
    c = config["GPT"]
    instance = GPT(c["v_size"], c["emb_dim"], c["length"], c["num_blocks"], c["head"], c["hidden_dim"], c["dropout"])
    return instance
