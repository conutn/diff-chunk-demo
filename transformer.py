import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import time
import math
import random
import unidecode
import time
import json

def read_vocab(fname):
  file = open(fname, encoding = "utf-8").read().splitlines()
  vocab = []
  for word in file:
    if any((ord(ch) > 128) for ch in word): continue
    if word[:2] == "##":
      vocab.append("§" + word[2:])
    else:
      vocab.append(word)
  vocab = list(filter(lambda x: len(x) <= 4 or
    ('§' in x and len(x) == 5), vocab))
  vocab += ["[SEP]", "[PAD]", "[UNK]"]
  return vocab

def get_vocab_dct_encoder(vocab):
  vocab_dct = {vocab[idx] : idx for idx in range(len(vocab))}
  return vocab_dct

whitespace = {'\u0009', '\u000a', '\u000b', '\u000c', '\u000d',
              '\u0020', '\u0085', '\u00a0', '\u1680', '\u2000',
              '\u2001', '\u2002', '\u2003', '\u2004', '\u2005',
              '\u2006', '\u2007', '\u2008', '\u2009', '\u200a',
              '\u2028', '\u2029', '\u202f', '\u205f', '\u3000'}

def is_whitespace(char: str) -> bool:
    """Returns true if char is a whitespace character."""
    return char in whitespace

class Tokenizer():
  # taken from Hugging Face's tokenizer implementation
  """A utility class for tokenizing text."""

  def clean_text(self, text: str, is_wiki = False) -> str:
    """Replaces non-alphanumeric characters with an ASCII
       approximation.
       Can be used with raw Wikipedia article text; removes all
       text after the References section in the article."""

    text = unidecode.unidecode(text)
    text = text.lower()
    output = []

    for sentence in text.splitlines():
      if sentence in {"references", "further reading", "sources", "external links"}: break
      if (is_wiki and len(sentence) and
        sentence[-1] not in self.punc): continue
      for char in sentence:
        if is_whitespace(char):
          output.append(' ')
        else:
          output.append(char)
      output.append(' ')

    return "".join(output)

  def __init__(self, vocab: list, unk_token: str):
    self.vocab = vocab
    self.unk_token = unk_token
    self.punc = {'.', '!', '?'} # for separating sentences

  def tokenize(self, text: str, is_wiki = False) -> str:
    """Converts text into tokens.
       Suffixes are prepended with a section character (§)."""

    set_vocab = set(self.vocab)

    # clean text and strip whitespace
    text = self.clean_text(text, is_wiki = is_wiki)
    text = text.strip()
    tokens = text.split()

    output = []

    for token in tokens:
      chars = list(token)

      is_valid = True
      start = 0
      sub_tokens = []

      # take tokens greedily from the beginning of the word
      while start < len(chars):
        end = len(chars)
        cur_substr = None

        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "§" + substr
          if substr in set_vocab:
            cur_substr = substr
            break
          end -= 1

        if cur_substr == None:
          is_valid = False
          break

        sub_tokens.append(cur_substr)
        start = end

      if not is_valid:
        output.append(self.unk_token)
      else:
        output.extend(sub_tokens)

    return output

def build_mask(seq):
  B, T = seq.shape
  device = seq.device
  # prevent seeing future tokens
  causal = torch.triu(torch.ones(T, T, dtype = torch.bool,
                                 device = device), diagonal = 1)
  # map 1 -> -1e9
  additive = causal.float() * float("-1e9")
  return additive.unsqueeze(0).expand(B, T, T)

class BoundaryPredictor(nn.Module):
  def __init__(self, emb_dim, hidden_dim):
    super().__init__()
    self.lstm = nn.LSTM(emb_dim, hidden_dim, 2, batch_first = True)
    self.linear = nn.Linear(hidden_dim, 1)

  def forward(self, x):
    out = self.lstm(x)[0]
    out = self.linear(out)
    out = torch.sigmoid(out)
    return out.squeeze(-1)

class ChunkedDataset(Dataset):
  def __init__(self, tokens, block_size = 256):
    ids = tokens
    self.ids = torch.tensor(ids, dtype = torch.long)
    self.block_size = block_size

  def __len__(self):
    return len(self.ids) - self.block_size

  def __getitem__(self, idx):
    x = self.ids[idx: idx + self.block_size]
    y = self.ids[idx + 1: idx + 1 + self.block_size]
    return x, y

def collate(batch):
  x, y = zip(*batch)
  return torch.stack(x, dim = 0), torch.stack(y, dim = 0)

def chunk_weights(C, K, beta):
  device = C.device
  # for each chunk, cumulative sum should be approximately k-1
  k = torch.arange(K, device = device).view(1, 1, K)
  C = C.unsqueeze(-1)
  logits = -beta * (C - k) ** 2
  return F.softmax(logits, dim = -1)

def hard_chunk_bounds(chunk_id, K):
  B, T = chunk_id.shape
  device = chunk_id.device

  # one hot vector for each token representing chunk
  one_hot = torch.nn.functional.one_hot(chunk_id, K).bool()

  # for each sequence, does it have tokens in this chunk?
  has = one_hot.any(dim = 1)

  # find starting index (not differentiable)
  start = one_hot.float().argmax(dim = 1)

  # reverse the one hot to find the ends (not differentiable)
  rev = torch.flip(one_hot, dims=[1])
  last = rev.float().argmax(dim = 1)
  end = T - last

  start = torch.where(has, start, torch.full_like(start, T))
  end = torch.where(has, end, torch.full_like(end, T))

  return start, end

def gumbel_hard(weights, tau = 1.0):
  # convert probabilities to log for stability
  logp = torch.log(weights.clamp_min(1e-9))

  # sampling
  gumbel = -torch.log(-torch.log(torch.rand_like(logp)))

  # compute chunk assignments
  y_soft = F.softmax((logp) / tau, dim = -1) # F.softmax((logp + gumbel) / tau, dim = -1)

  index = y_soft.argmax(dim = -1)
  y_hard = F.one_hot(index, y_soft.size(-1)).type_as(y_soft)

  y = y_hard.detach() - y_soft.detach() + y_soft

  return index, y

def extract_chunks(x, start, end, L_max):
    B, T, D = x.shape
    device = x.device

    lengths = (end - start).clamp_min(0)

    pos = torch.arange(L_max, device = device)

    idx = start.unsqueeze(-1) + pos

    mask = pos < lengths.unsqueeze(-1)

    idx = idx.clamp(0, T - 1)

    x_exp = x.unsqueeze(1).expand(B, start.size(1), T, D)
    chunks = torch.gather(
        x_exp,
        dim = 2,
        index = idx.unsqueeze(-1).expand(-1, -1, -1, D)
    )

    chunks = chunks * mask.unsqueeze(-1)

    return chunks, mask

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

class BatchedAttentionLayer(nn.Module):
  def __init__(self, d_model, d_ffn, nhead, dropout = 0.0, bias = True):
    super().__init__()
    assert d_model % nhead == 0

    self.d_model = d_model
    self.nhead = nhead
    self.d_head = d_model // nhead

    self.W_q = nn.Linear(d_model, d_model, bias=bias)
    self.W_k = nn.Linear(d_model, d_model, bias=bias)
    self.W_v = nn.Linear(d_model, d_model, bias=bias)

    self.W_a = nn.Linear(d_model, d_ffn, bias=bias)
    self.W_b = nn.Linear(d_ffn, d_model, bias=bias)

    self.dropout = dropout
    self.relu = nn.ReLU()

    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)

  def forward(self, x, mask=None):
    B, T, D = x.shape
    H = self.nhead
    dH = self.d_head

    residual_input_attn = x
    norm_x_attn = self.ln1(x)

    q = self.W_q(norm_x_attn)
    k = self.W_k(norm_x_attn)
    v = self.W_v(norm_x_attn)

    q = q.view(B, T, H, dH).transpose(1, 2)
    k = k.view(B, T, H, dH).transpose(1, 2)
    v = v.view(B, T, H, dH).transpose(1, 2)

    if mask is not None:
      attn_mask = mask.unsqueeze(1)
    else:
      attn_mask = None

    attn_output = F.scaled_dot_product_attention(
      q, k, v,
      attn_mask = attn_mask,
      dropout_p = self.dropout if self.training else 0.0,
      is_causal = False
    )

    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

    x = residual_input_attn + attn_output

    residual_input_ffn = x
    norm_x_ffn = self.ln2(x)

    ffn_output = self.W_a(norm_x_ffn)
    ffn_output = self.relu(ffn_output)
    ffn_output = F.dropout(ffn_output, p = self.dropout, training = self.training)
    ffn_output = self.W_b(ffn_output)
    ffn_output = F.dropout(ffn_output, p = self.dropout, training = self.training)

    x = residual_input_ffn + ffn_output

    return x

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout = 0.1, max_len = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p = dropout)
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)

    self.register_buffer("pe", pe)

  def forward(self, x, inplace = True):
    T = x.size(1)
    x = x + self.pe[:T].unsqueeze(0)
    if inplace:
      return self.dropout(x)
    return F.dropout(x, inplace = True)

class HierarchicalTransformer(nn.Module):
  def __init__(self, d_model, d_ffn, nhead, token_layers, chunk_layers,
               vocab_size = 16000, max_chunks = 8, beta = 5.0,
               max_len = 256, dropout = 0.0):
    super().__init__()

    self.d_model = d_model
    self.vocab_size = vocab_size
    self.max_chunks = max_chunks
    self.beta = beta

    self.embedding = nn.Embedding(vocab_size, d_model)
    self.positional = PositionalEncoding(d_model, max_len = max_len)
    self.max_len = max_len

    self.token_layers = nn.ModuleList(
        [BatchedAttentionLayer(d_model, d_ffn, nhead, dropout)
        for i in range(token_layers)]
    )

    self.boundary = BoundaryPredictor(d_model, 64)

    self.chunk_layers = nn.ModuleList(
        [BatchedAttentionLayer(d_model, d_ffn, nhead, dropout)
        for i in range(chunk_layers)]
    )

    self.unembedding = nn.Linear(d_model, vocab_size, bias = False)
    self.unembedding.weight = self.embedding.weight

    self.start = nn.Parameter(torch.randn(self.d_model))

  def forward(self, tokens, mask = None, tau = 0.7):
    device = tokens.device
    B, T = tokens.shape
    K = min(self.max_chunks, T)

    # token-level transformer
    x = self.embedding(tokens)
    x = self.positional(x)

    for layer in self.token_layers:
      x = layer(x, mask = mask) # (B, T, D)

    # boundary prediction
    B_prob = self.boundary(x).clamp(0.01, 0.99) # (B, T)
    B_prob = torch.ones_like(B_prob)
    C = torch.cumsum(B_prob, dim = 1)

    # normalize C into [0, K)
    expected_final = B_prob.mean(dim = -1, keepdim = True) * T
    C = C / (expected_final.detach() + 1e-6) * (K - 1)

    # soft weights
    weights = chunk_weights(C, K, self.beta) # (B, T, K)

    # gumbel-softmax
    if self.training:
      chunk_id, hard_assign = gumbel_hard(weights, tau = tau)
    else:
      chunk_id = weights.argmax(dim = -1)
      hard_assign = torch.nn.functional.one_hot(chunk_id, K).float()

    # hard chunk boundaries
    start, end = hard_chunk_bounds(chunk_id, K) # (B, K)

    # extract reduced-length chunks
    L_max = math.ceil(T / K) * 2
    chunks, chunk_mask = extract_chunks(x, start, end, L_max)
    # (B, K, L_max, D), (B, K, L_max)

    chunk_repr = chunks.sum(dim = 2) / chunk_mask.sum(dim = 2).clamp(min = 1).unsqueeze(-1)
    # (B, K, D)
    chunk_repr = chunk_repr.unsqueeze(2).expand(B, K, K, -1).transpose(1, 2)
    # (B, K, K, D)

    chunk_repr_mask = torch.tril(torch.ones(K, K, device = device), diagonal = -1)
    # (K, K)

    chunk_repr = chunk_repr * chunk_repr_mask.view(1, K, K, 1).clone()
    chunk_repr[:, :, -1, :] = self.start

    chunks = torch.cat((chunk_repr, chunks), 2)
    # (B, K, K+L, D)

    B, K, L, D = chunks.shape
    chunks = chunks.view(B * K, L, D)

    # chunk-level transformer
    causal = torch.zeros(L, L, device = device)
    partial_mask = torch.triu(torch.ones(L, L, device = device), diagonal = 1).bool()
    causal.masked_fill_(partial_mask, -1e9)
    causal.unsqueeze_(0)

    chunks = self.positional(chunks, inplace = True)

    for layer in self.chunk_layers:
      chunks = layer(chunks, mask = causal)

    chunks = chunks.view(B, K, L, D)
    chunks = chunks[:, :, K:, :]

    # map chunk outputs to vocab

    B, K, L, D = chunks.shape
    V = self.vocab_size

    # unembed
    logits = self.unembedding(chunks) # (B, K, L, V)

    # absolute token positions
    positions = (start.unsqueeze(-1) +
                 torch.arange(L, device = device).view(1, 1, L))

    # valid mask
    valid = (positions < end.unsqueeze(-1)) & (positions < T) & chunk_mask

    positions = positions.view(B, K, L).clamp(0, T - 1)
    vm = valid

    idx = positions.unsqueeze(-1).expand(B, K, L, V).reshape(B, K * L, V)
    logits = logits.view(B, K, L, V).masked_fill(~vm.unsqueeze(-1), 0).reshape(B, K * L, V)
    counts = vm.float().reshape(B, K * L)

    out = logits.new_zeros(B, T, V, device = device)
    coverage = counts.new_zeros(B, T, device = device)

    out.scatter_add_(1, idx, logits)
    coverage.scatter_add_(1, positions.reshape(B, K * L), counts)

    out = out / coverage.clamp_min(1).unsqueeze(-1)

    return out

def load_model(fname: str, param_dct: dict, device: torch.device) -> torch.nn.Module:
  if param_dct["model_type"] == "HierarchicalTransformer":
    model = HierarchicalTransformer(
        param_dct["emb_dim"],
        param_dct["hidden_dim"],
        param_dct["nhead"],
        param_dct["token_layers"],
        param_dct["chunk_layers"],
        param_dct["vocab_size"],
        param_dct["max_chunks"],
        param_dct["beta"],
        param_dct["max_len"],
        param_dct["dropout"]).to(device)
    ckpt = torch.load(fname, weights_only = False, map_location = device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
  return model