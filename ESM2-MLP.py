import os, random, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt


def read_fasta(path):
    data = {}
    for rec in SeqIO.parse(path, "fasta"):
        data[rec.id] = str(rec.seq).upper().replace("U","X")
    return data

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def esm2_embed(pairs, tokenizer, model, device="cuda", batch_size=2, max_len=1024):
    model.eval()
    out = {}
    with torch.no_grad():
        for batch in tqdm(list(chunk(pairs, batch_size)), desc="Embedding ESM-2"):
            ids = [x[0] for x in batch]
            seqs = [x[1][:max_len] for x in batch]
            toks = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            toks = {k:v.to(device) for k,v in toks.items()}
            hs = model(**toks).last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)
            mean = (hs*mask).sum(1) / mask.sum(1)
            for pid, vec in zip(ids, mean):
                out[pid] = vec.cpu().numpy().astype(np.float32)
    return out

def choose_threshold(y_true, scores):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2*p*r/(p+r+1e-9)
    i = np.argmax(f1)
    thr = t[max(0, i-1)] if len(t) else 0.5
    ap = average_precision_score(y_true, scores)
    return float(thr), float(f1[i]), float(ap)

human_fa = "human.fasta"
pos_fa = "nucleolus.fasta"

random.seed(42); np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

human = read_fasta(human_fa)
pos = read_fasta(pos_fa)
pos_ids = set(pos.keys())
neg = {k:v for k,v in human.items() if k not in pos_ids}

print(f"Positive={len(pos)}, Negative pool={len(neg)}")

ratio = 1.0
neg_sampled = dict(random.sample(list(neg.items()), int(len(pos)*ratio)))
print("Negatives used:", len(neg_sampled))

"facebook/esm2_t6_8M_UR50D"

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

train_pairs = list(pos.items()) + list(neg_sampled.items())
y = np.array([1]*len(pos) + [0]*len(neg_sampled), dtype=np.int32)

embs = esm2_embed(train_pairs, tokenizer, model, device=device, batch_size=1, max_len=1024)
X = np.vstack([embs[k] for k,_ in train_pairs])

Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xval_s = scaler.transform(Xval)

import torch.nn as nn
import torch.optim as optim

Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32).to(device)
Xval_t = torch.tensor(Xval_s, dtype=torch.float32).to(device)
ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1).to(device)
yval_t = torch.tensor(yval, dtype=torch.float32).unsqueeze(1).to(device)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP(Xtr_s.shape[1]).to(device)
optimizer = optim.Adam(mlp.parameters(), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(1, 26):
    mlp.train()
    optimizer.zero_grad()
    loss = criterion(mlp(Xtr_t), ytr_t)
    loss.backward()
    optimizer.step()

    mlp.eval()
    with torch.no_grad():
        val_loss = criterion(mlp(Xval_t), yval_t)

    if epoch % 3 == 0:
        print(f"Epoch {epoch}/25 loss={loss.item():.4f} val={val_loss.item():.4f}")

mlp.eval()
with torch.no_grad():
    val_score = mlp(Xval_t).cpu().numpy().flatten()

thr, f1, ap = choose_threshold(yval, val_score)
print(f"[Validation] F1={f1:.3f} AP={ap:.3f} Thr={thr:.3f}")

pairs_all = list(human.items())
emb_all = esm2_embed(pairs_all, tokenizer, model, device=device, batch_size=1)

Xh = np.vstack([emb_all[k] for k,_ in pairs_all])
Xh_s = scaler.transform(Xh)
with torch.no_grad():
    scores = mlp(torch.tensor(Xh_s, dtype=torch.float32).to(device)).cpu().numpy().flatten()

pred = (scores >= thr).astype(int)

out = pd.DataFrame({
    "protein_id":[k for k,_ in pairs_all],
    "score":scores,
    "pred_label":pred,
    "known":[1 if k in pos_ids else 0 for k,_ in pairs_all],
})

out.to_csv("ESM2_MLP_predictions.csv", index=False)
print("Saved: ESM2_MLP_predictions.csv")
display(out.head())

unknown = out[(out.pred_label==1)&(out.known==0)].copy()
unknown.to_csv("ESM2_MLP_new_candidates.csv", index=False)
print(f"New candidates: {len(unknown)} saved to ESM2_MLP_new_candidates.csv")
display(unknown.head())


def get_gene(pid):
    m = re.search(r'\|[^|]+\|([A-Za-z0-9\-]+)_HUMAN', pid)
    if m: return m.group(1)
    m = re.search(r'GN=([A-Za-z0-9\-]+)', pid)
    if m: return m.group(1)
    return pid

unknown["gene"] = unknown["protein_id"].apply(get_gene)
unknown[["gene","score"]].to_csv("ESM2_MLP_new_genes.csv", index=False)
print(f"Genes saved → ESM2_MLP_new_genes.csv")
display(unknown.head())

p, r, _ = precision_recall_curve(yval, val_score)
plt.figure(figsize=(4.5,4))
plt.plot(r, p)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("PR Curve - ESM2-MLP")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()

