import torch
from transformers import BertTokenizer, BertModel
from Bio import SeqIO
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


print("Loading sequences...")

human_sequences = {record.id: str(record.seq) for record in SeqIO.parse("/data/tianlab/Eric/AINUC/human.fasta", "fasta")}
positive_sequences = {record.id: str(record.seq) for record in SeqIO.parse("/data/tianlab/Eric/AINUC/EXP.fasta", "fasta")}
positive_ids = set(positive_sequences.keys())

negative_ids = [pid for pid in human_sequences if pid not in positive_ids]
random.seed(42)
sampled_negative_ids = random.sample(negative_ids, len(positive_ids))

train_sequences = []
labels = []

for pid in positive_ids:
    train_sequences.append(positive_sequences[pid])
    labels.append(1)

for pid in sampled_negative_ids:
    train_sequences.append(human_sequences[pid])
    labels.append(0)

print(f"positive: {len(positive_ids)}, negative: {len(sampled_negative_ids)}")


print("Loading ProtBERT model...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Using device: {device}")

def preprocess_sequence(seq):
    seq = seq.replace('U','X').replace('Z','X').replace('O','X')
    return ' '.join(list(seq))

@torch.no_grad()
def get_protbert_embedding(seq):
    seq = preprocess_sequence(seq)
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:,0,:]
    return embedding.cpu().squeeze().numpy()

print("Extracting ProtBERT embeddings...")
X_emb = []
for seq in tqdm(train_sequences):
    X_emb.append(get_protbert_embedding(seq))
X_emb = np.array(X_emb)

def extract_motifs(seqs, min_len=4, max_len=8, top_k=50):
     motif_counts = Counter()
    for seq in seqs:
        for L in range(min_len, max_len+1):
            for i in range(len(seq)-L+1):
                motif = seq[i:i+L]
                motif_counts[motif] +=1
    top_motifs = [m for m,_ in motif_counts.most_common(top_k)]
    return top_motifs

positive_seqs = [seq for seq,label in zip(train_sequences, labels) if label==1]
motifs = extract_motifs(positive_seqs)
print(f"Top {len(motifs)} motifs extracted.")

def motif_features(seq, motifs):
    return [1 if m in seq else 0 for m in motifs]

X_motif = [motif_features(seq, motifs) for seq in train_sequences]
X_motif = np.array(X_motif)


X = np.hstack([X_emb, X_motif])
y = np.array(labels)
print(f"Combined feature shape: {X.shape}")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ProteinDataset(X_train_scaled, y_train)
val_dataset = ProteinDataset(X_val_scaled, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128,1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

input_dim = X_train_scaled.shape[1]
model_mlp = MLPClassifier(input_dim).to(device)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)
criterion = nn.BCELoss()


epochs = 20
for epoch in range(epochs):
    model_mlp.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        pred = model_mlp(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*len(yb)
    train_loss /= len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")


model_mlp.eval()
y_val_prob = []
with torch.no_grad():
    for xb, _ in val_loader:
        xb = xb.to(device)
        y_val_prob.extend(model_mlp(xb).cpu().numpy().flatten())
y_val_prob = np.array(y_val_prob)

precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
f1_scores = 2*precision*recall/(precision+recall)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print("Best threshold based on F1:", best_threshold)

plt.figure()
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(best_threshold, color='red', linestyle='--', label="Best Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("precision_recall_curve.png", dpi=300)
plt.close()

print("Predicting all human proteins...")

all_ids = list(human_sequences.keys())
all_seqs = list(human_sequences.values())

X_all_emb = []
for seq in tqdm(all_seqs):
    X_all_emb.append(get_protbert_embedding(seq))
X_all_emb = np.array(X_all_emb)

X_all_motif = np.array([motif_features(seq, motifs) for seq in all_seqs])
X_all_combined = np.hstack([X_all_emb, X_all_motif])
X_all_scaled = scaler.transform(X_all_combined)

all_dataset = ProteinDataset(X_all_scaled, np.zeros(len(X_all_scaled)))
all_loader = DataLoader(all_dataset, batch_size=32)

model_mlp.eval()
probs = []
with torch.no_grad():
    for xb, _ in all_loader:
        xb = xb.to(device)
        probs.extend(model_mlp(xb).cpu().numpy().flatten())
probs = np.array(probs)
preds = (probs >= best_threshold).astype(int)

df = pd.DataFrame({
    "Protein_ID": all_ids,
    "Predicted_Nucleolus": ["Yes" if p==1 else "No" for p in preds],
    "Probability": probs
})
df.to_csv("/data/tianlab/Eric/AINUC/nucleolus_predictions_mlp_motif4_8.csv", index=False)
print("✅ 预测完成，结果保存为 nucleolus_predictions_mlp_motif4_8.csv")




print("Predicting all human proteins (excluding known positives)...")

all_ids = [pid for pid in human_sequences.keys() if pid not in positive_ids]
all_seqs = [human_sequences[pid] for pid in all_ids]

X_all_emb = []
for seq in tqdm(all_seqs):
    X_all_emb.append(get_protbert_embedding(seq))
X_all_emb = np.array(X_all_emb)

X_all_motif = np.array([motif_features(seq, motifs) for seq in all_seqs])
X_all_combined = np.hstack([X_all_emb, X_all_motif])
X_all_scaled = scaler.transform(X_all_combined)

all_dataset = ProteinDataset(X_all_scaled, np.zeros(len(X_all_scaled)))
all_loader = DataLoader(all_dataset, batch_size=32)

model_mlp.eval()
probs = []
with torch.no_grad():
    for xb, _ in all_loader:
        xb = xb.to(device)
        probs.extend(model_mlp(xb).cpu().numpy().flatten())
probs = np.array(probs)
preds = (probs >= best_threshold).astype(int)

df = pd.DataFrame({
    "Protein_ID": all_ids,
    "Predicted_Nucleolus": ["Yes" if p==1 else "No" for p in preds],
    "Probability": probs
})
df.to_csv("/data/tianlab/Eric/AINUC/nucleolus_predictions_mlp_motif4_8_no_positive.csv", index=False)


motif_feature_cols = [f"motif_{mot}" for mot in motifs]

motif_scores = X_all_motif.sum(axis=1)

combine_scores = probs * motif_scores

df = pd.DataFrame({
    "Protein_ID": all_ids,
    "Predicted_Nucleolus": ["Yes" if p == 1 else "No" for p in preds],
    "Probability": probs,
    "Motif_Score": motif_scores,
    "Combined_Score": combine_scores
})

# Append motif feature columns
motif_df = pd.DataFrame(X_all_motif, columns=motif_feature_cols)
df = pd.concat([df, motif_df], axis=1)

df.to_csv("nucleolus_predictions_with_motif_and_scores.csv", index=False)
print(" Saved: nucleolus_predictions_with_motif_and_scores.csv")



# Calculate motif enrichment scores from training data
pos_seqs = [positive_sequences[pid] for pid in positive_ids]
neg_seqs = [human_sequences[pid] for pid in negative_ids]

def motif_count_dict(seqs, motif_list):
    counts = {mot: 0 for mot in motif_list}
    for seq in seqs:
        feats = motif_features(seq, motif_list)
        for i, mot in enumerate(motif_list):
            counts[mot] += feats[i]
    return counts

pos_counts = motif_count_dict(pos_seqs, motifs)
neg_counts = motif_count_dict(neg_seqs, motifs)

N_pos = len(pos_seqs)
N_neg = len(neg_seqs)

motif_scores = {}
for mot in motifs:
    motif_scores[mot] = np.log2(((pos_counts[mot] + 1) / N_pos) /
                                ((neg_counts[mot] + 1) / N_neg))

# Compute protein motif score using data-driven motif ranking
motif_scores_list = np.array([
    sum(X_all_motif[i][j] * motif_scores[mot] for j, mot in enumerate(motifs))
    for i in range(len(X_all_motif))
])

# Normalize to [0,1] for nice plotting
motif_scores_norm = (motif_scores_list - motif_scores_list.min()) / \
                    (motif_scores_list.max() - motif_scores_list.min())

# Combine: model probability * motif confidence
combined_scores = probs * motif_scores_norm

# Update output DataFrame
df["Motif_Score"] = motif_scores_norm
df["Combined_Score"] = combined_scores
df = df.sort_values("Combined_Score", ascending=False)
df.to_csv("/data/tianlab/Eric/AINUC/nucleolus_predictions_enriched_motif.csv", index=False)
print(" Saved: nucleolus_predictions_enriched_motif.csv")



import requests
import time

def extract_uniprot_id(protein_id):
    """从 fasta ID 中提取 UniProt accession"""
    if "|" in protein_id:
        parts = protein_id.split("|")
        for p in parts:
            if len(p) >= 5 and p[0].isalpha():
                return p
    return protein_id.split("_")[0]

def query_uniprot_gene(acc):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": acc,
        "fields": "accession,gene_primary",
        "format": "tsv"
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        lines = resp.text.strip().split("\n")
        if len(lines) > 1:
            parts = lines[1].split("\t")
            if len(parts) > 1 and parts[1]:
                return parts[1]
    return None

print("🔍 Mapping protein IDs to gene names via UniProt API...")
df = pd.read_csv("/data/tianlab/Eric/AINUC/nucleolus_predictions_enriched_motif.csv")

gene_names = []
for pid in tqdm(df["Protein_ID"], desc="Querying UniProt"):
    acc = extract_uniprot_id(pid)
    gene = query_uniprot_gene(acc)
    gene_names.append(gene if gene else "Unknown")
    time.sleep(0.2) 

df["Gene_Name"] = gene_names
df.to_csv("/data/tianlab/Eric/AINUC/nucleolus_predictions_enriched_motif_gene.csv", index=False)


