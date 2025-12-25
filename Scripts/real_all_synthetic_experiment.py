import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import random

# =========================
# Config
# =========================

BASE_DIR = Path("/Users/aysasorahi/Documents/master/SLAM LAB/REZA/code_scripts")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results_real_all_synthetic_3class"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

SEEDS = [1, 2, 3, 4, 5]
N_FOLDS = 2

LABELS = ["INTJ", "DM", "DIR"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Dataset
# =========================

class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["sample"].tolist()
        self.labels = df["function"].map(label2id).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

# =========================
# Utilities
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# Load data
# =========================

dfs_real = []
dfs_synth = []

for fname in [
    "INTJ_hdbscan_clusters.csv",
    "DM_hdbscan_clusters.csv",
    "DIR_hdbscan_clusters.csv",
]:
    df = pd.read_csv(ARTIFACTS_DIR / fname)
    dfs_real.append(df[df["Label"].isna()])
    dfs_synth.append(df[df["Label"] == "L"])

real_df = pd.concat(dfs_real, ignore_index=True)
synth_df = pd.concat(dfs_synth, ignore_index=True)

real_df = real_df[real_df["function"].isin(LABELS)]
synth_df = synth_df[synth_df["function"].isin(LABELS)]

print("\nReal data distribution:")
print(real_df["function"].value_counts())

print("\nSynthetic data distribution:")
print(synth_df["function"].value_counts())

# =========================
# Experiment
# =========================

all_results = []
conf_matrix_total = np.zeros((len(LABELS), len(LABELS)), dtype=int)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X = real_df["sample"].values
y = real_df["function"].values

for seed in SEEDS:
    set_seed(seed)

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

        train_real = real_df.iloc[train_idx]
        test_df = real_df.iloc[test_idx]

        train_df = pd.concat([train_real, synth_df], ignore_index=True)

        train_ds = TextDataset(train_df, tokenizer)
        test_ds = TextDataset(test_df, tokenizer)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        loss_fn = torch.nn.CrossEntropyLoss()

        # ---- Training ----
        model.train()
        for _ in range(EPOCHS):
            for batch in train_loader:
                optimizer.zero_grad()
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch["labels"])
                loss.backward()
                optimizer.step()

        # ---- Evaluation ----
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=1)
                y_true.extend(batch["labels"].cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        macro_f1 = f1_score(y_true, y_pred, average="macro")
        per_class_f1 = f1_score(
            y_true, y_pred, average=None, labels=list(range(len(LABELS)))
        )

        conf_matrix_total += confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABELS)))
        )

        result = {
            "seed": seed,
            "fold": fold,
            "macro_f1": macro_f1
        }

        for i, lbl in id2label.items():
            result[f"f1_{lbl}"] = per_class_f1[i]

        all_results.append(result)

        print(f"Seed {seed} | Fold {fold} | Macro F1 = {macro_f1:.3f}")

# =========================
# Save results
# =========================

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / "results_real_all_synthetic_runs.csv", index=False)

summary_df = results_df.describe().loc[["mean", "std"]]
summary_df.to_csv(RESULTS_DIR / "results_real_all_synthetic_summary.csv")

conf_df = pd.DataFrame(conf_matrix_total, index=LABELS, columns=LABELS)
conf_df.to_csv(RESULTS_DIR / "confusion_real_all_synthetic.csv")

print("\n Real + All Synthetic 3-class experiment completed and saved.")
