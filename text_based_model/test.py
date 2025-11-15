import argparse
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizerFast, BertForTokenClassification


LABELS = ["FP", "RP", "RV", "RS", "PW"]  # disfluency labels from the paper/model


# ========================= Dataset + Collate ========================= #

class FluencyBankWordDataset(Dataset):
    """
    Dataset for FluencyBank-style disfluency transcripts.

    Expects:
      - metadata file (csv or xlsx) with a segment ID column (default: 'segid')
      - per-segment csv files in `word_dir` named like '{segid}.csv'
        with columns at least: 'word', 'fp', 'rp', 'rv', 'pw'
        (RS is not in the file; we treat it as 0 for all words)

    Labels are per-word multi-label; we project them to BERT subword tokens
    using is_split_into_words=True and the word_ids mapping.
    """

    def __init__(
        self,
        metadata_path: str,
        word_dir: str,
        tokenizer: BertTokenizerFast,
        max_length: int = 256,
        segid_column: str = "segid",
    ):
        super().__init__()
        self.word_dir = word_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.segid_column = segid_column

        # Load metadata (csv or xlsx)
        if metadata_path.endswith(".csv"):
            meta = pd.read_csv(metadata_path)
        else:
            meta = pd.read_excel(metadata_path)

        if segid_column not in meta.columns:
            raise ValueError(f"Metadata must contain column '{segid_column}'")

        self.segments = meta[segid_column].astype(str).tolist()

    def __len__(self):
        return len(self.segments)

    def _load_segment_words_and_labels(self, segid: str):
        """
        Load the word-level disfluency information for a segment.

        Expected CSV: {segid}.csv with columns at least:
            word, fp, rp, rv, pw
        """
        csv_path = os.path.join(self.word_dir, f"{segid}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing word-level file: {csv_path}")

        df = pd.read_csv(csv_path)

        required = ["word", "fp", "rp", "rv", "pw"]
        for col in required:
            if col not in df.columns:
                raise ValueError(
                    f"{csv_path} must contain column '{col}'. "
                    f"Found columns: {list(df.columns)}"
                )

        words = df["word"].astype(str).str.lower().tolist()

        fp = df["fp"].fillna(0).astype(int).tolist()
        rp = df["rp"].fillna(0).astype(int).tolist()
        rv = df["rv"].fillna(0).astype(int).tolist()
        pw = df["pw"].fillna(0).astype(int).tolist()
        rs = [0] * len(df)  # RS not provided → assume zero

        # [num_words, 5], order: FP, RP, RV, RS, PW
        labels = np.stack([fp, rp, rv, rs, pw], axis=-1).astype(np.float32)
        return words, labels

    def __getitem__(self, idx):
        segid = self.segments[idx]
        words, word_labels = self._load_segment_words_and_labels(segid)

        # Tokenize at word level so we can map back from tokens to words
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0]            # [T]
        attention_mask = encoding["attention_mask"][0]  # [T]
        word_ids = encoding.word_ids(batch_index=0)     # length T

        num_tokens = len(word_ids)
        num_labels = word_labels.shape[1]
        token_labels = np.zeros((num_tokens, num_labels), dtype=np.float32)
        label_mask = np.zeros(num_tokens, dtype=bool)   # True for real tokens (not [CLS]/[SEP]/pad)

        for t_idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            if w_id >= len(word_labels):
                continue
            token_labels[t_idx] = word_labels[w_id]
            label_mask[t_idx] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(token_labels, dtype=torch.float32),
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),
            "segid": segid,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    label_mask = torch.stack([b["label_mask"] for b in batch], dim=0)
    segids = [b["segid"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_mask": label_mask,
        "segids": segids,
    }


# ========================= Metrics ========================= #

def compute_metrics(y_true: np.ndarray,
                    y_logits: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, Any]:
    """
    y_true:   [N, 5]  (FP, RP, RV, RS, PW)  0/1 ground truth
    y_logits: [N, 5]  raw logits from model

    We compute per-label precision, recall, F1 for:
      - 5 disfluency labels
      - non-disfluent ("ND") where all 5 ground-truth labels are 0
    """
    if y_true.size == 0:
        print("Warning: no tokens to evaluate (y_true is empty).")
        return {}

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= threshold).astype(int)

    num_labels = y_true.shape[1]  # should be 5
    results = {}

    def prf1(t, p):
        t = t.astype(int)
        p = p.astype(int)
        tp = np.logical_and(t == 1, p == 1).sum()
        fn = np.logical_and(t == 1, p == 0).sum()
        fp = np.logical_and(t == 0, p == 1).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support_pos = int((t == 1).sum())
        return precision, recall, f1, support_pos

    # 1) per disfluency label
    for i in range(num_labels):
        t = y_true[:, i]
        p = y_pred[:, i]
        precision, recall, f1, support_pos = prf1(t, p)
        results[LABELS[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_pos": support_pos,
        }

    # 2) non-disfluent (all 5 labels = 0)
    true_nd = (y_true.sum(axis=1) == 0).astype(int)
    pred_nd = (y_pred.sum(axis=1) == 0).astype(int)
    precision_nd, recall_nd, f1_nd, support_nd = prf1(true_nd, pred_nd)
    results["ND"] = {
        "precision": precision_nd,
        "recall": recall_nd,
        "f1": f1_nd,
        "support_pos": support_nd,
    }

    # macro averages over the 6 classes
    recalls = [results[k]["recall"] for k in LABELS + ["ND"]]
    f1s = [results[k]["f1"] for k in LABELS + ["ND"]]
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    results["macro"] = {
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    return results


# ========================= Evaluation Loop ========================= #

def evaluate(
    metadata_path: str,
    word_dir: str,
    weights_path: str,
    batch_size: int = 32,
    max_length: int = 256,
    threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    segid_column: str = "segid",
):
    # Tokenizer & model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABELS),
    )

    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    # Some checkpoints include a static position_ids that conflict with HF models
    if "bert.embeddings.position_ids" in state_dict:
        del state_dict["bert.embeddings.position_ids"]
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # Dataset
    dataset = FluencyBankWordDataset(
        metadata_path=metadata_path,
        word_dir=word_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        segid_column=segid_column,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    criterion = nn.BCEWithLogitsLoss()

    all_true = []
    all_logits = []
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)          # [B, T, 5]
            label_mask = batch["label_mask"].to(device)  # [B, T]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [B, T, 5]

            # ---- FIXED PART: flatten correctly and mask tokens, not 1D elements ----
            B, T, L = logits.shape  # L should be 5
            logits_2d = logits.view(B * T, L)    # [B*T, 5]
            labels_2d = labels.view(B * T, L)    # [B*T, 5]
            mask_flat = label_mask.view(B * T).bool()  # [B*T]

            logits_flat = logits_2d[mask_flat]  # [N_real, 5]
            labels_flat = labels_2d[mask_flat]  # [N_real, 5]
            # --------------------------------------------------------

            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            total_steps += 1

            all_true.append(labels_flat.cpu().numpy())
            all_logits.append(logits_flat.cpu().numpy())

    avg_loss = total_loss / max(1, total_steps)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0, len(LABELS)))
    y_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, len(LABELS)))

    metrics = compute_metrics(y_true, y_logits, threshold=threshold)

    print("\n=== Evaluation Results ===")
    print(f"Average BCEWithLogits loss: {avg_loss:.4f}\n")

    if not metrics:
        print("No metrics to display (empty y_true).")
        return metrics

    for lbl in LABELS + ["ND"]:
        m = metrics[lbl]
        print(
            f"{lbl:>2}: "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, "
            f"f1={m['f1']:.4f}, "
            f"support_pos={m['support_pos']}"
        )

    print("\nMacro averages (5 disfluencies + ND):")
    print(f"  macro recall = {metrics['macro']['macro_recall']:.4f}")
    print(f"  macro F1     = {metrics['macro']['macro_f1']:.4f}")

    return metrics


# ========================= CLI ========================= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BERT disfluency model on FluencyBank.")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to test metadata (e.g. splits/test_metadata.csv).")
    parser.add_argument("--word_dir", type=str, required=True,
                        help="Directory containing per-segment word CSVs (e.g. 502_001.csv, fa_025.csv, ...).")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to model weights (e.g. Switchboard language.pt or fine-tuned weights).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for classifying a label as present.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--segid_column", type=str, default="segid",
                        help="Column in metadata that contains segment IDs.")
    args = parser.parse_args()

    evaluate(
        metadata_path=args.metadata_path,
        word_dir=args.word_dir,
        weights_path=args.weights_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        threshold=args.threshold,
        device=args.device,
        segid_column=args.segid_column,
    )
