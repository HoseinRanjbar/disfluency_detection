import argparse
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW


LABELS = ["FP", "RP", "RV", "RS", "PW"]  # order must match your model


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
        Load the word-level disfluency information for a a segment.

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

def compute_metrics(
    y_true: np.ndarray,
    y_logits: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    y_true:   [N, 5]  (FP, RP, RV, RS, PW)  0/1 ground truth
    y_logits: [N, 5]  raw logits from model

    We compute per-label precision, recall, F1,
    and UAR = mean recall over the 5 disfluency classes.
    """
    if y_true.size == 0:
        print("Warning: no tokens to evaluate (y_true is empty).")
        return {}

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= threshold).astype(int)

    num_labels = y_true.shape[1]  # should be 5
    results: Dict[str, Any] = {}

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

    recalls = []
    f1s = []

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
        recalls.append(recall)
        f1s.append(f1)

    # UAR and macro F1 over the 5 disfluency labels
    results["UAR"] = float(np.mean(recalls))
    results["macro_f1"] = float(np.mean(f1s))

    return results


# ========================= Evaluation ========================= #

def evaluate(
    model: BertForTokenClassification,
    dataloader: DataLoader,
    criterion,
    device: str,
    threshold: float = 0.5,
    desc: str = "val",
) -> Tuple[float, Dict[str, Any]]:
    model.eval()

    all_true = []
    all_logits = []
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)          # [B, T, 5]
            label_mask = batch["label_mask"].to(device)  # [B, T]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [B, T, 5]

            # Flatten and mask only real tokens (not [CLS]/[SEP]/pad)
            B, T, L = logits.shape
            logits_2d = logits.view(B * T, L)    # [B*T, 5]
            labels_2d = labels.view(B * T, L)    # [B*T, 5]
            mask_flat = label_mask.view(B * T).bool()  # [B*T]

            logits_flat = logits_2d[mask_flat]  # [N_real, 5]
            labels_flat = labels_2d[mask_flat]  # [N_real, 5]

            if logits_flat.numel() == 0:
                continue

            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            total_steps += 1

            all_true.append(labels_flat.cpu().numpy())
            all_logits.append(logits_flat.cpu().numpy())

    avg_loss = total_loss / max(1, total_steps)
    y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0, len(LABELS)))
    y_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, len(LABELS)))

    metrics = compute_metrics(y_true, y_logits, threshold=threshold)

    return avg_loss, metrics


# ========================= Training (step-based eval) ========================= #

def train(
    train_metadata_path: str,
    val_metadata_path: str,
    word_dir: str,
    output_weights: str,
    init_weights: str = None,
    batch_size: int = 32,
    lr: float = 5e-5,
    epochs: int = 20,
    max_length: int = 256,
    threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    segid_column: str = "segid",
    eval_every: int = 100,
    patience_evals: int = 10,
):
    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Datasets
    train_dataset = FluencyBankWordDataset(
        metadata_path=train_metadata_path,
        word_dir=word_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        segid_column=segid_column,
    )

    val_dataset = FluencyBankWordDataset(
        metadata_path=val_metadata_path,
        word_dir=word_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        segid_column=segid_column,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    steps_per_epoch = len(train_loader)
    if eval_every > steps_per_epoch:
        print(f"eval_every ({eval_every}) > steps_per_epoch ({steps_per_epoch}), "
              f"using eval_every = {steps_per_epoch} instead.")
        eval_every = steps_per_epoch

    print(f"Training examples: {len(train_dataset)} "
          f"(steps/epoch = {steps_per_epoch}, batch_size = {batch_size})")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Eval every {eval_every} steps, patience {patience_evals} evals.\n")

    # Model
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABELS),
    )

    # Initialize from Switchboard / paper model if provided
    print(f"Loading weights from: {init_weights}")
    state_dict = torch.load(init_weights, map_location="cpu")
    # Some checkpoints include a static position_ids that conflict with HF models
    if "bert.embeddings.position_ids" in state_dict:
        del state_dict["bert.embeddings.position_ids"]
    model.load_state_dict(state_dict, strict=False)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # multi-label

    best_val_uar = -1.0
    best_step = -1
    evals_no_improve = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps_this_epoch = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)          # [B, T, 5]
            label_mask = batch["label_mask"].to(device)  # [B, T]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [B, T, 5]

            B, T, L = logits.shape
            logits_2d = logits.view(B * T, L)
            labels_2d = labels.view(B * T, L)
            mask_flat = label_mask.view(B * T).bool()

            logits_flat = logits_2d[mask_flat]
            labels_flat = labels_2d[mask_flat]

            if logits_flat.numel() == 0:
                continue

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps_this_epoch += 1
            global_step += 1

            pbar.set_postfix({"loss": running_loss / max(1, steps_this_epoch)})

            # ----- STEP-BASED EVALUATION -----
            if global_step % eval_every == 0:
                val_loss, val_metrics = evaluate(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=device,
                    threshold=threshold,
                    desc=f"val @ step {global_step}",
                )

                if val_metrics:
                    val_uar = val_metrics["UAR"]
                    print(f"\nStep {global_step}: val_loss = {val_loss:.4f}, val_UAR = {val_uar:.4f}")
                else:
                    val_uar = -1.0
                    print(f"\nStep {global_step}: val_loss = {val_loss:.4f}, (no val metrics)")

                # Model selection by UAR
                if val_uar > best_val_uar:
                    best_val_uar = val_uar
                    best_step = global_step
                    evals_no_improve = 0

                    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
                    torch.save(model.state_dict(), output_weights)
                    print(f"  -> New best model (step {global_step}) saved to {output_weights}")
                else:
                    evals_no_improve += 1
                    print(f"  No UAR improvement for {evals_no_improve} eval(s).")

                # Early stopping based on eval count
                if evals_no_improve >= patience_evals:
                    print(f"\nEarly stopping: no UAR improvement for {evals_no_improve} evals.")
                    print(f"Best step: {best_step}, best val UAR: {best_val_uar:.4f}")
                    print(f"Best model weights saved to: {output_weights}")
                    return

        # optional: you could also eval at end of each epoch,
        # but pure step-based eval (like above) is already close to the paper.

    print("\nTraining complete.")
    print(f"Best step: {best_step}, best val UAR: {best_val_uar:.4f}")
    print(f"Best model weights saved to: {output_weights}")


# ========================= CLI ========================= #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT disfluency model on FluencyBank with step-based dev eval.")
    parser.add_argument("--train_metadata_path", type=str, required=True,
                        help="Path to train metadata (e.g. splits/train_metadata.csv).")
    parser.add_argument("--val_metadata_path", type=str, required=True,
                        help="Path to val metadata (e.g. splits/val_metadata.csv).")
    parser.add_argument("--word_dir", type=str, required=True,
                        help="Directory containing per-segment word CSVs.")
    parser.add_argument("--output_weights", type=str, default="./weights/language_fluencybank.pt",
                        help="File to save the best model weights.")
    parser.add_argument("--init_weights", type=str, default=None,
                        help="Path to existing weights (e.g. Switchboard language.pt) to initialize from.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for metrics.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--segid_column", type=str, default="segid",
                        help="Column in metadata that contains segment IDs.")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate on dev every N training steps.")
    parser.add_argument("--patience_evals", type=int, default=10,
                        help="Number of dev evals with no UAR improvement before early stopping.")
    args = parser.parse_args()

    train(
        train_metadata_path=args.train_metadata_path,
        val_metadata_path=args.val_metadata_path,
        word_dir=args.word_dir,
        output_weights=args.output_weights,
        init_weights=args.init_weights,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        threshold=args.threshold,
        device=args.device,
        segid_column=args.segid_column,
        eval_every=args.eval_every,
        patience_evals=args.patience_evals,
    )
