# speciesnet_trainer.py — Train & Evaluate Lightweight Species Classifier (augmented + logs)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import pandas as pd
import numpy as np

from config.settings import MEDIA_ROOT  # Root folder for image paths
from db.db import SessionLocal
from db.training_model import ModelRun, ModelResult


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class SpeciesDataset(Dataset):
    """
    Loads species-labeled JPEG images and maps species (common_name) to class IDs.
    A shared label_map must be passed so train/val/test use identical indices.
    """

    def __init__(self, dataframe, transform=None, label_map=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        if label_map is None:
            classes = sorted(self.df["common_name"].unique())
            self.label_map = {v: i for i, v in enumerate(classes)}
        else:
            self.label_map = label_map
        self.inverse_label_map = {i: v for v, i in self.label_map.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(MEDIA_ROOT) / row["jpeg_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_map[row["common_name"]]
        return image, label, row["jpeg_path"]


# ------------------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------------------
def get_transforms(train: bool):
    if train:
        # Realistic but strong-ish augmentations to improve generalization
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random'),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ------------------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, class_names):
    """
    Computes top-1/top-5 accuracy, confusion matrix, per-class report, and
    per-sample top-5 lists (for UI). Used for final epoch evaluation.
    """
    model.eval()
    y_true, y_pred = [], []
    top5_correct = 0
    n_samples = 0
    sample_rows = []

    for images, labels, paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1)
        k = min(5, probs.shape[1])
        top5_prob, top5_idx = probs.topk(k, dim=1)
        top5_hit = (top5_idx == labels.unsqueeze(1)).any(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        top5_correct += top5_hit.sum().item()
        n_samples += labels.size(0)

        for i in range(labels.size(0)):
            topk = [(class_names[int(top5_idx[i, j])], float(top5_prob[i, j].cpu()))
                    for j in range(k)]
            sample_rows.append({
                "path": str(paths[i]),
                "true": class_names[int(labels[i].cpu())],
                "pred": class_names[int(pred[i].cpu())],
                "top5": topk
            })

    acc_top1 = (sum(t == p for t, p in zip(y_true, y_pred)) / n_samples) if n_samples else 0.0
    acc_top5 = (top5_correct / n_samples) if n_samples else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    # Compute micro-F1 explicitly (for single-label multiclass, equals accuracy)
    try:
        from sklearn.metrics import precision_recall_fscore_support
        _, _, micro_f1_value, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    except Exception:
        micro_f1_value = acc_top1  # safe fallback

    return {
        "acc_top1": acc_top1,
        "acc_top5": acc_top5,
        "cm": cm.tolist(),
        "labels": class_names,
        "report": report,
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "micro_f1": float(micro_f1_value),
        "samples": sample_rows
    }


@torch.no_grad()
def eval_loss_and_acc(model, loader, device, criterion):
    """Lightweight epoch evaluation: average loss and top-1 accuracy."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# ------------------------------------------------------------------------------
# Group-by-day helper
# ------------------------------------------------------------------------------
def _derive_capture_day_column(df: pd.DataFrame, src_col: str = "capture_date",
                               out_col: str = "capture_day") -> pd.DataFrame:
    df = df.copy()
    if src_col not in df.columns:
        raise ValueError(f"Expected '{src_col}' column in DataFrame to derive groups by day.")

    def _to_day(v):
        if pd.isna(v):
            return "unknown"
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.strftime("%Y-%m-%d")
        s = str(v)
        try:
            return pd.to_datetime(s).strftime("%Y-%m-%d")
        except Exception:
            return s[:10]

    df[out_col] = df[src_col].map(_to_day)
    return df


# ------------------------------------------------------------------------------
# Main train + evaluate
# ------------------------------------------------------------------------------
def train_and_evaluate_speciesnet(
        df: pd.DataFrame,
        epochs: int = 20,
        lr: float = 1e-4,
        batch_size: int = 32,
        tag: str = "",
        val_size: float = 0.30,  # 70/30 default
        test_size: float | None = None,  # e.g., 0.20 for 60/20/20
        random_state: int = 42,
        use_capture_date_groups: bool = True,
        debug_split_checks: bool = True,
        use_class_weight_in_loss: bool = False  # set True to combine with sampler if desired
) -> dict:
    """
    Train ResNet18 with augmentations, balanced sampling, and per-epoch logging.
    """

    # --- Filter out species with too few images ---
    counts = df["common_name"].value_counts()
    valid_species = counts[counts >= 2].index
    df = df[df["common_name"].isin(valid_species)].reset_index(drop=True)

    # --- Build label map from full filtered df ---
    classes = sorted(df["common_name"].unique())
    label_map = {v: i for i, v in enumerate(classes)}

    # --- Optional: groups by capture day ---
    group_col = None
    if use_capture_date_groups:
        df = _derive_capture_day_column(df, "capture_date", "capture_day")
        group_col = "capture_day"

    # --- Split helpers ---
    def _strat_2(frame, frac):
        tr, va = train_test_split(frame, test_size=frac, stratify=frame["common_name"], random_state=random_state)
        return tr.reset_index(drop=True), va.reset_index(drop=True)

    def _strat_3(frame, val_frac, test_frac):
        holdout = val_frac + test_frac
        train_df, temp_df = train_test_split(frame, test_size=holdout, stratify=frame["common_name"],
                                             random_state=random_state)
        rel_test = test_frac / holdout
        val_df, test_df = train_test_split(temp_df, test_size=rel_test, stratify=temp_df["common_name"],
                                           random_state=random_state)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _group_2(frame, frac, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=random_state)
        i_tr, i_va = next(gss.split(frame, groups=groups))
        return frame.iloc[i_tr].reset_index(drop=True), frame.iloc[i_va].reset_index(drop=True)

    def _group_3(frame, val_frac, test_frac, groups):
        holdout = val_frac + test_frac
        gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=random_state)
        i_tr, i_temp = next(gss1.split(frame, groups=groups))
        train_df = frame.iloc[i_tr].reset_index(drop=True)
        temp_df = frame.iloc[i_temp].reset_index(drop=True)
        rel_test = test_frac / holdout
        temp_groups = temp_df[group_col].values
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=random_state)
        i_va, i_te = next(gss2.split(temp_df, groups=temp_groups))
        val_df = temp_df.iloc[i_va].reset_index(drop=True)
        test_df = temp_df.iloc[i_te].reset_index(drop=True)
        return train_df, val_df, test_df

    # --- Split ---
    if test_size and test_size > 0:
        if group_col is not None:
            train_df, val_df, test_df = _group_3(df, val_size, test_size, df[group_col].values)
        else:
            train_df, val_df, test_df = _strat_3(df, val_size, test_size)
    else:
        if group_col is not None:
            train_df, val_df = _group_2(df, val_size, df[group_col].values)
            test_df = None
        else:
            train_df, val_df = _strat_2(df, val_size)
            test_df = None

    # --- Optional split verification (prints to logs) ---
    if debug_split_checks:
        def _check(name_a, da, name_b, db_, grp):
            print(f"[SPLIT] {name_a}: {len(da)} rows, {name_b}: {len(db_)} rows")
            ca, cb = set(da["common_name"].unique()), set(db_["common_name"].unique())
            print(f"[SPLIT] same classes across {name_a}/{name_b}: {sorted(ca) == sorted(cb)}")
            ia, ib = set(da["jpeg_path"]), set(db_["jpeg_path"])
            print(f"[SPLIT] jpeg_path overlap: {len(ia & ib)} (should be 0)")
            if grp:
                ga, gb = set(da[grp]), set(db_[grp])
                print(f"[SPLIT] group overlap ({grp}): {len(ga & gb)} (should be 0)")

        _check("train", train_df, "val", val_df, group_col)
        if test_df is not None:
            _check("train", train_df, "test", test_df, group_col)
            _check("val", val_df, "test", test_df, group_col)

    # --- Transforms ---
    t_train = get_transforms(train=True)
    t_eval = get_transforms(train=False)

    # --- Datasets ---
    train_ds = SpeciesDataset(train_df, t_train, label_map=label_map)
    val_ds = SpeciesDataset(val_df, t_eval, label_map=label_map)
    test_ds = SpeciesDataset(test_df, t_eval, label_map=label_map) if test_df is not None else None

    # --- Balanced sampling (class imbalance) ---
    # Compute per-sample weights ~ 1/freq(class)
    train_counts = train_df["common_name"].value_counts()
    class_weights = {cls: 1.0 / cnt for cls, cnt in train_counts.items()}
    sample_weights = train_df["common_name"].map(class_weights).astype(float).values
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # --- Loaders ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds is not None else None

    # --- Model ---
    num_classes = len(classes)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Loss / Optim / Sched ---
    if use_class_weight_in_loss:
        # Optional: also weight the loss by class frequency (not required with sampler)
        weights_for_loss = torch.tensor(
            [1.0 / train_counts.get(cls, 1.0) for cls in classes],
            dtype=torch.float, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=weights_for_loss)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # --- Training loop with per-epoch logs ---
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            seen += labels.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)

        # Evaluate on validation each epoch
        val_loss, val_acc = eval_loss_and_acc(model, val_loader, device, criterion)

        # Step LR scheduler on val loss (no 'verbose' kwarg in this torch version)
        old_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)
        new_lr = float(optimizer.param_groups[0]["lr"])
        if new_lr != old_lr:
            print(f"[LR Scheduler] Reduced learning rate: {old_lr:.2e} → {new_lr:.2e}")

        # Save best (by val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # Log epoch
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

    # Load best weights before final eval
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # --- Final evaluation on validation (and test if present) ---
    class_names = classes
    eval_val = evaluate(model, val_loader, device, class_names)
    eval_test = evaluate(model, test_loader, device, class_names) if test_loader is not None else None

    # --- Save model ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    save_dir = Path(MEDIA_ROOT) / "models" / "speciesnet"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"speciesnet_{ts}{suffix}.pt"
    torch.save(model.state_dict(), output_path)

    # --- Build predictions DataFrame from val samples ---
    def fmt_top5(t5):  # list[(label, prob)]
        return " | ".join([f"{lbl} ({p:.2f})" for lbl, p in t5])

    pred_rows = [{
        "jpeg_path": s["path"],
        "true_label": s["true"],
        "predicted_label": s["pred"],
        "match": s["true"] == s["pred"],
        "top5": fmt_top5(s["top5"]),
    } for s in eval_val["samples"]]
    predictions_df = pd.DataFrame(pred_rows)

    # --- Persist to DB ---
    model_run_id = None
    with SessionLocal() as session:
        try:
            run_row = ModelRun(
                model_name="speciesnet",
                model_version="resnet18",
                tag=tag or None,
                epochs=int(epochs),
                lr=float(lr),
                batch_size=int(batch_size),
                num_classes=len(class_names),
                num_train=len(train_df),
                num_val=len(val_df),
                top1_accuracy=float(eval_val["acc_top1"]),
                top5_accuracy=float(eval_val["acc_top5"]),
                confusion_matrix=eval_val["cm"],
                classification_report=eval_val["report"],
                model_path=str(output_path),
                finished_at=datetime.utcnow()

            )
            session.add(run_row)
            session.flush()
            model_run_id = run_row.model_run_id

            results = []
            for s in eval_val["samples"]:
                results.append(ModelResult(
                    model_run_id=model_run_id,
                    jpeg_path=s["path"],
                    true_label=s["true"],
                    predicted_label=s["pred"],
                    correct=(s["true"] == s["pred"]),
                    top5=s["top5"],
                ))
            session.bulk_save_objects(results)
            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to persist model run/results: {e}")

    # --- Return (includes curves for easy plotting in Streamlit) ---
    out = {
        "model_run_id": model_run_id,
        "model_path": str(output_path),
        "val_acc": float(eval_val["acc_top1"]),
        "top5_acc": float(eval_val["acc_top5"]),
        "confusion_matrix": eval_val["cm"],
        "labels": eval_val["labels"],
        "classification_report": eval_val["report"],
        "predictions": predictions_df,
        "history": history,  # <-- epoch curves
    }
    if eval_test is not None:
        out.update({
            "test_acc": float(eval_test["acc_top1"]),
            "test_top5_acc": float(eval_test["acc_top5"]),
            "test_confusion_matrix": eval_test["cm"],
            "test_labels": eval_test["labels"],
            "test_classification_report": eval_test["report"],
            "test_macro_f1": float(eval_test["report"].get("macro avg", {}).get("f1-score", 0.0)),
            "test_weighted_f1": float(eval_test["report"].get("weighted avg", {}).get("f1-score", 0.0)),
            "test_micro_f1": float(eval_test.get("micro_f1", eval_test["acc_top1"])),
        })
    return out
