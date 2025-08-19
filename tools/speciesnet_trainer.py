# speciesnet_trainer.py — Train & Evaluate Lightweight Species Classifier

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import numpy as np
from config.settings import MEDIA_ROOT  # Root folder for image paths
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
from db.db import SessionLocal
from db.training_model import ModelRun, ModelResult


# ------------------------------------------------------------------------------
# Custom Dataset Class
# ------------------------------------------------------------------------------
class SpeciesDataset(Dataset):
    """
    Loads species-labeled JPEG images and maps species (common_name) to class IDs.
    """
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        # Map common_name -> class id
        self.label_map = {v: i for i, v in enumerate(sorted(self.df["common_name"].unique()))}
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


@torch.no_grad()
def evaluate(model, val_loader, device, class_names):
    """
    Runs evaluation to compute top-1/top-5 accuracy, confusion matrix, per-class report,
    and per-sample top-5 lists.
    """
    model.eval()
    y_true, y_pred = [], []
    top5_correct = 0
    n_samples = 0
    sample_rows = []  # {"path": str, "true": str, "pred": str, "top5": [(label, prob), ...]}

    for images, labels, paths in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)

        # top-1
        pred = torch.argmax(probs, dim=1)

        # top-5 (works even if num_classes < 5)
        k = min(5, probs.shape[1])
        top5_prob, top5_idx = probs.topk(k, dim=1)
        top5_hit = (top5_idx == labels.unsqueeze(1)).any(dim=1)

        # bookkeeping
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        top5_correct += top5_hit.sum().item()
        n_samples += labels.size(0)

        # collect rows for UI
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

    return {
        "acc_top1": acc_top1,
        "acc_top5": acc_top5,
        "cm": cm.tolist(),       # JSON-safe
        "labels": class_names,
        "report": report,        # dict[str]->metrics
        "samples": sample_rows   # list of dicts
    }


# ------------------------------------------------------------------------------
# Training + Evaluation Function
# ------------------------------------------------------------------------------
def train_and_evaluate_speciesnet(
    df: pd.DataFrame,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 32,
    tag: str = ""
) -> dict:
    """
    Train a ResNet18 model on labeled wildlife image data.

    Args:
        df: DataFrame containing 'jpeg_path' and 'common_name'.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        tag: Optional string suffix for saved model filename.

    Returns:
        Dict with model path, top-1/top-5 accuracy, confusion matrix, per-class report,
        and a predictions DataFrame (including top-5 strings).
    """

    # --- Step 0: Filter out species with fewer than 2 images ---
    counts = df["common_name"].value_counts()
    valid_species = counts[counts >= 2].index
    df = df[df["common_name"].isin(valid_species)].reset_index(drop=True)

    # --- Step 1: Stratified train/validation split ---
    train_df, val_df = train_test_split(
        df, test_size=0.5, stratify=df["common_name"], random_state=42
    )

    # --- Step 2: Image preprocessing transforms ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Step 3: Dataset and DataLoader setup ---
    train_ds = SpeciesDataset(train_df, transform)
    val_ds   = SpeciesDataset(val_df, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    num_classes = len(train_ds.label_map)

    # --- Step 4: Load pretrained ResNet18 and replace classifier ---
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # --- Step 5: Move to GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Step 6: Set loss function and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Step 7: Training loop ---
    for _ in range(epochs):
        model.train()
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # --- Step 8: Evaluation on validation set ---
    # Build class_names (ordered by the dataset's inverse_label_map -> common_name)
    class_names = [val_ds.inverse_label_map[i] for i in range(len(val_ds.inverse_label_map))]
    eval_out = evaluate(model, val_loader, device, class_names)

    # --- Step 9: Save model to disk ---


    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    save_dir = Path(MEDIA_ROOT) / "models" / "speciesnet"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"speciesnet_{ts}{suffix}.pt"

    try:
        torch.save(model.state_dict(), output_path)
    except Exception as e:
        # bubble up a helpful error
        raise RuntimeError(f"Failed to save model state_dict to {output_path}: {e}")


    # --- Step 10: Build predictions DataFrame from eval_out samples ---
    def fmt_top5(t5):  # list[(label, prob)]
        return " | ".join([f"{lbl} ({p:.2f})" for lbl, p in t5])


    pred_rows = [{
        "jpeg_path": s["path"],
        "true_label": s["true"],
        "predicted_label": s["pred"],
        "match": s["true"] == s["pred"],
        "top5": fmt_top5(s["top5"]),
    } for s in eval_out["samples"]]
    predictions_df = pd.DataFrame(pred_rows)

    # --- Step 11: Persist to DB (model_run + model_result) ---
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
                top1_accuracy=float(eval_out["acc_top1"]),
                top5_accuracy=float(eval_out["acc_top5"]),
                confusion_matrix=eval_out["cm"],  # JSONB-safe (list of lists)
                classification_report=eval_out["report"],  # JSONB-safe (dict)
                model_path=str(output_path),
                finished_at=datetime.utcnow(),
            )
            session.add(run_row)
            session.flush()  # get PK
            model_run_id = run_row.model_run_id

            # Bulk insert per-sample rows (store raw top5 JSON, not the pretty string)
            results = []
            for s in eval_out["samples"]:
                results.append(ModelResult(
                    model_run_id=model_run_id,
                    jpeg_path=s["path"],
                    true_label=s["true"],
                    predicted_label=s["pred"],
                    correct=(s["true"] == s["pred"]),
                    top5=s["top5"],  # list[[label, prob], ...] — JSONB-safe
                ))
            session.bulk_save_objects(results)
            session.commit()
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Failed to persist model run/results: {e}")

    # --- Step 12: Return results (including run id) ---
    return {
        "model_run_id": model_run_id,
        "model_path": str(output_path),
        "val_acc": float(eval_out["acc_top1"]),
        "top5_acc": float(eval_out["acc_top5"]),
        "confusion_matrix": eval_out["cm"],
        "labels": eval_out["labels"],
        "classification_report": eval_out["report"],
        "predictions": predictions_df,
    }

