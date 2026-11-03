"""
Comprehensive evaluation & scientific style-probing suite for `ArtistStyleNet`.

Answers: "Can a DNN predict an artist from a painting, and is artistic style implicitly learned?"
1. Evaluates Top-1, Top-3, Macro-F1, and Per-Artist accuracy on the held-out test set.
2. Runs controlled Style Probes (Handcrafted Palette/Brushwork vs. Spatial-Invariant Gatys Gram
   Matrices vs. Full Fine-Tuned DNN) to quantify how much of artist recognition comes from local
   brushstroke/texture vs. high-level composition.
3. Computes 2D t-SNE & PCA embeddings of the learned style manifold and saves `outputs/embeddings.npz`.
4. Generates high-resolution figures (`confusion_matrix.png`, `style_tsne.png`,
   `style_probe_comparison.png`, `training_curves.png`, `gradcam_gallery.png`).
"""

import json
import os
from pathlib import Path

# Prevent joblib/loky on Windows 11 24H2+ from invoking removed `wmic.exe`
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 4) // 2)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.model import ArtistStyleNet, PaintingDataset, get_transforms
from src.style_features import (
    GradCAMExtractor,
    extract_handcrafted_style_vector,
    overlay_gradcam_on_image,
)


def top_k_acc_from_probs(probs: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    top_k_preds = np.argsort(-probs, axis=1)[:, :k]
    matches = (top_k_preds == labels[:, None]).any(axis=1)
    return float(matches.mean())


def run_evaluation(
    data_dir: str = "data",
    output_dir: str = "outputs",
    batch_size: int = 32,
):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path / "metadata.csv")
    with open(data_path / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_artist = {int(k): v for k, v in label_map["idx_to_artist"].items()}
    artist_names = [idx_to_artist[i] for i in range(len(idx_to_artist))]
    num_classes = len(artist_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = out_path / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    image_size = ckpt.get("image_size", 224)

    model = ArtistStyleNet(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, eval_tf = get_transforms(image_size=image_size)
    full_ds = PaintingDataset(df, root_dir=".", transform=eval_tf)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print("[Evaluator] Extracting DNN style embeddings, Gram matrices, and handcrafted features...")
    all_logits = []
    all_probs = []
    all_style_emb = []
    all_gram_raw = []
    all_gram_emb = []
    all_sem_emb = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in full_loader:
            images = images.to(device)
            logits, feats = model(images, return_features=True)
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_style_emb.append(feats["style_embedding"].cpu().numpy())
            all_gram_raw.append(feats["gram_raw"].cpu().numpy())
            all_gram_emb.append(feats["gram_embedding"].cpu().numpy())
            all_sem_emb.append(feats["semantic_embedding"].cpu().numpy())
            all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_style_emb = np.concatenate(all_style_emb, axis=0)
    all_gram_raw = np.concatenate(all_gram_raw, axis=0)
    all_gram_emb = np.concatenate(all_gram_emb, axis=0)
    all_sem_emb = np.concatenate(all_sem_emb, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Extract handcrafted color palette & edge/brushwork features for every image
    handcrafted_feats = []
    for rel_path in df["filepath"]:
        img = Image.open(rel_path).convert("RGB")
        handcrafted_feats.append(extract_handcrafted_style_vector(img))
    all_handcrafted = np.stack(handcrafted_feats, axis=0)

    # Split masks
    is_train = (df["split"] == "train").values
    is_val = (df["split"] == "val").values
    is_test = (df["split"] == "test").values

    y_train, y_test = all_labels[is_train], all_labels[is_test]
    probs_test = all_probs[is_test]
    preds_test = probs_test.argmax(axis=1)

    # 1. Main DNN Test Metrics
    test_top1 = float((preds_test == y_test).mean())
    test_top3 = top_k_acc_from_probs(probs_test, y_test, k=3)
    test_macro_f1 = float(f1_score(y_test, preds_test, average="macro"))

    # 2. Controlled Style Representation Probes (Train -> Test)
    print("[Evaluator] Fitting controlled style probes (Handcrafted vs. Gram Texture vs. Full DNN)...")

    # Probe A: Handcrafted Color Palette + Edge/Brushstroke Orientation (64-D)
    clf_handcrafted = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    )
    clf_handcrafted.fit(all_handcrafted[is_train], y_train)
    probs_hc = clf_handcrafted.predict_proba(all_handcrafted[is_test])
    hc_top1 = float((probs_hc.argmax(axis=1) == y_test).mean())
    hc_top3 = top_k_acc_from_probs(probs_hc, y_test, k=3)

    # Probe B: Pure Spatial-Invariant Gatys Gram Matrices (`layer1`+`layer2`+`layer3`, 900-D)
    clf_gram = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=0.2, random_state=42)
    )
    clf_gram.fit(all_gram_raw[is_train], y_train)
    probs_gram = clf_gram.predict_proba(all_gram_raw[is_test])
    gram_top1 = float((probs_gram.argmax(axis=1) == y_test).mean())
    gram_top3 = top_k_acc_from_probs(probs_gram, y_test, k=3)

    # Probe C: Learned Gram Texture Stream (128-D)
    clf_gram_emb = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_gram_emb.fit(all_gram_emb[is_train], y_train)
    probs_gram_emb = clf_gram_emb.predict_proba(all_gram_emb[is_test])
    gram_emb_top1 = float((probs_gram_emb.argmax(axis=1) == y_test).mean())
    gram_emb_top3 = top_k_acc_from_probs(probs_gram_emb, y_test, k=3)

    # Probe D: Deep Semantic Stream Alone (256-D)
    clf_sem = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf_sem.fit(all_sem_emb[is_train], y_train)
    probs_sem = clf_sem.predict_proba(all_sem_emb[is_test])
    sem_top1 = float((probs_sem.argmax(axis=1) == y_test).mean())
    sem_top3 = top_k_acc_from_probs(probs_sem, y_test, k=3)

    probe_comparison = [
        {
            "representation": "Random Guess Baseline",
            "dims": 0,
            "top1_acc": round(1.0 / num_classes, 4),
            "top3_acc": round(3.0 / num_classes, 4),
            "description": "Uniform random guessing across the 10 artists",
        },
        {
            "representation": "Handcrafted Palette + Brushstroke Edges",
            "dims": 64,
            "top1_acc": round(hc_top1, 4),
            "top3_acc": round(hc_top3, 4),
            "description": "HSV color histograms + luminance contrast + Sobel edge orientation entropy",
        },
        {
            "representation": "Raw Gatys Gram Matrices (Texture Only)",
            "dims": 900,
            "top1_acc": round(gram_top1, 4),
            "top3_acc": round(gram_top3, 4),
            "description": "Spatial-invariant CNN feature correlations (layer1–3) with all object layout averaged out",
        },
        {
            "representation": "Learned Gram Texture Stream",
            "dims": 128,
            "top1_acc": round(gram_emb_top1, 4),
            "top3_acc": round(gram_emb_top3, 4),
            "description": "Fine-tuned multi-scale Gram matrix projection head (pure brushwork/texture)",
        },
        {
            "representation": "Full ArtistStyleNet (Gram + Semantic Fusion)",
            "dims": 256,
            "top1_acc": round(test_top1, 4),
            "top3_acc": round(test_top3, 4),
            "description": "End-to-end hybrid DNN combining multi-scale Gram brushwork & deep compositional features",
        },
    ]

    # Per-artist metrics
    report_dict = classification_report(
        y_test, preds_test, target_names=artist_names, output_dict=True, zero_division=0
    )
    per_artist_metrics = []
    for idx, artist in enumerate(artist_names):
        mask_a = y_test == idx
        top3_a = top_k_acc_from_probs(probs_test[mask_a], y_test[mask_a], k=3) if mask_a.any() else 0.0
        # Most common confusion for this artist
        wrong_mask = mask_a & (preds_test != y_test)
        if wrong_mask.any():
            vals, counts = np.unique(preds_test[wrong_mask], return_counts=True)
            most_confused = idx_to_artist[int(vals[np.argmax(counts)])]
        else:
            most_confused = "None (100% accurate)"

        per_artist_metrics.append(
            {
                "artist": artist,
                "precision": round(float(report_dict[artist]["precision"]), 4),
                "recall_top1": round(float(report_dict[artist]["recall"]), 4),
                "top3_acc": round(float(top3_a), 4),
                "f1_score": round(float(report_dict[artist]["f1-score"]), 4),
                "support": int(report_dict[artist]["support"]),
                "most_confused_with": most_confused,
            }
        )

    # 3. Compute 2D t-SNE and PCA on learned 256-D Implicit Style Embeddings
    print("[Evaluator] Computing 2D t-SNE and PCA style manifold coordinates...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(all_style_emb)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsne_coords = tsne.fit_transform(all_style_emb)

    # Also compute t-SNE on pure Gram texture embeddings to compare
    tsne_gram = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    tsne_gram_coords = tsne_gram.fit_transform(all_gram_emb)

    np.savez_compressed(
        out_path / "embeddings.npz",
        style_embeddings=all_style_emb,
        gram_embeddings=all_gram_emb,
        handcrafted_features=all_handcrafted,
        probs=all_probs,
        labels=all_labels,
        pca_coords=pca_coords,
        tsne_coords=tsne_coords,
        tsne_gram_coords=tsne_gram_coords,
    )

    # Save comprehensive metrics JSON
    cm = confusion_matrix(y_test, preds_test)
    metrics_payload = {
        "num_artists": num_classes,
        "total_paintings": len(df),
        "test_paintings": int(is_test.sum()),
        "test_top1_acc": round(test_top1, 4),
        "test_top3_acc": round(test_top3, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "probe_comparison": probe_comparison,
        "per_artist": per_artist_metrics,
        "artist_names": artist_names,
        "confusion_matrix": cm.tolist(),
    }
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 4. Generate Visualizations
    print("[Evaluator] Generating scientific figures in outputs/...")
    _plot_confusion_matrix(cm, artist_names, out_path / "confusion_matrix.png")
    _plot_style_tsne(tsne_coords, tsne_gram_coords, all_labels, artist_names, df["style"].values, out_path / "style_tsne.png")
    _plot_probe_comparison(probe_comparison, out_path / "style_probe_comparison.png")
    if (out_path / "training_history.json").exists():
        with open(out_path / "training_history.json", "r", encoding="utf-8") as f:
            history = json.load(f)
        _plot_training_curves(history, out_path / "training_curves.png")

    _generate_gradcam_gallery(model, df[is_test], idx_to_artist, eval_tf, device, out_path / "gradcam_gallery.png")

    print("\n=== SCIENTIFIC EVALUATION SUMMARY ===")
    for p in probe_comparison:
        print(f"  {p['representation']:<45} | Top-1: {p['top1_acc']*100:5.1f}% | Top-3: {p['top3_acc']*100:5.1f}%")
    return metrics_payload


def _plot_confusion_matrix(cm: np.ndarray, artist_names: list, save_path: Path):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall Proportion")

    ax.set_xticks(np.arange(len(artist_names)))
    ax.set_yticks(np.arange(len(artist_names)))
    ax.set_xticklabels(artist_names, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticklabels(artist_names, fontsize=9.5)
    ax.set_xlabel("Predicted Artist", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Artist", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set Confusion Matrix (ArtistStyleNet)", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(artist_names)):
        for j in range(len(artist_names)):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            if val > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "#1f2937"
                ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_style_tsne(
    tsne_coords: np.ndarray,
    tsne_gram_coords: np.ndarray,
    labels: np.ndarray,
    artist_names: list,
    styles: np.ndarray,
    save_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=160)
    cmap = plt.get_cmap("tab10")

    for i, artist in enumerate(artist_names):
        mask = labels == i
        axes[0].scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            label=artist,
            color=cmap(i % 10),
            alpha=0.78,
            s=28,
            edgecolors="none",
        )
        axes[1].scatter(
            tsne_gram_coords[mask, 0],
            tsne_gram_coords[mask, 1],
            label=artist,
            color=cmap(i % 10),
            alpha=0.78,
            s=28,
            edgecolors="none",
        )

    axes[0].set_title("t-SNE of Full Implicit Style Embeddings (256-D)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].grid(True, alpha=0.2)

    axes[1].set_title("t-SNE of Pure Gatys Gram Texture Stream (128-D, Spatial-Invariant)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9.5, frameon=True, title="Artist")

    plt.suptitle("Implicit Artistic Style Clustering in Deep Feature Space", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_probe_comparison(probe_comparison: list, save_path: Path):
    names = [p["representation"] for p in probe_comparison]
    top1 = [p["top1_acc"] * 100 for p in probe_comparison]
    top3 = [p["top3_acc"] * 100 for p in probe_comparison]

    y = np.arange(len(names))
    height = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    bars1 = ax.barh(y - height / 2, top1, height, label="Top-1 Accuracy (%)", color="#2563eb")
    bars3 = ax.barh(y + height / 2, top3, height, label="Top-3 Accuracy (%)", color="#60a5fa")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Held-Out Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_title("How Much of Artist Identity Is Pure Style/Texture? (Ablation Probes)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold", color="#1e3a8a")
    for bar in bars3:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, color="#1d4ed8")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_training_curves(history: list, save_path: Path):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_top1_acc"] * 100 for h in history]
    val_acc = [h["val_top1_acc"] * 100 for h in history]
    val_top3 = [h["val_top3_acc"] * 100 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#dc2626")
    axes[0].plot(epochs, val_loss, "s--", label="Val Loss", color="#2563eb")
    axes[0].set_title("Cross-Entropy Loss (with Label Smoothing)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, "o-", label="Train Top-1 Acc", color="#16a34a")
    axes[1].plot(epochs, val_acc, "s-", label="Val Top-1 Acc", color="#2563eb")
    axes[1].plot(epochs, val_top3, "^--", label="Val Top-3 Acc", color="#9333ea")
    axes[1].set_title("Classification Accuracy (%)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _generate_gradcam_gallery(
    model: ArtistStyleNet,
    test_df: pd.DataFrame,
    idx_to_artist: dict,
    eval_tf,
    device: torch.device,
    save_path: Path,
):
    """Generates a 2x5 gallery of Grad-CAM style attribution overlays across different artists."""
    gradcam = GradCAMExtractor(model, model.layer4)
    sampled_rows = []
    for label_idx in sorted(idx_to_artist.keys())[:5]:
        sub = test_df[test_df["label"] == label_idx]
        if len(sub) > 0:
            sampled_rows.append(sub.iloc[0])

    fig, axes = plt.subplots(2, len(sampled_rows), figsize=(3.3 * len(sampled_rows), 6.6), dpi=160)
    for col, row in enumerate(sampled_rows):
        img = Image.open(row["filepath"]).convert("RGB")
        inp = eval_tf(img).unsqueeze(0).to(device)
        heatmap, pred_idx, probs = gradcam.generate(inp)
        overlay = overlay_gradcam_on_image(img.resize((224, 224)), heatmap, alpha=0.45)

        true_artist = row["artist"]
        pred_artist = idx_to_artist[pred_idx]
        conf = probs[pred_idx] * 100

        axes[0, col].imshow(img.resize((224, 224)))
        axes[0, col].set_title(f"True: {true_artist}\n\"{row['title'][:22]}\"", fontsize=9, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"Grad-CAM Focus\nPred: {pred_artist} ({conf:.0f}%)", fontsize=9, color="#1e3a8a")
        axes[1, col].axis("off")

    gradcam.remove_hooks()
    plt.suptitle("Grad-CAM Visual Style Attribution: Which Brushstrokes & Motifs Drive Artist Prediction?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    run_evaluation()

