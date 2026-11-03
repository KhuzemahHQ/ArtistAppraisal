"""
Interactive Streamlit Web Demo for ArtistAppraisal:
- Live Painting Artist Prediction + Grad-CAM Brushwork Attribution
- Extracted Color Palette & Brushstroke/Chiaroscuro Style Profile
- Nearest Style Neighbors in 256-D Implicit Style Space vs. 128-D Gram Texture Space
- 2D t-SNE / PCA Style Manifold & Artist Style Similarity Matrix
- Scientific Evaluation & Ablation Probe Dashboard
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import ArtistStyleNet, get_transforms
from src.style_features import (
    GradCAMExtractor,
    compute_painting_style_stats,
    extract_dominant_palette,
    overlay_gradcam_on_image,
)

st.set_page_config(
    page_title="ArtistAppraisal — DNN Style & Artist Recognition",
    page_icon="🎨",
    layout="wide",
)


@st.cache_resource
def load_model_and_data():
    data_path = Path("data")
    out_path = Path("outputs")

    df = pd.read_csv(data_path / "metadata.csv")
    with open(data_path / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_artist = {int(k): v for k, v in label_map["idx_to_artist"].items()}

    with open(out_path / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    emb_data = np.load(out_path / "embeddings.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(out_path / "best_model.pt", map_location=device, weights_only=False)
    model = ArtistStyleNet(num_classes=len(idx_to_artist), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, eval_tf = get_transforms(image_size=ckpt.get("image_size", 224))
    return model, eval_tf, device, df, idx_to_artist, metrics, emb_data


def render_palette_html(palette):
    swatches = []
    for hex_color, pct in palette:
        swatches.append(
            f'<div style="flex:{max(pct, 0.08):.2f}; background:{hex_color}; height:44px; '
            f'display:flex; align-items:center; justify-content:center; color:#fff; '
            f'text-shadow:0 1px 3px rgba(0,0,0,0.85); font-size:12px; font-weight:600; '
            f'border-radius:6px; margin-right:4px;">{hex_color} ({pct*100:.0f}%)</div>'
        )
    return f'<div style="display:flex; width:100%; margin-bottom:12px;">{"".join(swatches)}</div>'


def main():
    st.title("🎨 ArtistAppraisal: Can a Deep Neural Network Learn Artistic Style?")
    st.markdown(
        "Investigating whether a Deep Neural Network (`ArtistStyleNet`: **ResNet50 + Multi-Scale Gatys Gram-Matrix Style Head**) "
        "can identify the painter of an artwork and **implicitly learn brushwork, texture, and palette representations**."
    )

    if not Path("outputs/best_model.pt").exists() or not Path("outputs/metrics.json").exists():
        st.warning("Model artifacts not found in `outputs/`. Please run `python run_pipeline.py` first.")
        return

    model, eval_tf, device, df, idx_to_artist, metrics, emb_data = load_model_and_data()

    # Top KPI banner
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    col_k1.metric("Curated Dataset", f"{metrics['total_paintings']:,} Paintings", f"{metrics['num_artists']} Iconic Artists")
    col_k2.metric("Held-Out Test Top-1 Acc", f"{metrics['test_top1_acc']*100:.1f}%", f"{metrics['test_top1_acc']*10:.1f}x Random Chance")
    col_k3.metric("Held-Out Test Top-3 Acc", f"{metrics['test_top3_acc']*100:.1f}%", "High Candidate Recall")
    gram_probe = next((p for p in metrics["probe_comparison"] if "Gram Texture Stream" in p["representation"]), None)
    if gram_probe:
        col_k4.metric("Pure Texture (Gram) Top-1", f"{gram_probe['top1_acc']*100:.1f}%", "Zero Spatial Layout")
    col_k5.metric("Macro F1-Score", f"{metrics['test_macro_f1']:.3f}", f"Test N={metrics['test_paintings']}")

    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Live Painting Appraisal & Grad-CAM",
            "🌌 Implicit Style Manifold & Similarity",
            "📊 Scientific Findings: Is Style Implicitly Learned?",
        ]
    )

    # =========================================================================
    # TAB 1: LIVE PAINTING APPRAISAL
    # =========================================================================
    with tab1:
        st.subheader("Inspect How the DNN Recognizes an Artist's Signature Style")
        src_col1, src_col2, src_col3 = st.columns([1.2, 1.5, 1.3])

        with src_col1:
            input_mode = st.radio(
                "Select Painting Source:",
                ["Held-Out Test Set Sample", "Upload Custom Painting Image"],
                horizontal=False,
            )

        selected_row = None
        img = None

        if input_mode == "Held-Out Test Set Sample":
            test_df = df[df["split"] == "test"].reset_index(drop=True)
            with src_col2:
                artist_filter = st.selectbox(
                    "Filter by True Artist:",
                    ["All Artists"] + sorted(test_df["artist"].unique().tolist()),
                )
            filtered_test = (
                test_df if artist_filter == "All Artists" else test_df[test_df["artist"] == artist_filter]
            ).reset_index(drop=True)

            with src_col3:
                options = [
                    f"[{r['artist']}] {r['title']} ({r['date']})"
                    for _, r in filtered_test.iterrows()
                ]
                chosen_idx = st.selectbox("Choose Held-Out Painting:", range(len(options)), format_func=lambda i: options[i])
                selected_row = filtered_test.iloc[chosen_idx]
                img = Image.open(selected_row["filepath"]).convert("RGB")
        else:
            with src_col2:
                uploaded = st.file_uploader("Upload a painting (.jpg, .png, .webp)", type=["jpg", "jpeg", "png", "webp"])
                if uploaded is not None:
                    img = Image.open(uploaded).convert("RGB")

        if img is not None:
            inp_tensor = eval_tf(img).unsqueeze(0).to(device)
            gradcam = GradCAMExtractor(model, model.layer4)
            heatmap, pred_idx, probs = gradcam.generate(inp_tensor)
            gradcam.remove_hooks()

            with torch.no_grad():
                _, feats = model(inp_tensor, return_features=True)
                query_style_emb = feats["style_embedding"][0].cpu().numpy()
                query_gram_emb = feats["gram_embedding"][0].cpu().numpy()

            overlay_img = overlay_gradcam_on_image(img.resize((320, 320)), heatmap, alpha=0.45)

            vis_col1, vis_col2, vis_col3 = st.columns([1.05, 1.05, 1.4])
            with vis_col1:
                st.image(img.resize((320, 320)), caption="Input Painting", use_container_width=True)
                if selected_row is not None:
                    st.caption(
                        f"**True Artist:** {selected_row['artist']}  \n"
                        f"**Title:** *{selected_row['title']}* ({selected_row['date']})  \n"
                        f"**Art Movement:** {selected_row['style']} | **Genre:** {selected_row['genre']}"
                    )

            with vis_col2:
                st.image(
                    overlay_img,
                    caption=f"Grad-CAM Attribution for '{idx_to_artist[pred_idx]}'",
                    use_container_width=True,
                )
                st.caption(
                    "Warm red/yellow regions show the specific brushstrokes, contours, and lighting "
                    "textures that most strongly activated the predicted artist neuron."
                )

            with vis_col3:
                pred_artist = idx_to_artist[pred_idx]
                top_conf = probs[pred_idx] * 100
                if selected_row is not None:
                    is_correct = pred_artist == selected_row["artist"]
                    badge = "✅ Correct Prediction" if is_correct else f"⚠️ Confused with {pred_artist}"
                    st.markdown(f"### Predicted: **{pred_artist}** ({top_conf:.1f}%) — {badge}")
                else:
                    st.markdown(f"### Predicted Artist: **{pred_artist}** ({top_conf:.1f}%)")

                st.markdown("**Top-5 Artist Probabilities:**")
                top5_indices = np.argsort(-probs)[:5]
                for rank, c_idx in enumerate(top5_indices, 1):
                    a_name = idx_to_artist[int(c_idx)]
                    p_val = float(probs[c_idx])
                    st.progress(p_val, text=f"#{rank} {a_name}: {p_val*100:.1f}%")

                st.markdown("**Extracted Dominant Color Palette:**")
                palette = extract_dominant_palette(img, n_colors=5)
                st.markdown(render_palette_html(palette), unsafe_allow_html=True)

                style_stats = compute_painting_style_stats(img)
                s_cols = st.columns(3)
                for idx_s, (k, v) in enumerate(style_stats.items()):
                    s_cols[idx_s % 3].metric(k, f"{v*100:.0f}/100")

            st.divider()
            st.subheader("🔗 Nearest Style Neighbors in Learned Representation Space")
            nn_mode = st.radio(
                "Similarity Metric Space:",
                [
                    "Unified Implicit Style Embedding (256-D: Brushwork + Composition)",
                    "Pure Gatys Gram Texture Embedding (128-D: Spatial-Invariant Brushwork Only)",
                ],
                horizontal=True,
            )
            if "Gram" in nn_mode:
                db_embs = emb_data["gram_embeddings"]
                q_emb = query_gram_emb
            else:
                db_embs = emb_data["style_embeddings"]
                q_emb = query_style_emb

            sims = db_embs @ q_emb
            order = np.argsort(-sims)
            # Exclude exact self if test sample
            neighbors = []
            for idx_cand in order:
                cand_row = df.iloc[int(idx_cand)]
                if selected_row is not None and cand_row["id"] == selected_row["id"]:
                    continue
                neighbors.append((cand_row, float(sims[idx_cand])))
                if len(neighbors) >= 4:
                    break

            nn_cols = st.columns(4)
            for col_nn, (nrow, sim_score) in zip(nn_cols, neighbors):
                with col_nn:
                    n_img = Image.open(nrow["filepath"]).convert("RGB").resize((240, 240))
                    st.image(n_img, use_container_width=True)
                    st.markdown(
                        f"**{nrow['artist']}** (`cos={sim_score:.3f}`)  \n"
                        f"*{nrow['title']}* ({nrow['style']})"
                    )

    # =========================================================================
    # TAB 2: IMPLICIT STYLE MANIFOLD
    # =========================================================================
    with tab2:
        st.subheader("How Does the DNN Organize Artists & Movements in Latent Style Space?")
        st.markdown(
            "Without ever being given explicit labels for *Impressionism*, *Baroque Chiaroscuro*, *Surrealism*, or *Northern Renaissance*, "
            "`ArtistStyleNet` organizes paintings into coherent stylistic neighborhoods."
        )
        if Path("outputs/style_tsne.png").exists():
            st.image("outputs/style_tsne.png", use_container_width=True)

        st.divider()
        st.subheader("Pairwise Artist Style Similarity Matrix (Cosine Similarity of Learned Style Centroids)")
        style_embs = emb_data["style_embeddings"]
        labels_arr = emb_data["labels"]
        artist_names = metrics["artist_names"]
        centroids = []
        for i in range(len(artist_names)):
            c = style_embs[labels_arr == i].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-8)
            centroids.append(c)
        centroid_mat = np.stack(centroids, axis=0)
        sim_mat = centroid_mat @ centroid_mat.T

        fig_sim, ax_sim = plt.subplots(figsize=(8.5, 6.5), dpi=140)
        im = ax_sim.imshow(sim_mat, cmap="YlGnBu", vmin=-0.2, vmax=1.0)
        plt.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04, label="Cosine Similarity")
        ax_sim.set_xticks(range(len(artist_names)))
        ax_sim.set_yticks(range(len(artist_names)))
        ax_sim.set_xticklabels(artist_names, rotation=40, ha="right", fontsize=9)
        ax_sim.set_yticklabels(artist_names, fontsize=9)
        for i in range(len(artist_names)):
            for j in range(len(artist_names)):
                ax_sim.text(
                    j,
                    i,
                    f"{sim_mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if sim_mat[i, j] > 0.55 else "black",
                )
        ax_sim.set_title("Learned Style Centroid Cosine Similarity Across Artists", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_sim, use_container_width=False)

    # =========================================================================
    # TAB 3: SCIENTIFIC EVALUATION
    # =========================================================================
    with tab3:
        st.subheader("Empirical Answer: Does a DNN Implicitly Learn Artistic Style?")
        st.markdown(
            """
            To rigorously test **what** the neural network learns when predicting an artist, we compare five representations on the exact same held-out test paintings:
            1. **Random Guess Baseline** ($10.0\\%$ Top-1 across 10 artists).
            2. **Handcrafted Color Palette + Brushstroke Edges (64-D)**: HSV color distributions, luminance contrast, and Sobel edge orientation histograms.
            3. **Raw Gatys Gram Matrices (`layer1`–`layer3`, 900-D)**: Feature-map cross-correlations $G_{ij}^l = \\frac{1}{HW}\\sum_{h,w} F_{i,h,w}^l F_{j,h,w}^l$ where **all spatial coordinates $(h,w)$ are averaged out**, leaving purely local texture and brushwork statistics.
            4. **Learned Gram Texture Stream (128-D)**: Our fine-tuned Gram-matrix projection head (still 100% spatial-invariant texture).
            5. **Full `ArtistStyleNet` (256-D)**: End-to-end fusion of multi-scale Gram brushwork and deep compositional features.
            """
        )

        probe_df = pd.DataFrame(metrics["probe_comparison"])
        probe_df["Top-1 Accuracy (%)"] = (probe_df["top1_acc"] * 100).round(1)
        probe_df["Top-3 Accuracy (%)"] = (probe_df["top3_acc"] * 100).round(1)
        st.dataframe(
            probe_df[["representation", "dims", "Top-1 Accuracy (%)", "Top-3 Accuracy (%)", "description"]],
            use_container_width=True,
            hide_index=True,
        )

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            if Path("outputs/style_probe_comparison.png").exists():
                st.image("outputs/style_probe_comparison.png", caption="Ablation Probe Comparison", use_container_width=True)
        with col_fig2:
            if Path("outputs/confusion_matrix.png").exists():
                st.image("outputs/confusion_matrix.png", caption="Held-Out Test Confusion Matrix", use_container_width=True)

        st.subheader("Per-Artist Held-Out Test Performance & Stylistic Confusions")
        per_artist_df = pd.DataFrame(metrics["per_artist"])
        per_artist_df["Precision (%)"] = (per_artist_df["precision"] * 100).round(1)
        per_artist_df["Top-1 Recall (%)"] = (per_artist_df["recall_top1"] * 100).round(1)
        per_artist_df["Top-3 Recall (%)"] = (per_artist_df["top3_acc"] * 100).round(1)
        per_artist_df["F1 Score"] = per_artist_df["f1_score"].round(3)
        st.dataframe(
            per_artist_df[
                ["artist", "Precision (%)", "Top-1 Recall (%)", "Top-3 Recall (%)", "F1 Score", "support", "most_confused_with"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if Path("outputs/gradcam_gallery.png").exists():
            st.subheader("Grad-CAM Visual Attribution Across Held-Out Test Paintings")
            st.image("outputs/gradcam_gallery.png", use_container_width=True)


if __name__ == "__main__":
    main()

