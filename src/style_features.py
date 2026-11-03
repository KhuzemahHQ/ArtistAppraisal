"""
Style feature extraction and visual attribution utilities:
1. Handcrafted Color Palette & Brushstroke/Edge Statistics (interpretable baseline).
2. Gatys-style Gram Matrix texture descriptors from intermediate CNN feature maps
   (spatial-invariant brushstroke & color co-occurrence statistics).
3. Grad-CAM (Gradient-weighted Class Activation Mapping) for visualizing which
   regions/brushstrokes of a painting drive the DNN's artist prediction.
"""

from typing import Dict, List, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def extract_handcrafted_style_vector(img: Image.Image) -> np.ndarray:
    """
    Computes a 64-dimensional interpretable handcrafted style vector from a PIL Image:
    - 24-bin Hue histogram + 12-bin Saturation histogram + 12-bin Value/Luminance histogram (48 dims)
    - RGB channel means & standard deviations (6 dims)
    - Sobel gradient magnitude statistics (mean, std, 90th percentile = brushwork intensity) (3 dims)
    - 7-bin Edge Orientation histogram + orientation entropy (brushstroke directionality) (7 dims)
    """
    img_small = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    rgb = np.asarray(img_small, dtype=np.float32) / 255.0

    # HSV histograms
    hsv = np.asarray(img_small.convert("HSV"), dtype=np.float32) / 255.0
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=24, range=(0.0, 1.0), density=True)
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=12, range=(0.0, 1.0), density=True)
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=12, range=(0.0, 1.0), density=True)
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    s_hist = s_hist / (s_hist.sum() + 1e-8)
    v_hist = v_hist / (v_hist.sum() + 1e-8)

    # RGB moments
    rgb_mean = rgb.mean(axis=(0, 1))
    rgb_std = rgb.std(axis=(0, 1))

    # Grayscale luminance & finite-difference Sobel-like gradients for brushstroke texture
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]

    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # [0, 1]

    mag_stats = np.array(
        [mag.mean(), mag.std(), np.percentile(mag, 90)], dtype=np.float32
    )
    orient_hist, _ = np.histogram(
        angle, bins=6, range=(0.0, 1.0), weights=mag + 1e-6, density=True
    )
    orient_hist = orient_hist / (orient_hist.sum() + 1e-8)
    orient_entropy = np.array(
        [-np.sum(orient_hist * np.log(orient_hist + 1e-8))], dtype=np.float32
    )

    feat = np.concatenate(
        [h_hist, s_hist, v_hist, rgb_mean, rgb_std, mag_stats, orient_hist, orient_entropy]
    ).astype(np.float32)
    return feat


def extract_dominant_palette(img: Image.Image, n_colors: int = 5) -> List[Tuple[str, float]]:
    """
    Returns top `n_colors` dominant hex colors and their proportions for UI visualization.
    """
    small = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32).reshape(-1, 3)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=256, n_init=3)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels, minlength=n_colors)
    order = np.argsort(-counts)
    total = float(len(labels))
    palette = []
    for idx in order:
        r, g, b = np.clip(kmeans.cluster_centers_[idx], 0, 255).astype(int)
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        palette.append((hex_code, float(counts[idx] / total)))
    return palette


def compute_painting_style_stats(img: Image.Image) -> Dict[str, float]:
    """
    Computes human-readable brushstroke & color style metrics for the interactive demo.
    """
    small = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = gray[1:-1, 2:] - gray[1:-1, :-2]
    gy = gray[2:, 1:-1] - gray[:-2, 1:-1]
    mag = np.hypot(gx, gy)

    return {
        "Brushwork / Edge Energy": float(np.clip(mag.mean() * 8.5, 0.0, 1.0)),
        "Chiaroscuro (Luminance Contrast)": float(np.clip(hsv[:, :, 2].std() * 3.2, 0.0, 1.0)),
        "Color Saturation": float(hsv[:, :, 1].mean()),
        "Overall Brightness": float(hsv[:, :, 2].mean()),
        "Warmth (Red-Yellow vs Blue)": float(
            np.clip(0.5 + 1.5 * ((rgb[:, :, 0].mean() - rgb[:, :, 2].mean())), 0.0, 1.0)
        ),
    }


def compute_gram_matrix_descriptor(feature_map: torch.Tensor, pool_channels: int = 32) -> torch.Tensor:
    """
    Computes a compact, spatial-invariant Gatys Gram matrix style vector from a 4D CNN
    activation tensor (B, C, H, W).
    Always computes in FP32 and scales by 1/sqrt(H*W) before matrix multiplication
    to prevent FP16 overflow when H*W is large (e.g., 56x56 = 3136).
    """
    f32 = feature_map.float()
    b, c, h, w = f32.shape
    tokens = f32.view(b, c, h * w)  # (B, C, N)
    # Channel-group pooling to keep dimensionality compact & prevent overfitting
    tokens_t = tokens.transpose(1, 2)  # (B, N, C)
    pooled_t = F.adaptive_avg_pool1d(tokens_t, pool_channels)  # (B, N, pool_channels)
    f_map = pooled_t.transpose(1, 2) / (float(h * w) ** 0.5)  # (B, pool_channels, N)

    gram = torch.bmm(f_map, f_map.transpose(1, 2))
    triu_indices = torch.triu_indices(pool_channels, pool_channels, device=feature_map.device)
    gram_vec = gram[:, triu_indices[0], triu_indices[1]]
    gram_vec = F.normalize(gram_vec, p=2, dim=1, eps=1e-6)
    return gram_vec


class GradCAMExtractor:
    """
    Computes Grad-CAM heatmaps on a target convolutional layer (e.g., ResNet50 `layer4`)
    to highlight which regions and brushwork patterns caused the DNN to predict a given artist.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Runs a forward and backward pass for a single image tensor (1, 3, H, W).
        Returns:
          - heatmap: (H_img, W_img) float32 array in [0, 1]
          - target_class: int index of the explained class
          - probs: (num_classes,) numpy array of softmax probabilities
        """
        self.model.zero_grad()
        logits, _ = self.model(input_tensor, return_features=True)
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        score.backward()

        # Global average pool the gradients across spatial dimensions (H', W')
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam[0, 0].cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class, probs

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_gradcam_on_image(
    img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """
    Blends a [0, 1] Grad-CAM heatmap (using the 'jet' / 'magma' colormap) onto the original PIL image.
    """
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
        (w, h), Image.Resampling.BICUBIC
    )
    heatmap_resized = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_resized)[:, :, :3]  # (H, W, 3) in [0, 1]

    orig_np = np.asarray(img_rgb, dtype=np.float32) / 255.0
    blended = (1.0 - alpha) * orig_np + alpha * colored_heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

