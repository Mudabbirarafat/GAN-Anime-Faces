# Anime Face Generation with GANs

A PyTorch implementation of a Deep Convolutional GAN (DCGAN) with spectral normalization for generating anime faces, trained on the [Anime Face Dataset](https://www.kaggle.com/splcher/animefacedataset).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Setup](#dataset-setup)
3. [GAN Architecture](#gan-architecture)
4. [Training Process](#training-process)
5. [Evaluation](#evaluation)
6. [Results & Analysis](#results--analysis)
7. [Hyperparameter Guide](#hyperparameter-guide)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision numpy matplotlib tqdm Pillow scipy
```

> Tested with Python 3.9+, PyTorch 2.x. CUDA is strongly recommended (training on CPU is very slow).

### 2. Download the dataset

**Via Kaggle CLI:**
```bash
pip install kaggle
kaggle datasets download -d splcher/animefacedataset -p ./data
unzip ./data/animefacedataset.zip -d ./data/anime
```

**Manually:** Download from [Kaggle](https://www.kaggle.com/splcher/animefacedataset) and unzip so images live at `./data/anime/images/*.jpg`.

The script uses PyTorch's `ImageFolder` loader, so images should be inside **one sub-folder**:

```
data/
└── anime/
    └── images/        ← all .jpg files go here
        ├── 0001.jpg
        ├── 0002.jpg
        └── ...
```

### 3. Train

```bash
# Default run (64×64 images, 100 epochs)
python gan_anime_faces.py --data_dir ./data/anime --epochs 100

# Higher resolution (128×128) — needs more VRAM
python gan_anime_faces.py --data_dir ./data/anime --image_size 128 --ngf 128 --ndf 128 --epochs 150
```

### 4. Generate images from a checkpoint

```python
from gan_anime_faces import load_checkpoint, generate_images, parse_args
import torch, argparse

args = parse_args()                              # use same flags as training
device = torch.device("cuda")
G, D, epoch = load_checkpoint("outputs/checkpoints/ckpt_epoch_0100.pt", args, device)
generate_images(G, n=64, latent_dim=args.latent_dim, device=device,
                out_path="my_anime_faces.png")
```

---

## Dataset Setup

| Property | Details |
|----------|---------|
| Source   | Kaggle — `splcher/animefacedataset` |
| Size     | ~63,000 anime face images |
| Format   | JPEG, variable resolution |
| License  | Community Data License Agreement |

**Preprocessing applied at runtime:**

| Step | Details |
|------|---------|
| Resize | Shortest edge → `image_size` (default 64) |
| CenterCrop | Square crop to `image_size × image_size` |
| RandomHorizontalFlip | p = 0.5 (data augmentation) |
| Normalize | `(pixel − 0.5) / 0.5` → range [−1, 1] |

---

## GAN Architecture

The model follows the **DCGAN** paper (Radford et al., 2015) with two additions: **spectral normalization** on the Discriminator and **one-sided label smoothing** during training.

### Generator

The Generator maps a random latent vector **z ∈ ℝ¹²⁸** to a full RGB image via a series of transposed convolution (fractionally strided convolution) blocks.

```
Input:  z ~ N(0, I)  shape (B, 128)
        ↓ Linear → reshape (B, 512, 4, 4)
        ↓ ConvTranspose2d  4→8    512→256  BN + ReLU
        ↓ ConvTranspose2d  8→16   256→128  BN + ReLU
        ↓ ConvTranspose2d  16→32  128→64   BN + ReLU
        ↓ ConvTranspose2d  32→64  64→3     Tanh
Output: image  shape (B, 3, 64, 64)  ∈ [−1, 1]
```

**Design choices:**
- `Tanh` output matches the normalised data range [−1, 1].
- No Dropout — BatchNorm provides sufficient regularisation.
- Weights initialised from `N(0, 0.02)` per the DCGAN paper.
- Number of stages scales automatically with `image_size` (must be a power of 2).

### Discriminator

The Discriminator is a mirrored convolutional encoder that outputs an unnormalised logit (no Sigmoid — we use `BCEWithLogitsLoss` for numerical stability).

```
Input:  image  shape (B, 3, 64, 64)
        ↓ SN-Conv2d  64→32   64ch   LeakyReLU(0.2)  [no BN on first layer]
        ↓ SN-Conv2d  32→16   128ch  BN + LeakyReLU(0.2)
        ↓ SN-Conv2d  16→8    256ch  BN + LeakyReLU(0.2)
        ↓ SN-Conv2d  8→4     512ch  BN + LeakyReLU(0.2)
        ↓ SN-Conv2d  4→1     1ch    (logit)
Output: scalar logit  shape (B,)
```

**Design choices:**
- **Spectral Normalization** (Miyato et al., 2018) on all Conv layers — constrains the Lipschitz constant, stabilising training without hurting gradient flow.
- `LeakyReLU(0.2)` — prevents dying neurons (unlike ReLU).
- No Sigmoid — raw logits fed directly to `BCEWithLogitsLoss`.
- No BatchNorm on first layer — standard practice from the DCGAN paper.

---

## Training Process

### Loss Functions

| Network | Objective | Formula |
|---------|-----------|---------|
| Discriminator | Maximise ability to separate real from fake | `L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)` |
| Generator | Fool the Discriminator (non-saturating) | `L_G = BCE(D(G(z)), 1)` |

Using `BCEWithLogitsLoss` combines the sigmoid and BCE in one numerically stable operation.

### Stabilisation Techniques

| Technique | Why |
|-----------|-----|
| **One-sided label smoothing** | Real labels drawn from U[1−ε, 1] instead of hard 1. Prevents overconfident Discriminator; reduces mode collapse. |
| **Instance noise** | Gaussian noise added to both real and fake images before D sees them. Std annealed to 0 over the first 50% of training. Smooths the data manifold. |
| **Spectral normalization** | Constrains Lipschitz constant of D, prevents gradient explosion. |
| **Separate Adam optimisers** | β₁ = 0.5 (not 0.9) per DCGAN recommendation — lower momentum for non-stationary GAN loss landscape. |

### Training Loop (per batch)

```
1. Sample real batch x from DataLoader
2. Add instance noise to x  →  x̃
3. Forward x̃ through D  →  logit_real
4. Compute loss_D_real = BCE(logit_real, smooth_labels)

5. Sample z ~ N(0,I)
6. Generate fake = G(z).detach()   # stop gradient through G
7. Add instance noise to fake  →  f̃
8. Forward f̃ through D  →  logit_fake
9. Compute loss_D_fake = BCE(logit_fake, 0)

10. loss_D = loss_D_real + loss_D_fake
11. Backprop, step opt_D

12. Sample fresh z ~ N(0,I)
13. Generate gen = G(z)            # no detach this time
14. Forward gen through D  →  logit_gen
15. loss_G = BCE(logit_gen, 1)     # G wants D to say "real"
16. Backprop, step opt_G
```

### Saved Outputs

| File | Description |
|------|-------------|
| `outputs/samples/epoch_XXXX.png` | 8×8 grid of generated images (fixed noise) |
| `outputs/checkpoints/ckpt_epoch_XXXX.pt` | Full checkpoint: weights + optimiser states + history |
| `outputs/loss_curves.png` | G/D loss and D output plots |
| `outputs/final_generated.png` | Final 8×8 generation grid |
| `outputs/fid_score.txt` | FID score computed after training |

---

## Evaluation

### Fréchet Inception Distance (FID)

FID measures the similarity between the distribution of real images and generated images, using features extracted by a pre-trained **InceptionV3** network.

**Formula:**

```
FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2·sqrt(Σ_r · Σ_g))
```

where (μ_r, Σ_r) and (μ_g, Σ_g) are the mean and covariance of the InceptionV3 pool features for real and generated images respectively.

**Interpretation:**

| FID Range | Quality |
|-----------|---------|
| < 10 | Excellent (close to state-of-the-art) |
| 10–30 | Good — visually plausible faces |
| 30–60 | Decent — typical for DCGAN at 64×64 |
| > 100 | Poor — significant artefacts or mode collapse |

FID is computed on 5,000 samples by default. It requires `scipy` and an internet connection (for downloading InceptionV3 weights on first run).

### Visual Inspection

Beyond FID, qualitative inspection of the epoch sample grids reveals:

- **Early epochs (1–20):** Noisy, blob-like shapes
- **Mid training (20–60):** Colour and rough facial structure emerge
- **Late training (60–100):** Defined eyes, hair, and facial features

---

## Results & Analysis

### Expected Training Behaviour

A well-behaving DCGAN run exhibits:

- **D_loss** converges to ~1.4 (ln 2 + ln 2 ≈ 1.386 — perfect 50/50 balance)
- **G_loss** stays roughly in the 1–3 range
- **D(x)** (real image score) starts near 1 and settles around 0.5–0.7
- **D(G(z))** (fake image score) starts near 0 and rises toward 0.4–0.5
- Loss values never diverge to infinity

### Typical Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| G_loss → ∞ | D is too strong — G gets no gradient | Reduce ndf, add more G steps, increase label smoothing |
| D_loss ≈ 0 | Mode collapse — D has memorised fakes | Reduce lr_D, add spectral norm, check data balance |
| Images are uniform colour / single pattern | Severe mode collapse | Restart with lower lr, increase latent_dim |
| Checkerboard artefacts | Transposed convolution stride misalignment | Replace ConvTranspose with Upsample + Conv |

### Improvements Beyond This Baseline

- **Progressive Growing GAN (ProGAN)** — grow resolution during training for higher quality
- **StyleGAN / StyleGAN2** — inject noise at each resolution; state-of-the-art for face synthesis
- **Consistency Regularization (CR-GAN)** — penalise D's sensitivity to image augmentation
- **Differentiable Augmentation (DiffAugment)** — augment both real and fake images fed to D
- **Wasserstein GAN with gradient penalty (WGAN-GP)** — more stable loss with theoretical grounding

---

## Hyperparameter Guide

| Argument | Default | Effect |
|----------|---------|--------|
| `--latent_dim` | 128 | Larger → more expressive G; diminishing returns above 256 |
| `--ngf` / `--ndf` | 64 | Larger → more capacity; doubles VRAM per doubling |
| `--lr` | 2e-4 | Standard DCGAN value; lower if training is unstable |
| `--batch_size` | 64 | 64–128 typical; larger needs more VRAM |
| `--image_size` | 64 | Must be a power of 2; 128 needs ~4× VRAM of 64 |
| `--label_smooth` | 0.1 | 0.05–0.15 works well; 0 = hard labels |
| `--instance_noise_std` | 0.05 | Higher → more regularisation; 0 to disable |
| `--epochs` | 100 | 50 epochs for quick test; 200+ for best quality |

---

## Project Structure

```
.
├── gan_anime_faces.py      # Complete GAN pipeline
├── README.md               # This file
├── data/
│   └── anime/
│       └── images/         # Anime face JPEG images
└── outputs/
    ├── samples/            # Generated image grids per epoch
    ├── checkpoints/        # Model checkpoints
    ├── loss_curves.png     # Training loss plot
    ├── final_generated.png # Final generated faces
    └── fid_score.txt       # FID evaluation result
```

---

## Troubleshooting

**`FileNotFoundError` on dataset:**
Ensure images are inside `data/anime/images/` (one sub-folder required for `ImageFolder`).

**CUDA out of memory:**
Reduce `--batch_size` to 32 or `--image_size` to 32, or reduce `--ngf`/`--ndf`.

**FID computation hangs:**
First run downloads InceptionV3 (~100 MB). Check your internet connection or pass `n_samples=1000` for a faster estimate.

**Training diverges after many epochs:**
Try reducing `--lr` to `1e-4` or enabling cosine LR decay (not built-in — add to `GANTrainer`).

---

## References

1. Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
2. Miyato, T., et al. (2018). *Spectral Normalization for Generative Adversarial Networks.* [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
3. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
4. Sønderby, C. K., et al. (2017). *Amortised MAP Inference for Image Super-resolution.* (Instance noise trick) [arXiv:1610.09461](https://arxiv.org/abs/1610.09461)
