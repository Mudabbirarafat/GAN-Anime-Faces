"""
gan_anime_faces.py
==================
Anime Face Generation using Deep Convolutional GAN (DCGAN)
Dataset: https://www.kaggle.com/splcher/animefacedataset

Architecture: DCGAN with spectral normalization on the Discriminator
              and progressive upsampling in the Generator.

Requirements:
    pip install torch torchvision numpy matplotlib tqdm Pillow

Usage:
    python gan_anime_faces.py --data_dir ./data/anime --epochs 100 --batch_size 64
"""

import os
import argparse
import random
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from torchvision.models import inception_v3

# ──────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCGAN — Anime Face Generation")

    # Data
    parser.add_argument("--data_dir",    type=str,   default="./data/anime",
                        help="Root folder containing anime face images")
    parser.add_argument("--image_size",  type=int,   default=64,
                        help="Spatial resolution to resize images to (default: 64)")
    parser.add_argument("--workers",     type=int,   default=4,
                        help="DataLoader worker count")

    # Training
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=2e-4,
                        help="Learning rate (Adam, same for G and D)")
    parser.add_argument("--beta1",       type=float, default=0.5,
                        help="Adam β₁")
    parser.add_argument("--beta2",       type=float, default=0.999,
                        help="Adam β₂")

    # Architecture
    parser.add_argument("--latent_dim",  type=int,   default=128,
                        help="Length of the latent noise vector")
    parser.add_argument("--ngf",         type=int,   default=64,
                        help="Generator feature-map multiplier")
    parser.add_argument("--ndf",         type=int,   default=64,
                        help="Discriminator feature-map multiplier")
    parser.add_argument("--n_channels",  type=int,   default=3,
                        help="Number of image channels (3 = RGB)")

    # Label smoothing / noise
    parser.add_argument("--label_smooth", type=float, default=0.1,
                        help="One-sided label smoothing for real labels")
    parser.add_argument("--instance_noise_std", type=float, default=0.05,
                        help="Std of Gaussian noise added to D inputs (annealed to 0)")

    # Logging / saving
    parser.add_argument("--save_dir",    type=str,   default="./outputs")
    parser.add_argument("--log_interval",type=int,   default=100,
                        help="Print losses every N batches")
    parser.add_argument("--save_interval",type=int,  default=10,
                        help="Save checkpoint and sample grid every N epochs")
    parser.add_argument("--n_samples",   type=int,   default=64,
                        help="Fixed noise samples visualised during training")
    parser.add_argument("--seed",        type=int,   default=42)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloader(data_dir: str, image_size: int, batch_size: int,
                     workers: int) -> DataLoader:
    """
    Expects data_dir to contain images directly OR a single sub-folder
    (ImageFolder format, e.g. data_dir/anime/*.jpg).

    Preprocessing:
        - Resize shortest edge to image_size, then centre-crop
        - Random horizontal flip (p=0.5)  — augmentation
        - Normalize to [-1, 1]  (matches Generator Tanh output)
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    # Support both flat directories and ImageFolder-style layouts
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir}'. "
            "Download from https://www.kaggle.com/splcher/animefacedataset "
            "and unzip so that images are inside data_dir/ or data_dir/<class>/."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
        drop_last=True,      # Keep batch size constant throughout
    )
    print(f"[Data] {len(dataset):,} images | {len(loader):,} batches/epoch | "
          f"batch_size={batch_size} | image_size={image_size}×{image_size}")
    return loader


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────

def weights_init(m: nn.Module) -> None:
    """
    Custom weight initialisation from the DCGAN paper:
        Conv / ConvTranspose  →  N(0, 0.02)
        BatchNorm γ           →  N(1, 0.02), β = 0
    """
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    DCGAN Generator
    ───────────────
    Input  : z ~ N(0,1) of shape (B, latent_dim)
    Output : image of shape (B, 3, image_size, image_size) in [-1, 1]

    Architecture (for image_size=64, ngf=64):
        Linear  →  4×4×(ngf*8)
        ConvT   →  8×8×(ngf*4)
        ConvT   →  16×16×(ngf*2)
        ConvT   →  32×32×(ngf)
        ConvT   →  64×64×3          + Tanh
    Each ConvTranspose block uses BatchNorm + ReLU.
    """

    def __init__(self, latent_dim: int, ngf: int, n_channels: int,
                 image_size: int) -> None:
        super().__init__()
        self.latent_dim  = latent_dim
        self.image_size  = image_size

        # Compute number of upsampling stages dynamically
        # image_size must be a power of 2, minimum 8
        assert image_size >= 8 and (image_size & (image_size - 1)) == 0, \
            "image_size must be a power of 2 (e.g. 32, 64, 128)"
        self.n_stages = int(math.log2(image_size)) - 2   # e.g. log2(64)-2 = 4

        # Project and reshape noise to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ngf * (2 ** (self.n_stages - 1)) * 4 * 4),
        )

        # Build upsampling blocks
        layers = []
        in_ch = ngf * (2 ** (self.n_stages - 1))
        for i in range(self.n_stages - 1):
            out_ch = in_ch // 2
            layers += self._up_block(in_ch, out_ch)
            in_ch = out_ch

        # Final layer → RGB, no BN
        layers += [
            nn.ConvTranspose2d(in_ch, n_channels, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh(),
        ]
        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    @staticmethod
    def _up_block(in_ch: int, out_ch: int):
        return [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        h = self.fc(z)
        in_ch = h.size(1) // 16          # 4*4 = 16 spatial positions
        h = h.view(h.size(0), in_ch, 4, 4)
        return self.main(h)


class Discriminator(nn.Module):
    def __init__(self, ndf, n_channels, image_size):
        super().__init__()
        import math
        n_down = int(math.log2(image_size)) - 2
        layers = [
            nn.utils.spectral_norm(nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True),
        ]
        in_ch = ndf
        for _ in range(n_down - 1):
            out_ch = min(in_ch * 2, ndf * 8)
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True),
            ]
            in_ch = out_ch
        self.features    = nn.Sequential(*layers)
        self.pool        = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.utils.spectral_norm(nn.Linear(in_ch, 1, bias=False))
        nn.init.normal_(self.classifier.weight.data, 0.0, 0.02)

    def forward(self, x):
        h = self.pool(self.features(x))
        h = h.view(h.size(0), -1)
        return self.classifier(h).squeeze(1)


class GANTrainer:
    """
    Encapsulates the GAN training loop.

    Training strategy
    ─────────────────
    • Loss           : Binary Cross-Entropy with logits (non-saturating variant)
    • D step         : maximise  log D(x) + log(1 − D(G(z)))
    • G step         : maximise  log D(G(z))  (non-saturating)
    • Label smoothing: real labels drawn from U[1−ε, 1]  (one-sided)
    • Instance noise : Gaussian noise added to D inputs, std annealed to 0
                       over the first half of training (Sønderby et al., 2017)
    """

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args   = args
        self.device = device

        # Models
        self.G = Generator(args.latent_dim, args.ngf,
                           args.n_channels, args.image_size).to(device)
        self.D = Discriminator(args.ndf, args.n_channels,
                               args.image_size).to(device)

        print(f"\n[Generator]      params: {self._count_params(self.G):,}")
        print(f"[Discriminator]  params: {self._count_params(self.D):,}\n")

        # Optimisers
        opt_kwargs = dict(lr=args.lr, betas=(args.beta1, args.beta2))
        self.opt_G = optim.Adam(self.G.parameters(), **opt_kwargs)
        self.opt_D = optim.Adam(self.D.parameters(), **opt_kwargs)

        self.criterion = nn.BCEWithLogitsLoss()

        # Fixed noise for visualisation (kept constant across epochs)
        self.fixed_z = torch.randn(args.n_samples, args.latent_dim).to(device)

        # History
        self.history = {"G_loss": [], "D_loss": [], "D_real": [], "D_fake": []}

        # Paths
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "samples"),  exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "checkpoints"), exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _noise_std(self, epoch: int) -> float:
        """Linearly anneal instance noise to 0 over the first half of training."""
        halfway = self.args.epochs / 2
        return self.args.instance_noise_std * max(0.0, 1.0 - epoch / halfway)

    def _sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.args.latent_dim, device=self.device)

    # ── single training step ─────────────────────────────────────────────────

    def _step(self, real_imgs: torch.Tensor, epoch: int):
        B     = real_imgs.size(0)
        noise = self._noise_std(epoch)

        # ── Discriminator step ────────────────────────────────────────────
        self.opt_D.zero_grad()

        # Real images (with label smoothing and instance noise)
        real_with_noise = real_imgs + noise * torch.randn_like(real_imgs)
        real_labels = torch.empty(B, device=self.device).uniform_(
            1.0 - self.args.label_smooth, 1.0)
        logit_real = self.D(real_with_noise)
        loss_D_real = self.criterion(logit_real, real_labels)

        # Fake images
        z = self._sample_z(B)
        fake_imgs = self.G(z).detach()           # detach: no G gradient
        fake_with_noise = fake_imgs + noise * torch.randn_like(fake_imgs)
        fake_labels = torch.zeros(B, device=self.device)
        logit_fake  = self.D(fake_with_noise)
        loss_D_fake = self.criterion(logit_fake, fake_labels)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        self.opt_D.step()

        # ── Generator step ────────────────────────────────────────────────
        self.opt_G.zero_grad()

        z = self._sample_z(B)
        gen_imgs    = self.G(z)
        logit_gen   = self.D(gen_imgs)
        real_labels_G = torch.ones(B, device=self.device)   # G wants D to predict "real"
        loss_G = self.criterion(logit_gen, real_labels_G)

        loss_G.backward()
        self.opt_G.step()

        return {
            "G": loss_G.item(),
            "D": loss_D.item(),
            "D_real": torch.sigmoid(logit_real).mean().item(),
            "D_fake": torch.sigmoid(logit_fake).mean().item(),
        }

    # ── training loop ────────────────────────────────────────────────────────

    def train(self, loader: DataLoader) -> None:
        total_batches = len(loader)
        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()
            g_losses, d_losses, d_reals, d_fakes = [], [], [], []

            self.G.train()
            self.D.train()

            for i, (imgs, _) in enumerate(loader, 1):
                imgs = imgs.to(self.device)
                stats = self._step(imgs, epoch)

                g_losses.append(stats["G"])
                d_losses.append(stats["D"])
                d_reals.append(stats["D_real"])
                d_fakes.append(stats["D_fake"])

                if i % self.args.log_interval == 0:
                    print(
                        f"  [Epoch {epoch:>3}/{self.args.epochs}]"
                        f"  Batch {i:>4}/{total_batches}"
                        f"  |  G={stats['G']:.4f}"
                        f"  D={stats['D']:.4f}"
                        f"  D(x)={stats['D_real']:.3f}"
                        f"  D(G(z))={stats['D_fake']:.3f}"
                    )

            # Record epoch averages
            self.history["G_loss"].append(np.mean(g_losses))
            self.history["D_loss"].append(np.mean(d_losses))
            self.history["D_real"].append(np.mean(d_reals))
            self.history["D_fake"].append(np.mean(d_fakes))

            elapsed = time.time() - epoch_start
            print(
                f"[Epoch {epoch:>3}/{self.args.epochs}]"
                f"  G={self.history['G_loss'][-1]:.4f}"
                f"  D={self.history['D_loss'][-1]:.4f}"
                f"  time={elapsed:.1f}s"
            )

            if epoch % self.args.save_interval == 0 or epoch == self.args.epochs:
                self._save_samples(epoch)
                self._save_checkpoint(epoch)
                self._plot_losses()

        print("\n✓ Training complete.")

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def _save_samples(self, epoch: int) -> None:
        """Save a grid of images generated from the fixed noise vector."""
        self.G.eval()
        with torch.no_grad():
            fake = self.G(self.fixed_z).cpu()
        grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
        path = os.path.join(self.args.save_dir, "samples", f"epoch_{epoch:04d}.png")
        vutils.save_image(grid, path)
        print(f"  → Sample grid saved: {path}")

    def _save_checkpoint(self, epoch: int) -> None:
        """Persist model weights and optimiser states."""
        ckpt = {
            "epoch": epoch,
            "G_state":     self.G.state_dict(),
            "D_state":     self.D.state_dict(),
            "opt_G_state": self.opt_G.state_dict(),
            "opt_D_state": self.opt_D.state_dict(),
            "history":     self.history,
            "args":        vars(self.args),
        }
        path = os.path.join(self.args.save_dir, "checkpoints", f"ckpt_epoch_{epoch:04d}.pt")
        torch.save(ckpt, path)
        print(f"  → Checkpoint saved: {path}")

    def _plot_losses(self) -> None:
        """Plot and save G/D loss curves and D output averages."""
        epochs = range(1, len(self.history["G_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, self.history["G_loss"], label="G Loss", color="#e07b39")
        axes[0].plot(epochs, self.history["D_loss"], label="D Loss", color="#3976e0")
        axes[0].set_title("Generator & Discriminator Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, self.history["D_real"], label="D(x) — Real",  color="#2ca02c")
        axes[1].plot(epochs, self.history["D_fake"], label="D(G(z)) — Fake", color="#d62728")
        axes[1].axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
        axes[1].set_title("Discriminator Output Averages")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Sigmoid(logit)")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.args.save_dir, "loss_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION — FRÉCHET INCEPTION DISTANCE (FID)
# ──────────────────────────────────────────────────────────────────────────────

def _inception_features(images: torch.Tensor,
                         model: nn.Module,
                         device: torch.device) -> np.ndarray:
    """
    Extract 2048-d pool features from InceptionV3 for a batch of images.
    images should be normalised to [0,1] and resized to 299×299.
    """
    model.eval()
    images = F.interpolate(images, size=(299, 299), mode="bilinear",
                           align_corners=False)
    # InceptionV3 expects [0,1] — our images are in [-1,1]
    images = (images + 1) / 2.0
    with torch.no_grad():
        feat = model(images.to(device))
    return feat.cpu().numpy()


def compute_fid(real_loader: DataLoader, generator: nn.Module,
                device: torch.device, n_samples: int = 5000) -> float:
    """
    Compute FID between real images and GAN-generated images.

    FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)

    Reference: Heusel et al., 2017 — "GANs Trained by a Two Time-Scale
               Update Rule Converge to a Local Nash Equilibrium"
    """
    from scipy.linalg import sqrtm

    print("\n[Evaluation] Computing FID ...")
    incep = inception_v3(pretrained=True, transform_input=False)
    # Remove the final FC and AuxLogits — keep only pool features
    incep.fc       = nn.Identity()
    incep.AuxLogits = None
    incep = incep.to(device)
    incep.eval()

    latent_dim = generator.latent_dim

    real_feats, fake_feats = [], []

    for imgs, _ in real_loader:
        real_feats.append(_inception_features(imgs, incep, device))
        if len(real_feats) * imgs.size(0) >= n_samples:
            break

    generator.eval()
    collected = 0
    while collected < n_samples:
        batch = min(64, n_samples - collected)
        z = torch.randn(batch, latent_dim, device=device)
        with torch.no_grad():
            gen = generator(z).cpu()
        fake_feats.append(_inception_features(gen, incep, device))
        collected += batch

    mu_r, sigma_r = np.mean(np.concatenate(real_feats, axis=0), axis=0), \
                    np.cov(np.concatenate(real_feats, axis=0), rowvar=False)
    mu_g, sigma_g = np.mean(np.concatenate(fake_feats, axis=0), axis=0), \
                    np.cov(np.concatenate(fake_feats, axis=0), rowvar=False)

    diff  = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real       # Numerical stabilisation

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    print(f"[Evaluation] FID = {fid:.2f}  (lower is better, ~30 is decent for DCGAN)")
    return fid


# ──────────────────────────────────────────────────────────────────────────────
# 7. INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str, args: argparse.Namespace,
                    device: torch.device) -> tuple:
    """Load a saved checkpoint and reconstruct G & D."""
    ckpt = torch.load(path, map_location=device)
    G = Generator(args.latent_dim, args.ngf, args.n_channels, args.image_size).to(device)
    D = Discriminator(args.ndf, args.n_channels, args.image_size).to(device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} from {path}")
    return G, D, ckpt["epoch"]


def generate_images(generator: nn.Module, n: int,
                    latent_dim: int, device: torch.device,
                    out_path: str = "generated.png") -> None:
    """Generate n images from a trained generator and save as a grid."""
    generator.eval()
    z = torch.randn(n, latent_dim, device=device)
    with torch.no_grad():
        imgs = generator(z).cpu()
    nrow = int(math.ceil(math.sqrt(n)))
    grid = vutils.make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, out_path)
    print(f"[Inference] {n} images saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[Device] Using: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    loader = build_dataloader(args.data_dir, args.image_size,
                              args.batch_size, args.workers)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = GANTrainer(args, device)
    trainer.train(loader)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    try:
        fid = compute_fid(loader, trainer.G, device, n_samples=5000)
        with open(os.path.join(args.save_dir, "fid_score.txt"), "w") as f:
            f.write(f"FID: {fid:.4f}\n")
    except Exception as e:
        print(f"[Evaluation] FID skipped: {e}")

    # ── Final sample ─────────────────────────────────────────────────────────
    generate_images(trainer.G, n=64, latent_dim=args.latent_dim,
                    device=device,
                    out_path=os.path.join(args.save_dir, "final_generated.png"))


if __name__ == "__main__":
    main()
