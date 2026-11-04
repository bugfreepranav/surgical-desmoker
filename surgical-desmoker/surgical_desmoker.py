"""
Reimplementation of the model in "Seeing Through Smoke: Surgical Desmoking for
Improved Visual Perception" (arXiv 2603.25867), built from the paper text.
No official code was found, so details the paper does not give are assumptions
(marked ASSUMPTION).

Pipeline:  ViT-B (MAE) -> features from blocks {3,6,9,12} -> DPT decoder
           -> desmoking head predicting O=[K,B] (4ch) with J = K*I - B + I
           and smoke map S = conv(B / (K+1)).
Loss:      L1(J, J_gt) + 0.1 * L1(S, S_gt)
Optim:     AdamW, cosine LR 1e-5 -> 1e-7, 300 epochs, 512x640 input.

Usage:
    pip install torch torchvision timm pillow numpy
    python surgical_desmoker.py test
    python surgical_desmoker.py train --clean_dir path/to/clean_frames --epochs 300
    python surgical_desmoker.py infer --ckpt desmoker.pt --image smoky.png --out clean.png
"""
import argparse
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = (512, 640)  # (H, W), both divisible by 32


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ResidualConvUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(), nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(), nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class FusionBlock(nn.Module):
    """DPT/RefineNet-style fusion. `extra_block(c)` is a hook for your own
    attention module (e.g. prior-gated / frequency-split / specular-aware)."""

    def __init__(self, c, extra_block=None):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(c)
        self.rcu_main = ResidualConvUnit(c)
        self.proj = nn.Conv2d(c, c, 1)
        self.extra = extra_block(c) if extra_block is not None else nn.Identity()

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu_skip(skip)
        x = self.rcu_main(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.extra(self.proj(x))


class DPTDecoder(nn.Module):
    def __init__(self, vit_dim=768, chans=(96, 192, 384, 768), dec_ch=256, extra_block=None):
        super().__init__()
        self.project = nn.ModuleList([nn.Conv2d(vit_dim, c, 1) for c in chans])
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(chans[0], chans[0], 4, stride=4),  # 1/16 -> 1/4
            nn.ConvTranspose2d(chans[1], chans[1], 2, stride=2),  # 1/16 -> 1/8
            nn.Identity(),                                        # 1/16
            nn.Conv2d(chans[3], chans[3], 3, stride=2, padding=1),  # 1/16 -> 1/32
        ])
        self.to_dec = nn.ModuleList([nn.Conv2d(c, dec_ch, 3, padding=1, bias=False) for c in chans])
        self.fuse = nn.ModuleList([FusionBlock(dec_ch, extra_block) for _ in range(4)])

    def forward(self, feats):
        xs = [self.to_dec[i](self.resample[i](self.project[i](f))) for i, f in enumerate(feats)]
        x = self.fuse[3](xs[3])        # 1/32 -> 1/16
        x = self.fuse[2](x, xs[2])     # -> 1/8
        x = self.fuse[1](x, xs[1])     # -> 1/4
        x = self.fuse[0](x, xs[0])     # -> 1/2
        return x


class DesmokeHead(nn.Module):
    """Predicts O=[K,B] in R^{HxWx4}; J = K*I - B + I; S = conv(B/(K+1))."""

    def __init__(self, c=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(c // 2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 4, 1),
        )
        self.smoke = nn.Conv2d(3, 1, 3, padding=1)  # f(.) is "a convolutional layer"

    def forward(self, feat, img):
        o = self.net(feat)
        K, B = o[:, :1], o[:, 1:]                     # K: 1ch, B: 3ch (ASSUMPTION)
        J = K * img - B + img
        airlight = B / (K + 1.0).clamp(min=1e-3)      # A(1-t); K+1 = 1/t > 0
        S = self.smoke(airlight)
        return J, S


class SurgicalDesmoker(nn.Module):
    def __init__(self, img_size=IMG_SIZE, pretrained=True, mae_ckpt=None,
                 extra_block=None, dec_ch=256):
        super().__init__()
        # Generic MAE weights; the paper uses MAE pretrained on surgical images.
        self.vit = timm.create_model("vit_base_patch16_224.mae", pretrained=pretrained,
                                     img_size=img_size, num_classes=0)
        if mae_ckpt:  # plug in surgical-MAE weights if you have them
            self.vit.load_state_dict(torch.load(mae_ckpt, map_location="cpu"), strict=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.blocks = [2, 5, 8, 11]                    # ViT blocks 3,6,9,12 (0-indexed)
        self.decoder = DPTDecoder(self.vit.embed_dim, dec_ch=dec_ch, extra_block=extra_block)
        self.head = DesmokeHead(dec_ch)

    def forward(self, img):
        """img: (B,3,H,W) in [0,1]. Returns J (smoke-free), S (smoke map)."""
        x = (img - self.mean) / self.std
        feats = self.vit.get_intermediate_layers(x, n=self.blocks, reshape=True)
        return self.head(self.decoder(list(feats)), img)

    @torch.no_grad()
    def predict(self, img):
        J, S = self.forward(img)
        return J.clamp(0, 1), S


# --------------------------------------------------------------------------- #
# Synthetic smoke (paper: alpha blending + RGB correction + illumination
# enhancement + domain randomization over color/transparency/blend). The exact
# smoke-pattern generator is not given, so fractal noise is used (ASSUMPTION).
# --------------------------------------------------------------------------- #
def fractal_noise(h, w, octaves=5, persistence=0.55):
    out, amp, tot = torch.zeros(1, 1, h, w), 1.0, 0.0
    for o in range(octaves):
        gh = 2 ** (o + 1)
        gw = max(2, int(gh * w / h))
        n = torch.rand(1, 1, gh, gw)
        out += amp * F.interpolate(n, size=(h, w), mode="bicubic", align_corners=False)
        tot += amp
        amp *= persistence
    out = out[0, 0] / tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def synthesize_smoke(clean):
    """clean: (3,H,W) in [0,1] -> smoky (3,H,W), smoke map (1,H,W)."""
    _, h, w = clean.shape
    n = fractal_noise(h, w, octaves=random.randint(4, 6), persistence=random.uniform(0.45, 0.65))
    thr, gamma, density = random.uniform(0.2, 0.55), random.uniform(0.7, 1.6), random.uniform(0.35, 1.0)
    alpha = (((n - thr) / (1 - thr)).clamp(0, 1) ** gamma) * density
    k = random.choice([1, 5, 9, 15])                   # soften edges
    if k > 1:
        alpha = F.avg_pool2d(alpha[None, None], k, 1, k // 2)[0, 0]
    gray = random.uniform(0.65, 0.98)                  # smoke colour / brightness
    color = (torch.tensor([gray, gray, gray]) * (1 + 0.06 * torch.randn(3))).clamp(0, 1).view(3, 1, 1)
    smoky = clean * (1 - alpha) + color * alpha
    smoky = smoky * random.uniform(1.0, 1.15)          # illumination enhancement
    return smoky.clamp(0, 1), alpha[None]


class SmokeDataset(Dataset):
    def __init__(self, clean_dir, size=IMG_SIZE):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(p for p in Path(clean_dir).rglob("*") if p.suffix.lower() in exts)
        assert self.files, f"no images in {clean_dir}"
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB").resize(self.size[::-1], Image.BICUBIC)
        clean = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        if random.random() < 0.5:
            clean = clean.flip(-1)
        smoky, smoke = synthesize_smoke(clean)
        return smoky, clean, smoke


# --------------------------------------------------------------------------- #
# Train / infer
# --------------------------------------------------------------------------- #
def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b).item()
    return 10 * np.log10(1 / max(mse, 1e-10))


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SurgicalDesmoker(pretrained=not args.no_pretrained, mae_ckpt=args.mae_ckpt).to(dev)
    dl = DataLoader(SmokeDataset(args.clean_dir), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)
    for ep in range(args.epochs):
        model.train()
        run, n = 0.0, 0
        for smoky, clean, smoke in dl:
            smoky, clean, smoke = smoky.to(dev), clean.to(dev), smoke.to(dev)
            J, S = model(smoky)
            loss = F.l1_loss(J, clean) + 0.1 * F.l1_loss(S, smoke)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item(); n += 1
        sched.step()
        print(f"epoch {ep + 1}/{args.epochs}  loss {run / n:.4f}  psnr(last batch) {psnr(J.detach(), clean):.2f}")
        torch.save(model.state_dict(), args.out)


def infer(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SurgicalDesmoker(pretrained=False).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    img = Image.open(args.image).convert("RGB")
    orig = img.size
    x = torch.from_numpy(np.asarray(img.resize(IMG_SIZE[::-1], Image.BICUBIC)).copy())
    x = x.permute(2, 0, 1).float().div(255)[None].to(dev)
    J, S = model.predict(x)
    out = (J[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(out).resize(orig, Image.BICUBIC).save(args.out)
    smoke = (S[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(smoke).resize(orig).save(Path(args.out).with_suffix(".smoke.png"))


def smoke_test():
    model = SurgicalDesmoker(pretrained=False)
    x = torch.rand(1, 3, *IMG_SIZE)
    J, S = model(x)
    print("J", tuple(J.shape), "S", tuple(S.shape),
          "params(M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
    s, m = synthesize_smoke(torch.rand(3, *IMG_SIZE))
    print("synthetic smoky", tuple(s.shape), "smoke map", tuple(m.shape))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    t = sub.add_parser("train")
    t.add_argument("--clean_dir", required=True)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--bs", type=int, default=8)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", default="desmoker.pt")
    t.add_argument("--mae_ckpt", default=None)
    t.add_argument("--no_pretrained", action="store_true")
    i = sub.add_parser("infer")
    i.add_argument("--ckpt", required=True)
    i.add_argument("--image", required=True)
    i.add_argument("--out", default="clean.png")
    a = ap.parse_args()
    {"test": lambda: smoke_test(), "train": lambda: train(a), "infer": lambda: infer(a)}[a.cmd]()
