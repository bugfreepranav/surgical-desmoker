import os

project_files = {
    "option.py": '''import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Surgical Desmoker Training and Evaluation")
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--dec_ch", type=int, default=256)
    parser.add_argument("--mae_ckpt", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--clean_dir", type=str, default="./data/clean_frames")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./trained_models")
    parser.add_argument("--ckpt", type=str, default="./trained_models/latest.pt")
    parser.add_argument("--test_image", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./results")
    return parser
''',

    "models/__init__.py": "from .desmoker import SurgicalDesmoker\nfrom .losses import TotalDesmokingLoss\n",

    "models/desmoker.py": '''import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ResidualConvUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

class FusionBlock(nn.Module):
    def __init__(self, c, extra_block=None):
        super().__init__()
        self.rcu_skip = ResidualConvUnit(c)
        self.rcu_main = ResidualConvUnit(c)
        self.proj = nn.Conv2d(c, c, kernel_size=1)
        self.extra = extra_block(c) if extra_block is not None else nn.Identity()

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.rcu_skip(skip)
        x = self.rcu_main(x)
        return self.extra(self.proj(x))

class DPTDecoder(nn.Module):
    def __init__(self, vit_dim=768, chans=(96, 192, 384, 768), dec_ch=256, extra_block=None):
        super().__init__()
        self.project = nn.ModuleList([nn.Conv2d(vit_dim, c, kernel_size=1) for c in chans])
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(chans[0], chans[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(chans[1], chans[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(chans[3], chans[3], kernel_size=3, stride=2, padding=1)
        ])
        self.to_dec = nn.ModuleList([nn.Conv2d(c, dec_ch, kernel_size=3, padding=1, bias=False) for c in chans])
        self.fuse = nn.ModuleList([FusionBlock(dec_ch, extra_block) for _ in range(4)])

    def forward(self, feats):
        xs = [self.to_dec[i](self.resample[i](self.project[i](f))) for i, f in enumerate(feats)]
        x = self.fuse[3](xs[3])
        x = self.fuse[2](x, skip=xs[2])
        x = self.fuse[1](x, skip=xs[1])
        x = self.fuse[0](x, skip=xs[0])
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

class DesmokeHead(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(c // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        self.smoke = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feat, img):
        o = self.net(feat)
        K, B = o[:, :1], o[:, 1:]
        J = K * img - B + img
        denom = F.softplus(K + 1.0) + 1e-4
        airlight = B / denom
        S = self.smoke(airlight)
        return J, S

class SurgicalDesmoker(nn.Module):
    def __init__(self, img_size=(512, 640), pretrained=True, mae_ckpt=None, extra_block=None, dec_ch=256):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224.mae",
            pretrained=pretrained,
            img_size=img_size,
            dynamic_img_size=True,
            num_classes=0
        )
        if mae_ckpt:
            ckpt = torch.load(mae_ckpt, map_location="cpu")
            self.vit.load_state_dict(ckpt.get("model", ckpt), strict=False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.blocks = [2, 5, 8, 11]
        self.decoder = DPTDecoder(self.vit.embed_dim, dec_ch=dec_ch, extra_block=extra_block)
        self.head = DesmokeHead(dec_ch)

    def forward(self, img):
        x = (img - self.mean) / self.std
        feats = self.vit.get_intermediate_layers(x, n=self.blocks, reshape=True)
        decoded = self.decoder(list(feats))
        return self.head(decoded, img)

    @torch.no_grad()
    def predict(self, img):
        J, S = self.forward(img)
        return J.clamp(0.0, 1.0), S.clamp(0.0, 1.0)
''',

    "models/losses.py": '''import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_norm), self.vgg(target_norm))

class TotalDesmokingLoss(nn.Module):
    def __init__(self, lambda_smoke=0.1, lambda_perc=0.04):
        super().__init__()
        self.lambda_smoke = lambda_smoke
        self.lambda_perc = lambda_perc
        self.perceptual = PerceptualLoss()

    def forward(self, J_pred, J_gt, S_pred, S_gt):
        l1_img = F.l1_loss(J_pred, J_gt)
        l1_smoke = F.l1_loss(S_pred, S_gt)
        perc = self.perceptual(J_pred, J_gt)
        return l1_img + (self.lambda_smoke * l1_smoke) + (self.lambda_perc * perc)
''',

    "data/__init__.py": "from .dataset import CleanFrameDataset\nfrom .synthesizer import GPUSmokeSynthesizer\n",

    "data/synthesizer.py": '''import torch
import torch.nn as nn
import torch.nn.functional as F

class GPUSmokeSynthesizer(nn.Module):
    def __init__(self, octaves=5, persistence=0.55):
        super().__init__()
        self.octaves = octaves
        self.persistence = persistence

    @torch.no_grad()
    def forward(self, clean):
        b, _, h, w = clean.shape
        device, dtype = clean.device, clean.dtype

        out = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
        amp, tot = 1.0, 0.0
        for o in range(self.octaves):
            gh = max(2, int(h / (2 ** (self.octaves - o - 1))))
            gw = max(2, int(w / (2 ** (self.octaves - o - 1))))
            n = torch.rand(b, 1, gh, gw, device=device, dtype=dtype)
            out += amp * F.interpolate(n, size=(h, w), mode="bilinear", align_corners=False)
            tot += amp
            amp *= self.persistence

        out = out / max(tot, 1e-6)
        min_v = out.amin(dim=(-2, -1), keepdim=True)
        max_v = out.amax(dim=(-2, -1), keepdim=True)
        noise = (out - min_v) / (max_v - min_v + 1e-6)

        thr = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.35 + 0.20
        gamma = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.90 + 0.70
        density = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.65 + 0.35

        alpha = (((noise - thr) / (1.0 - thr).clamp(min=1e-4)).clamp(0.0, 1.0) ** gamma) * density
        alpha = F.avg_pool2d(alpha, kernel_size=9, stride=1, padding=4)

        gray = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * 0.33 + 0.65
        color_offset = 1.0 + 0.05 * torch.randn(b, 3, 1, 1, device=device, dtype=dtype)
        smoke_color = (gray.expand(-1, 3, -1, -1) * color_offset).clamp(0.0, 1.0)

        smoky = (clean * (1.0 - alpha) + smoke_color * alpha).clamp(0.0, 1.0)
        return smoky, alpha
''',

    "data/dataset.py": '''from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class CleanFrameDataset(Dataset):
    def __init__(self, clean_dir, size=(512, 640)):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(p for p in Path(clean_dir).rglob("*") if p.suffix.lower() in exts)
        assert len(self.files) > 0, f"No clean images in {clean_dir}"
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as img:
            img = img.convert("RGB").resize(self.size[::-1], Image.Resampling.BILINEAR)
            clean = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255.0)

        if random.random() < 0.5:
            clean = clean.flip(-1)
        return clean
''',

    "utils/__init__.py": "from .metrics import calc_psnr\n",

    "utils/metrics.py": '''import math
import torch.nn.functional as F

def calc_psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0)).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))
''',

    "main.py": '''import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from option import get_parser
from models.desmoker import SurgicalDesmoker
from models.losses import TotalDesmokingLoss
from data.dataset import CleanFrameDataset
from data.synthesizer import GPUSmokeSynthesizer
from utils.metrics import calc_psnr

def main():
    opt = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.save_dir, exist_ok=True)

    model = SurgicalDesmoker(
        img_size=(opt.img_h, opt.img_w),
        pretrained=not opt.no_pretrained,
        mae_ckpt=opt.mae_ckpt,
        dec_ch=opt.dec_ch
    ).to(device)

    smoke_synth = GPUSmokeSynthesizer().to(device)
    criterion = TotalDesmokingLoss().to(device)

    dataset = CleanFrameDataset(opt.clean_dir, size=(opt.img_h, opt.img_w))
    dl = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    for ep in range(opt.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        
        for clean in dl:
            clean = clean.to(device, non_blocking=True)
            smoky, smoke_gt = smoke_synth(clean)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                J, S = model(smoky)
                loss = criterion(J, clean, S, smoke_gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

        scheduler.step()
        print(f"Epoch [{ep+1}/{opt.epochs}] - Loss: {total_loss/steps:.4f} - PSNR: {calc_psnr(J.detach(), clean):.2f} dB")
        torch.save(model.state_dict(), Path(opt.save_dir) / "latest.pt")

if __name__ == "__main__":
    main()
''',

    "test.py": '''from pathlib import Path
import numpy as np
import torch
from PIL import Image

from option import get_parser
from models.desmoker import SurgicalDesmoker

def run_test():
    opt = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)

    model = SurgicalDesmoker(img_size=(opt.img_h, opt.img_w), pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(opt.ckpt, map_location=device))

    with Image.open(opt.test_image) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        resized = img.resize((opt.img_w, opt.img_h), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(resized)).permute(2, 0, 1).float().div(255.0)[None].to(device)

    with torch.no_grad():
        J, S = model.predict(x)

    out_np = (J[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    smoke_np = (S[0, 0].cpu().numpy() * 255.0).astype(np.uint8)

    stem = Path(opt.test_image).stem
    Image.fromarray(out_np).resize((orig_w, orig_h), Image.Resampling.BILINEAR).save(f"{opt.out_dir}/{stem}_clean.png")
    Image.fromarray(smoke_np).resize((orig_w, orig_h), Image.Resampling.BILINEAR).save(f"{opt.out_dir}/{stem}_smoke.png")
    print(f"Saved results to {opt.out_dir}/")

if __name__ == "__main__":
    run_test()
'''
}

# 1. Populate root directory
for path, code in project_files.items():
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Created (root): {path}")

# 2. Populate surgical-desmoker directory
for path, code in project_files.items():
    target_path = os.path.join("surgical-desmoker", path)
    directory = os.path.dirname(target_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Created (surgical-desmoker): {target_path}")

print("\nSetup complete. All files have been populated successfully.")
