import os
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
