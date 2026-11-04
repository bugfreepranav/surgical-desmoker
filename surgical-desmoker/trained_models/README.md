# Trained Models Directory
This directory stores PyTorch model checkpoints (`.pt`), including:
- `best_checkpoint.pt`: Checkpoint with highest validation PSNR/SSIM.
- `latest_checkpoint.pt`: Checkpoint saved at the most recent training epoch.
- `checkpoint_epoch_XXX.pt`: Periodic checkpoint snapshots saved according to `--save_freq`.
