import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import wandb
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import cv2


# =============================================================================
# WEIGHTS & BIASES CONFIG
# =============================================================================

WANDB_API_KEY = "YOUR_WANDB_API_KEY_HERE"  # ← replace with your actual key

wandb.login(key=WANDB_API_KEY)


# =============================================================================
# BASELINE NPO LOSS (NO MASKING)
# =============================================================================

class BaselineNPOLoss(nn.Module):
    """
    Standard NPO loss over the full latent spatial extent.
    No patch-level masking — all positions contribute equally to the loss.
    Used as a baseline to compare against patch-masked NPO.
    """

    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def forward(self, noise_pred_pos, noise_pred_neg, noise_target, segmentation_masks=None):
        # segmentation_masks is accepted but unused — kept for a consistent call signature.
        loss_pos = F.mse_loss(noise_pred_pos, noise_target, reduction='none')
        loss_neg = F.mse_loss(noise_pred_neg, noise_target, reduction='none')

        # Mean over all spatial positions and channels per sample.
        sample_loss_pos = loss_pos.mean(dim=[1, 2, 3])
        sample_loss_neg = loss_neg.mean(dim=[1, 2, 3])

        log_ratio = self.beta * (sample_loss_neg - sample_loss_pos)
        npo_loss = -F.logsigmoid(log_ratio)
        weighted_npo_loss = npo_loss.mean()

        reconstruction_loss = sample_loss_pos.mean()
        total_loss = weighted_npo_loss + 0.5 * reconstruction_loss

        return total_loss, weighted_npo_loss, reconstruction_loss


# =============================================================================
# LOCAL SYNTHSCARS DATASET
# =============================================================================

IMG_SIZE = 256

preprocess_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# Fallback prompts used when an image has no refs / segmentation data
FALLBACK_POS = (
    "Professional photography with anatomically correct features, "
    "proper proportions, natural appearance, high quality, sharp focus, photorealistic"
)
FALLBACK_NEG = (
    "Low quality image with visible generation artifacts, "
    "distorted anatomy, malformed features, unnatural appearance"
)


def polygons_to_mask(refs, img_w, img_h, out_size=IMG_SIZE):
    """
    Convert the refs[].segmentation polygon lists from promptoutputs.json
    into a binary uint8 mask of shape (out_size, out_size).

    Coordinates in the JSON are in pixel space relative to the *original*
    image dimensions (img_w × img_h). We scale them to out_size × out_size.
    """
    mask = np.zeros((out_size, out_size), dtype=np.uint8)

    scale_x = out_size / img_w
    scale_y = out_size / img_h

    for ref in refs:
        for polygon in ref.get("segmentation", []):
            # polygon is a flat list: [x0, y0, x1, y1, ...]
            if len(polygon) < 6:          # need at least 3 points
                continue
            coords = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= scale_x       # x
            coords[:, 1] *= scale_y       # y
            pts = coords.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

    return mask


class SynthScarsDataset(Dataset):
    """
    Loads images + segmentation masks + NPO prompts from the local
    SynthScars folder structure:

        SynthScars/
          train/
            images/          ← mixed PNG / JPG
            annotations/
              promptoutputs.json
          test/
            images/
            annotations/
              promptoutputs.json
    """

    def __init__(self, split="train", root="SynthScars", tokenizer=None):
        assert split in ("train", "test")
        self.images_dir = os.path.join(root, split, "images")
        annotations_path = os.path.join(root, split, "annotations", "promptoutputs.json")
        self.tokenizer = tokenizer

        # ── load & flatten the JSON ──────────────────────────────────────────
        with open(annotations_path, "r") as f:
            raw = json.load(f)

        # raw is a list of single-key dicts: [{"0": {...}}, {"1": {...}}, ...]
        self.records = {}          # img_file_name -> record dict
        for entry in raw:
            for _idx, data in entry.items():
                fname = data.get("img_file_name", "")
                if fname:
                    self.records[fname] = data

        # ── collect all image files present on disk ──────────────────────────
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        all_files = [
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        # keep only files that have an annotation entry; warn about the rest
        self.samples = [f for f in all_files if f in self.records]
        missing = len(all_files) - len(self.samples)
        if missing:
            print(
                f"[SynthScarsDataset/{split}] {missing} image(s) have no annotation "
                f"entry and will be skipped."
            )

        # also warn about annotation entries with no corresponding image
        on_disk = set(all_files)
        no_image = [k for k in self.records if k not in on_disk]
        if no_image:
            print(
                f"[SynthScarsDataset/{split}] {len(no_image)} annotation entry/entries "
                f"have no matching image on disk and will be skipped."
            )

        print(f"[SynthScarsDataset/{split}] {len(self.samples)} usable samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        record = self.records[fname]

        # ── image ────────────────────────────────────────────────────────────
        img_path = os.path.join(self.images_dir, fname)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size          # needed for polygon scaling
        pixel_values = preprocess_img(image) # (3, 256, 256)

        # ── segmentation mask ────────────────────────────────────────────────
        refs = record.get("refs", [])
        segmentation_mask = polygons_to_mask(refs, orig_w, orig_h, IMG_SIZE)
        artifact_ratio = float(segmentation_mask.mean())

        # ── prompts ──────────────────────────────────────────────────────────
        pos_prompt = record.get("pos_prompt", FALLBACK_POS)
        neg_prompt = record.get("neg_prompt", FALLBACK_NEG)
        caption    = record.get("caption", "")

        # ── tokenise (if tokenizer provided at construction time) ────────────
        if self.tokenizer is not None:
            pos_ids = self._tokenize(pos_prompt)
            neg_ids = self._tokenize(neg_prompt)
        else:
            pos_ids = torch.zeros(77, dtype=torch.long)
            neg_ids = torch.zeros(77, dtype=torch.long)

        return {
            "pixel_values":      pixel_values,                          # (3,H,W)
            "segmentation_mask": torch.from_numpy(segmentation_mask),   # (H,W) uint8
            "artifact_ratio":    artifact_ratio,
            "pos_input_ids":     pos_ids,                               # (seq,)
            "neg_input_ids":     neg_ids,
            "caption":           caption,
            "img_file_name":     fname,
        }

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)   # (seq_len,)


# =============================================================================
# MODEL SETUP
# =============================================================================

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

print("Loading tokenizer ...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

print("Building datasets ...")
train_full = SynthScarsDataset(split="train", root="SynthScars", tokenizer=tokenizer)
test_ds    = SynthScarsDataset(split="test",  root="SynthScars", tokenizer=tokenizer)

# ── train / val split from the local train folder ───────────────────────────
val_size   = max(1, int(0.15 * len(train_full)))
train_size = len(train_full) - val_size
train_ds, val_ds = random_split(
    train_full,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_dataloader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=2, pin_memory=True)
val_dataloader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

print("Loading models ...")
vae          = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
unet         = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
scheduler    = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.enable_gradient_checkpointing()

LEARNING_RATE = 5e-5
BETA          = 0.1

accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=4)
optimizer   = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
loss_fn     = BaselineNPOLoss(beta=BETA)

unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    unet, optimizer, train_dataloader, val_dataloader
)

# Move frozen models to the accelerator device manually
device = accelerator.device
vae.to(device)
text_encoder.to(device)


# =============================================================================
# VALIDATION LOOP
# =============================================================================

def run_validation():
    unet.eval()
    total_loss = total_npo = total_recon = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values  = batch["pixel_values"]
            seg_masks     = batch["segmentation_mask"]
            pos_input_ids = batch["pos_input_ids"]
            neg_input_ids = batch["neg_input_ids"]

            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            pos_embeddings = text_encoder(pos_input_ids)[0]
            neg_embeddings = text_encoder(neg_input_ids)[0]

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred_pos = unet(noisy_latents, timesteps, encoder_hidden_states=pos_embeddings).sample
            noise_pred_neg = unet(noisy_latents, timesteps, encoder_hidden_states=neg_embeddings).sample

            # Baseline loss ignores masks but we pass None for consistent signature
            loss, npo_loss, recon_loss = loss_fn(noise_pred_pos, noise_pred_neg, noise, None)

            total_loss  += accelerator.gather(loss).mean().item()
            total_npo   += accelerator.gather(npo_loss).mean().item()
            total_recon += accelerator.gather(recon_loss).mean().item()
            n_batches   += 1

    n = max(n_batches, 1)
    return {"total": total_loss / n, "npo": total_npo / n, "recon": total_recon / n}


# =============================================================================
# WEIGHTS & BIASES LOGGING
# =============================================================================

def init_wandb(epochs, lr, beta):
    """Initialise a W&B run and log all hyperparameters."""
    wandb.init(
        project="npo-artifact-removal",
        name=f"baseline_npo_lr{lr}_b{beta}",
        config={
            "epochs":           epochs,
            "learning_rate":    lr,
            "beta":             beta,
            "model":            MODEL_NAME,
            "img_size":         IMG_SIZE,
            "batch_size":       2,
            "grad_accum_steps": 4,
            "mixed_precision":  "fp16",
            "variant":          "baseline",
        },
    )


def log_step(loss, npo_loss, recon_loss, step):
    """Log per-step training losses to W&B."""
    if accelerator.is_main_process:
        wandb.log({
            "train/loss_step":  loss,
            "train/npo_step":   npo_loss,
            "train/recon_step": recon_loss,
        }, step=step)


def log_epoch(epoch, train_loss, train_npo, train_recon,
              val_loss, val_npo, val_recon):
    """Log per-epoch train + val metrics and print a summary line."""
    if accelerator.is_main_process:
        wandb.log({
            "epoch":             epoch,
            "train/loss_epoch":  train_loss,
            "train/npo_epoch":   train_npo,
            "train/recon_epoch": train_recon,
            "val/loss":          val_loss,
            "val/npo":           val_npo,
            "val/recon":         val_recon,
        }, step=epoch)

    print(
        f"[Epoch {epoch}] "
        f"Train: {train_loss:.4f} (NPO {train_npo:.4f} / Recon {train_recon:.4f}) | "
        f"Val: {val_loss:.4f} (NPO {val_npo:.4f} / Recon {val_recon:.4f})"
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

EPOCHS = 10

print("\n" + "=" * 70)
print("STARTING BASELINE NPO TRAINING (NO MASKING)  (local SynthScars dataset)")
print("=" * 70)
print(f"Epochs: {EPOCHS}  |  LR: {LEARNING_RATE}  |  Beta: {BETA}")
print("=" * 70 + "\n")

init_wandb(EPOCHS, LEARNING_RATE, BETA)
global_step = 0

for epoch in range(EPOCHS):
    unet.train()
    running_loss = running_npo = running_recon = 0.0
    n_steps = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for batch in pbar:
        with accelerator.accumulate(unet):
            pixel_values  = batch["pixel_values"]
            pos_input_ids = batch["pos_input_ids"]
            neg_input_ids = batch["neg_input_ids"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                pos_embeddings = text_encoder(pos_input_ids)[0]
                neg_embeddings = text_encoder(neg_input_ids)[0]

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred_pos = unet(noisy_latents, timesteps, encoder_hidden_states=pos_embeddings).sample
            noise_pred_neg = unet(noisy_latents, timesteps, encoder_hidden_states=neg_embeddings).sample

            # Baseline loss ignores masks — pass None for consistent signature
            loss, npo_loss, recon_loss = loss_fn(noise_pred_pos, noise_pred_neg, noise, None)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

        running_loss  += loss.item()
        running_npo   += npo_loss.item()
        running_recon += recon_loss.item()
        n_steps       += 1
        global_step   += 1

        log_step(loss.item(), npo_loss.item(), recon_loss.item(), global_step)

        pbar.set_postfix({
            "loss":  f"{loss.item():.4f}",
            "npo":   f"{npo_loss.item():.4f}",
            "recon": f"{recon_loss.item():.4f}",
        })

    train_loss  = running_loss  / n_steps
    train_npo   = running_npo   / n_steps
    train_recon = running_recon / n_steps

    print("Running validation ...")
    val_losses = run_validation()

    log_epoch(
        epoch + 1,
        train_loss, train_npo, train_recon,
        val_losses["total"], val_losses["npo"], val_losses["recon"],
    )

    if (epoch + 1) % 2 == 0:
        print(f"Saving checkpoint for epoch {epoch + 1} ...")
        accelerator.wait_for_everyone()
        accelerator.unwrap_model(unet).save_pretrained(f"./baseline_npo_checkpoint_epoch_{epoch + 1}")


# =============================================================================
# SAVE FINAL MODEL
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE — SAVING FINAL MODEL")
print("=" * 70)

accelerator.wait_for_everyone()
accelerator.unwrap_model(unet).save_pretrained("./baseline_npo_unet_final")
vae.save_pretrained("./baseline_npo_vae")
text_encoder.save_pretrained("./baseline_npo_text_encoder")
tokenizer.save_pretrained("./baseline_npo_tokenizer")

print("\n✅ Training complete! Models saved.")
print("  UNet:         ./baseline_npo_unet_final")
print("  VAE:          ./baseline_npo_vae")
print("  Text encoder: ./baseline_npo_text_encoder")
print("  Tokenizer:    ./baseline_npo_tokenizer")
print("=" * 70)

if accelerator.is_main_process:
    wandb.finish()
