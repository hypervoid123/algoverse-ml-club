import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import cv2


# =============================================================================
# ENHANCED NPO LOSS WITH PATCH-LEVEL MASKING
# =============================================================================

class EnhancedNPOLoss(nn.Module):
    """
    Enhanced NPO with patch-level artifact masking
    Only penalizes patches labeled as containing artifacts
    """
    
    def __init__(self, beta=0.1, artifact_weight=2.0, patch_size=64):
        super().__init__()
        self.beta = beta
        self.artifact_weight = artifact_weight
        self.patch_size = patch_size
    
    def get_patch_mask(self, segmentation_masks, latent_size):
        latent_h, latent_w = latent_size

        patch_labels = []

        for mask in segmentation_masks:
            h, w = mask.shape
            stride_h = h // latent_h
            stride_w = w // latent_w

            # Vectorized: max_pool2d with artifact-region stride gives 1 if any
            # artifact pixel falls within a latent patch, 0 otherwise.
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            downsampled = F.max_pool2d(
                mask_tensor,
                kernel_size=(stride_h, stride_w),
                stride=(stride_h, stride_w)
            ).squeeze()  # [latent_h, latent_w]

            patch_labels.append(downsampled)

        return torch.stack(patch_labels)  # [B, latent_h, latent_w]
    
    def forward(self, noise_pred_pos, noise_pred_neg, noise_target, segmentation_masks):
        _, _, latent_h, latent_w = noise_pred_pos.shape
        
        patch_labels = self.get_patch_mask(
            segmentation_masks, 
            (latent_h, latent_w)
        ).to(noise_pred_pos.device)

        patch_labels_expanded = patch_labels.unsqueeze(1)
        
        loss_pos = F.mse_loss(noise_pred_pos, noise_target, reduction='none')
        loss_neg = F.mse_loss(noise_pred_neg, noise_target, reduction='none')
        
        masked_loss_pos = loss_pos * patch_labels_expanded
        masked_loss_neg = loss_neg * patch_labels_expanded
        
        # Count artifact patches (spatial only) then scale by number of channels,
        # so the denominator matches the full [C, H, W] sum in masked_loss.
        num_channels = noise_pred_pos.shape[1]
        artifact_pixels = patch_labels.sum(dim=[1, 2]) * num_channels
        artifact_pixels = torch.clamp(artifact_pixels, min=1.0)
        
        sample_loss_pos = masked_loss_pos.sum(dim=[1, 2, 3]) / artifact_pixels
        sample_loss_neg = masked_loss_neg.sum(dim=[1, 2, 3]) / artifact_pixels
        
        log_ratio = self.beta * (sample_loss_neg - sample_loss_pos)
        npo_loss = -F.logsigmoid(log_ratio)
        
        artifact_ratio = patch_labels.mean(dim=[1, 2])
        weights = 1.0 + artifact_ratio * (self.artifact_weight - 1.0)
        weighted_npo_loss = (npo_loss * weights).mean()
        
        reconstruction_loss = sample_loss_pos.mean()
        total_loss = weighted_npo_loss + 0.5 * reconstruction_loss
        
        return total_loss, weighted_npo_loss, reconstruction_loss


# =============================================================================
# LOAD PREPROCESSED PROMPTS FROM JSON
# =============================================================================

PROMPTS_JSON_PATH = "train_with_prompts.json"  # Path to your preprocessed prompts file

print("Loading preprocessed prompts from JSON...")
with open(PROMPTS_JSON_PATH, "r") as f:
    raw_prompt_list = json.load(f)

# Build a lookup dict: img_file_name -> {pos_prompt, neg_prompt}
# The JSON is a list of dicts like [{"0": {...}}, {"1": {...}}, ...]
prompt_lookup = {}
for entry in raw_prompt_list:
    for idx, data in entry.items():
        filename = data.get("img_file_name", "")
        prompt_lookup[filename] = {
            "pos_prompt": data.get("pos_prompt", "a high quality photograph"),
            "neg_prompt": data.get("neg_prompt", "a low quality photograph"),
        }

print(f"Loaded prompts for {len(prompt_lookup)} images.")

# Fallback prompts for images not found in the JSON
FALLBACK_POS = (
    "Professional photography with anatomically correct features, "
    "proper proportions, natural appearance, high quality, sharp focus, photorealistic"
)
FALLBACK_NEG = (
    "Low quality image with visible generation artifacts, "
    "distorted anatomy, malformed features, unnatural appearance"
)

def get_prompts_for_filename(filename):
    """Look up pos/neg prompts by image filename, with fallback."""
    entry = prompt_lookup.get(filename)
    if entry:
        return entry["pos_prompt"], entry["neg_prompt"]
    return FALLBACK_POS, FALLBACK_NEG


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

print("Loading dataset...")
dataset = load_dataset("khr0516/SynthScars", split="train").shuffle(seed=42)

split = dataset.train_test_split(test_size=0.15, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

preprocess_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def preprocess_batch(batch):
    images = [preprocess_img(img.convert("RGB")) for img in batch["image"]]
    masks = []

    for segs in batch["segmentation"]:
        mask = np.zeros((256, 256), dtype=np.uint8)
        if segs:
            for seg in segs:
                pts = np.array(seg).reshape(-1, 2)
                pts[:, 0] *= 256
                pts[:, 1] *= 256
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        masks.append(mask)

    artifact_ratio = [m.mean() for m in masks]

    return {
        "pixel_values": images,
        "artifact_ratio": artifact_ratio,
        "segmentation_mask": masks,
        "caption": batch["caption"],
        # img_file_name is needed to look up prompts - keep it if present in dataset
        # If the dataset doesn't have this column, prompts fall back to caption-based logic below
        "img_file_name": batch.get("img_file_name", [""] * len(images)),
    }


train_dataset = train_dataset.map(
    preprocess_batch, batched=True, remove_columns=["image"]
)
val_dataset = val_dataset.map(
    preprocess_batch, batched=True, remove_columns=["image"]
)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")


# =============================================================================
# TOKENIZATION - LOADS PROMPTS FROM JSON LOOKUP
# =============================================================================

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")


def tokenize_with_npo(batch):
    """
    Tokenize positive and negative prompts loaded from the preprocessed JSON.
    Falls back to generic prompts if a filename isn't found in the lookup.
    """
    pos_prompts = []
    neg_prompts = []

    filenames = batch.get("img_file_name", [""] * len(batch["caption"]))

    for filename, caption in zip(filenames, batch["caption"]):
        pos, neg = get_prompts_for_filename(filename)

        # If filename wasn't in JSON, try matching by caption as a last resort
        if pos == FALLBACK_POS and caption:
            pos = (
                f"{caption.split('.')[0].strip()}. "
                "Professional photography with anatomically correct features, "
                "proper proportions, natural appearance, realistic skin texture, "
                "high quality, sharp focus, photorealistic, clean image"
            )
            neg = (
                f"{caption.split('.')[0].strip()}. "
                "Low quality image with visible generation artifacts, "
                "distorted anatomy, malformed features, unnatural appearance, "
                "AI generation defects"
            )

        pos_prompts.append(pos)
        neg_prompts.append(neg)

    pos_tokens = tokenizer(
        pos_prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )

    neg_tokens = tokenizer(
        neg_prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )

    batch["pos_input_ids"] = pos_tokens.input_ids
    batch["neg_input_ids"] = neg_tokens.input_ids

    return batch


train_dataset = train_dataset.map(tokenize_with_npo, batched=True)
val_dataset = val_dataset.map(tokenize_with_npo, batched=True)

train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)


# =============================================================================
# MODEL SETUP
# =============================================================================

print("Loading models...")
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.enable_gradient_checkpointing()

LEARNING_RATE = 5e-5
BETA = 0.1
ARTIFACT_WEIGHT = 3.0

accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

loss_fn = EnhancedNPOLoss(beta=BETA, artifact_weight=ARTIFACT_WEIGHT, patch_size=64)

unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    unet, optimizer, train_dataloader, val_dataloader
)


# =============================================================================
# VALIDATION LOOP
# =============================================================================

def run_validation():
    unet.eval()
    total_loss = 0.0
    total_npo = 0.0
    total_recon = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values = batch["pixel_values"]
            segmentation_masks = batch["segmentation_mask"]
            pos_input_ids = batch["pos_input_ids"]
            neg_input_ids = batch["neg_input_ids"]

            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

            pos_embeddings = text_encoder(pos_input_ids)[0]
            neg_embeddings = text_encoder(neg_input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred_pos = unet(
                noisy_latents, timesteps, encoder_hidden_states=pos_embeddings
            ).sample
            noise_pred_neg = unet(
                noisy_latents, timesteps, encoder_hidden_states=neg_embeddings
            ).sample

            loss, npo_loss, recon_loss = loss_fn(
                noise_pred_pos, noise_pred_neg, noise, segmentation_masks
            )

            total_loss += accelerator.gather(loss).mean().item()
            total_npo += accelerator.gather(npo_loss).mean().item()
            total_recon += accelerator.gather(recon_loss).mean().item()
            n_batches += 1

    return {
        'total': total_loss / max(n_batches, 1),
        'npo': total_npo / max(n_batches, 1),
        'recon': total_recon / max(n_batches, 1)
    }


# =============================================================================
# LOSS TRACKING
# =============================================================================

class LossLogger:
    def __init__(self):
        self.train_total = []
        self.train_npo = []
        self.train_recon = []
        self.val_total = []
        self.val_npo = []
        self.val_recon = []

    def log(self, train_total, train_npo, train_recon, val_total, val_npo, val_recon):
        self.train_total.append(train_total)
        self.train_npo.append(train_npo)
        self.train_recon.append(train_recon)
        self.val_total.append(val_total)
        self.val_npo.append(val_npo)
        self.val_recon.append(val_recon)

    def print_last(self, epoch):
        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {self.train_total[-1]:.4f} "
            f"(NPO: {self.train_npo[-1]:.4f}, Recon: {self.train_recon[-1]:.4f}) | "
            f"Val Loss: {self.val_total[-1]:.4f} "
            f"(NPO: {self.val_npo[-1]:.4f}, Recon: {self.val_recon[-1]:.4f})"
        )


class LiveLossPlotter:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        self.train_line, = self.ax1.plot([], [], label="Train Total", color='blue')
        self.val_line, = self.ax1.plot([], [], label="Val Total", color='red')
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Total Loss")
        self.ax1.set_title("Total Loss")
        self.ax1.legend()
        self.ax1.grid(True)
        
        self.train_npo_line, = self.ax2.plot([], [], label="Train NPO", color='green')
        self.train_recon_line, = self.ax2.plot([], [], label="Train Recon", color='orange')
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Loss")
        self.ax2.set_title("NPO vs Reconstruction Loss")
        self.ax2.legend()
        self.ax2.grid(True)

    def update(self, logger):
        epochs = range(1, len(logger.train_total) + 1)

        self.train_line.set_data(epochs, logger.train_total)
        self.val_line.set_data(epochs, logger.val_total)
        self.ax1.set_xlim(1, max(1, len(epochs)))
        self.ax1.set_ylim(0, max(logger.train_total + logger.val_total) * 1.1)

        self.train_npo_line.set_data(epochs, logger.train_npo)
        self.train_recon_line.set_data(epochs, logger.train_recon)
        self.ax2.set_xlim(1, max(1, len(epochs)))
        self.ax2.set_ylim(0, max(logger.train_npo + logger.train_recon) * 1.1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


logger = LossLogger()
plotter = LiveLossPlotter()


# =============================================================================
# NPO TRAINING LOOP
# =============================================================================

EPOCHS = 10

print("\n" + "=" * 70)
print("STARTING ENHANCED NPO TRAINING")
print("=" * 70)
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Beta (NPO strength): {BETA}")
print(f"Artifact Weight: {ARTIFACT_WEIGHT}")
print(f"Prompts loaded from: {PROMPTS_JSON_PATH}")
print("=" * 70 + "\n")

for epoch in range(EPOCHS):
    unet.train()
    running_loss = 0.0
    running_npo = 0.0
    running_recon = 0.0
    n_steps = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"]
            segmentation_masks = batch["segmentation_mask"]
            pos_input_ids = batch["pos_input_ids"]
            neg_input_ids = batch["neg_input_ids"]

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                pos_embeddings = text_encoder(pos_input_ids)[0]
                neg_embeddings = text_encoder(neg_input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred_pos = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=pos_embeddings
            ).sample

            noise_pred_neg = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=neg_embeddings
            ).sample

            loss, npo_loss, recon_loss = loss_fn(
                noise_pred_pos,
                noise_pred_neg,
                noise,
                segmentation_masks
            )

            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_npo += npo_loss.item()
            running_recon += recon_loss.item()
            n_steps += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'npo': f'{npo_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })

    train_loss = running_loss / n_steps
    train_npo = running_npo / n_steps
    train_recon = running_recon / n_steps
    
    print("Running validation...")
    val_losses = run_validation()

    logger.log(
        train_loss, train_npo, train_recon,
        val_losses['total'], val_losses['npo'], val_losses['recon']
    )
    logger.print_last(epoch + 1)
    plotter.update(logger)
    
    if (epoch + 1) % 2 == 0:
        print(f"Saving checkpoint for epoch {epoch+1}...")
        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(f"./npo_checkpoint_epoch_{epoch+1}")


# =============================================================================
# SAVE FINAL MODEL
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE - SAVING FINAL MODEL")
print("=" * 70)

accelerator.wait_for_everyone()
unwrapped_unet = accelerator.unwrap_model(unet)

unwrapped_unet.save_pretrained("./npo_artifact_free_unet_final")
vae.save_pretrained("./npo_artifact_free_vae")
text_encoder.save_pretrained("./npo_artifact_free_text_encoder")
tokenizer.save_pretrained("./npo_artifact_free_tokenizer")

print("\n✅ Training complete! Models saved.")
print("\nSaved components:")
print("  - UNet (fine-tuned):     ./npo_artifact_free_unet_final")
print("  - VAE:                   ./npo_artifact_free_vae")
print("  - Text Encoder:          ./npo_artifact_free_text_encoder")
print("  - Tokenizer:             ./npo_artifact_free_tokenizer")
print("\n" + "=" * 70)
