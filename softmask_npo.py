import os
import json
import random

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
# SEED CONTROL
# Global seed is set once and never touched inside the training loop.
# Dataloader shuffle gets its own isolated generator so the CUDA stochastic
# state (noise tensors) drifts freely — prevents the UNet from memorising
# specific noise offsets tied to specific flawed/clean pairs.
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

dataset_generator = torch.Generator().manual_seed(SEED)


# =============================================================================
# WEIGHTS & BIASES
# =============================================================================

wandb_key = "--redacted--"
wandb.login(key=wandb_key)


# =============================================================================
# ATTENTION-ISOLATED GRADIENT ROUTING  (AIGR)
#
# Problem
# -------
# Even with a spatial soft-mask on the NPO loss, the UNet's self-attention
# layers globally entangle token representations.  During the backward pass,
# the gradient of L w.r.t. the attention probability matrix A has shape
# [B*heads, N, N].  Entry (i, j) says "how much should token i update based
# on its attention to token j?"  Because A is fully dense, artifact tokens
# bleed gradient into background tokens and vice versa — fixing a distorted
# hand subtly shifts the background sky.
#
# Solution
# --------
# Register a backward hook on A inside every self-attention layer.  The hook
# intercepts the N×N gradient and zeros out cross-terms between artifact and
# background token groups via a routing matrix R:
#
#   R_ij = m_i · m_j  +  (1 − m_i)(1 − m_j)
#
# where m ∈ [0,1]^N is the soft token mask (continuous, not binarised).
# This keeps artifact↔artifact and background↔background gradients intact
# while blocking the artifact↔background cross-terms.
#
# Design decisions
# ----------------
# SOFT routing      — uses avg_pooled values directly; boundary tokens get
#                     partial gradient flow proportional to artifact coverage.
#
# COSINE schedule   — routing strength s(t) eases from 1.0 (fully strict)
#                     at low t to routing_floor (relaxed) at high t via a
#                     cosine curve.  At high t the model sets global structure
#                     and some cross-term flow is acceptable; at low t it
#                     refines local detail and needs surgical isolation.
#                     Cosine was chosen over sigmoid (step-function artefact)
#                     and linear (no ease-in) because it is harmonically
#                     aligned with the cosine noise schedule.
#
# ZERO-COVERAGE     — when artifact_ratio = 0, m ≈ 0 everywhere, so
#                     R = all-ones (full gradient flow, no constraint).
#                     Clean samples are handled separately in the loss via
#                     clean_sample_weight rather than via routing.
#
# COMPATIBILITY     — register_hook on intermediate tensors is incompatible
#                     with gradient checkpointing (GC recomputes activations
#                     during backward, creating new tensor objects that do not
#                     carry the original hooks).  GC is therefore DISABLED.
#                     Memory is recovered by increasing grad_accum_steps.
#
# NOTE ON DDP WRAPPING
# -----------------------------------------------------------------------
# AIGRManager installs processors on the base UNet before accelerator.prepare().
# After DDP wrapping, self._processors still holds references to the same
# processor objects (DDP does not copy them — it wraps the module in place),
# so aigr_manager.update() correctly reaches the live processors through
# those stored references.  If you switch to FSDP this may break and
# processors would need to be re-fetched from the unwrapped model.
# =============================================================================

class AIGRAttnProcessor:
    """
    Drop-in replacement for diffusers AttnProcessor.

    Performs standard attention in the forward pass (identical output to the
    default processor).  During training, registers a backward hook on the
    attention probability tensor for self-attention layers only.
    Cross-attention is left completely unmodified — hooking there would
    corrupt the text-conditioning pathway.

    IMPORTANT: AIGR has zero effect on the forward pass — it only modifies
    which direction Q/K/V projection weights move during the backward pass.
    Its effect is therefore not visible in loss curves alone; evaluation
    should compare how much background regions shift in generated images
    before vs. after fine-tuning, with and without AIGR.
    """

    def __init__(self, routing_floor=0.1):
        self.routing_floor    = routing_floor
        self._avg_pooled_mask = None   # [B, latent_H, latent_W], set per batch
        self._t_norm          = None   # [B], values in [0, 1],   set per batch

    def set_batch_info(self, avg_pooled_mask, t_norm):
        """
        Store routing data for the upcoming forward pass.
        Call once per batch (via AIGRManager.update) before unet().
        avg_pooled_mask and t_norm must already be detached.
        """
        self._avg_pooled_mask = avg_pooled_mask
        self._t_norm          = t_norm

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):

        is_cross_attn = encoder_hidden_states is not None
        residual      = hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        N = seq_len

        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)

        query = attn.head_to_batch_dim(attn.to_q(hidden_states))

        kv_src = encoder_hidden_states if is_cross_attn else hidden_states
        if is_cross_attn and getattr(attn, 'norm_cross', None) is not None:
            kv_src = attn.norm_cross(kv_src)

        key   = attn.head_to_batch_dim(attn.to_k(kv_src))
        value = attn.head_to_batch_dim(attn.to_v(kv_src))

        # Explicit attention probability computation — we need a named tensor
        # to call register_hook on.  Cannot use F.scaled_dot_product_attention
        # (fused op, no accessible intermediate tensor).
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # shape: [B*heads, N, N]

        # Hook: self-attention only, training mode only, mask available
        if attn.training and not is_cross_attn and self._avg_pooled_mask is not None:
            hook_fn = self._build_routing_hook(batch_size, N, attention_probs.device)
            if hook_fn is not None:
                attention_probs.register_hook(hook_fn)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if getattr(attn, 'residual_connection', False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / getattr(attn, 'rescale_output_factor', 1.0)

        return hidden_states

    def _build_routing_hook(self, B, N, device):
        """
        Construct and return the gradient hook for this attention layer.

        Routing matrix (soft, per-sample):
            R_ij = m_i·m_j + (1−m_i)(1−m_j)         ∈ [0, 1]

        Timestep-conditioned blending (cosine ease):
            s(t) = floor + (1−floor) · ½(1 + cos(π · t_norm))
            R_eff = s(t)·R + (1−s(t))·1

        At t=0   → s=1.0  → full routing constraint (surgical)
        At t=T   → s=floor → nearly unmasked (global structure phase)
        """
        avg_pooled    = self._avg_pooled_mask   # [B, lH, lW]
        t_norm        = self._t_norm            # [B]
        routing_floor = self.routing_floor

        H = int(N ** 0.5)
        if H * H != N:
            # Non-square spatial dim — skip to avoid shape errors
            return None

        # Project soft mask to this attention layer's token resolution
        pooled = F.adaptive_avg_pool2d(
            avg_pooled.unsqueeze(1).to(device), (H, H)
        ).squeeze(1).reshape(B, N).float()   # [B, N], values ∈ [0, 1]

        # Cosine routing strength schedule
        # s=1 at t_norm=0 (low t, strict), s=floor at t_norm=1 (high t, relaxed)
        cosine_factor = 0.5 * (1.0 + torch.cos(
            torch.pi * t_norm.to(device).float()
        ))   # [B], ∈ [0, 1]
        s = routing_floor + (1.0 - routing_floor) * cosine_factor   # [B]

        def hook(grad):
            # grad: [B*heads, N, N]
            orig_dtype = grad.dtype
            grad_f     = grad.float()

            BH = grad_f.shape[0]
            h  = BH // B   # number of heads

            m = pooled  # [B, N]

            # Soft block-diagonal routing matrix via outer products
            artifact_block = torch.einsum('bi,bj->bij', m,       m      )  # [B,N,N]
            bg_block       = torch.einsum('bi,bj->bij', 1.0 - m, 1.0 - m) # [B,N,N]
            R = artifact_block + bg_block   # [B, N, N], values ∈ [0, 1]

            # Blend: at high t relax toward all-ones (no constraint)
            R_eff = s.view(B, 1, 1) * R + (1.0 - s.view(B, 1, 1))   # [B, N, N]

            # Expand across heads: [B, N, N] → [B*heads, N, N]
            R_exp = (R_eff
                     .unsqueeze(1)
                     .expand(B, h, N, N)
                     .reshape(BH, N, N))

            return (grad_f * R_exp).to(orig_dtype)

        return hook


class AIGRManager:
    """
    Installs AIGRAttnProcessor into every attention layer of the UNet
    and provides a unified update() call for per-batch routing data.

    Construct BEFORE accelerator.prepare() — processors must be installed
    on the base model before any DDP/FSDP wrapping occurs.
    The manager holds direct references to processor objects, so update()
    works correctly regardless of how the model is subsequently wrapped.
    """

    def __init__(self, unet, routing_floor=0.1):
        self._processors = {}
        attn_proc_dict   = {}

        for name in unet.attn_processors.keys():
            proc                   = AIGRAttnProcessor(routing_floor=routing_floor)
            attn_proc_dict[name]   = proc
            self._processors[name] = proc

        unet.set_attn_processor(attn_proc_dict)

        n_self  = sum(1 for k in self._processors if 'attn1' in k)
        n_cross = sum(1 for k in self._processors if 'attn2' in k)
        print(f"[AIGR] Installed: {n_self} self-attn processors | "
              f"{n_cross} cross-attn processors (hooks inactive on cross-attn).")

    def update(self, avg_pooled_mask, t_norm):
        """
        Push current-batch routing data into all processors.
        Call after computing avg_pooled + timesteps, before unet().

        avg_pooled_mask : [B, latent_H, latent_W]  soft mask ∈ [0, 1], detached
        t_norm          : [B]  normalised timestep t/T ∈ [0, 1], detached
        """
        for proc in self._processors.values():
            proc.set_batch_info(avg_pooled_mask, t_norm)


# =============================================================================
# SOFTMASK NPO LOSS  (v4)
#
# Key design choices:
#   - avg_pooled is accepted as an optional precomputed argument so the
#     training loop can compute it once, pass it to AIGR, and pass the same
#     tensor here — avoiding a redundant adaptive_avg_pool2d per batch.
#   - snr_weights are applied to BOTH NPO and reconstruction terms so the
#     two components are weighted consistently across the timestep range.
#   - Clean sample bonus on the recon term: when artifact_ratio ≈ 0 the
#     sample is a pristine positive anchor; upweighting recon here teaches
#     the model what "correct" looks like without a separate loss.
#   - log_ratio is clamped to [-20, 20] to prevent fp16 overflow as
#     sample_loss_pos → 0 late in training.
#   - All noise tensors cast to fp32 for numerical stability under
#     mixed-precision training.
# =============================================================================

class SoftmaskNPOLoss(nn.Module):

    def __init__(self, beta=0.5, artifact_weight=3.0, background_weight=0.05,
                 recon_weight=0.3, mask_alpha_max=3.0, clean_sample_weight=0.5):
        super().__init__()
        self.beta                = beta
        self.artifact_weight     = artifact_weight
        self.background_weight   = background_weight
        self.recon_weight        = recon_weight
        self.mask_alpha_max      = mask_alpha_max
        self.clean_sample_weight = clean_sample_weight

    def get_soft_weights(self, segmentation_masks, latent_size, timesteps, T,
                         precomputed_avg_pooled=None):
        """
        Parameters
        ----------
        precomputed_avg_pooled : [B, latent_h, latent_w] or None
            If provided (already computed for AIGR), skip the pool op entirely.
            This avoids running adaptive_avg_pool2d twice per batch.

        Returns
        -------
        avg_pooled   : [B, latent_h, latent_w]  raw pooled mask ∈ [0, 1]
                       Used for: normalisation, artifact_ratio, AIGR
        soft_weights : [B, latent_h, latent_w]  sharpened + floored weights
                       Used for: spatial loss weighting
        """
        latent_h, latent_w = latent_size
        B      = segmentation_masks.shape[0]
        device = segmentation_masks.device

        # FIX: reuse precomputed avg_pooled from training loop (same tensor
        # already passed to AIGR) to avoid a redundant pool op per batch.
        if precomputed_avg_pooled is not None:
            avg_pooled = precomputed_avg_pooled
        else:
            avg_pooled = F.adaptive_avg_pool2d(
                segmentation_masks.float().unsqueeze(1), (latent_h, latent_w)
            ).squeeze(1)   # [B, lH, lW]

        # Timestep-conditioned sharpening
        # α(t) = 1 + (α_max − 1)(1 − t_norm)
        #   t → T : α → 1   (diffuse, model sets global structure)
        #   t → 0 : α → α_max  (sharp, model refines local detail)
        t_norm     = timesteps.float().to(device) / float(T)
        alpha      = 1.0 + (self.mask_alpha_max - 1.0) * (1.0 - t_norm)
        sharp_mask = avg_pooled ** alpha.view(B, 1, 1)

        soft_weights = sharp_mask * (1.0 - self.background_weight) + self.background_weight

        return avg_pooled, soft_weights

    def forward(self, noise_pred_pos, noise_pred_neg, noise_target,
                segmentation_masks, timesteps, T,
                snr_weights=None, precomputed_avg_pooled=None):
        """
        Parameters
        ----------
        noise_pred_pos          [B, C, H, W]
        noise_pred_neg          [B, C, H, W]
        noise_target            [B, C, H, W]
        segmentation_masks      [B, H_img, W_img]
        timesteps               [B]  integer timesteps
        T                       int  scheduler.config.num_train_timesteps
        snr_weights             [B] or None
            Applied to BOTH NPO and reconstruction terms for consistency.
        precomputed_avg_pooled  [B, latent_h, latent_w] or None
            Pass the avg_pooled already computed for AIGR to skip
            redundant pooling inside get_soft_weights.

        Returns
        -------
        total_loss, weighted_npo_loss, reconstruction_loss, avg_pooled
        avg_pooled is returned so callers that don't precompute it can
        still pass it to AIGR without a separate pool call.
        """
        # Cast to fp32 for numerical stability under mixed-precision training
        noise_pred_pos = noise_pred_pos.float()
        noise_pred_neg = noise_pred_neg.float()
        noise_target   = noise_target.float()

        B, C, latent_h, latent_w = noise_pred_pos.shape

        avg_pooled, soft_weights = self.get_soft_weights(
            segmentation_masks, (latent_h, latent_w), timesteps, T,
            precomputed_avg_pooled=precomputed_avg_pooled,
        )
        soft_weights_exp = soft_weights.unsqueeze(1)   # [B, 1, lH, lW]

        loss_pos = F.mse_loss(noise_pred_pos, noise_target, reduction='none')
        loss_neg = F.mse_loss(noise_pred_neg, noise_target, reduction='none')

        weighted_pos = loss_pos * soft_weights_exp
        weighted_neg = loss_neg * soft_weights_exp

        # Normalise using raw avg_pooled (not sharpened) so the denominator
        # is stable across the timestep range regardless of sharpening alpha.
        artifact_weight_sum = torch.clamp(avg_pooled.sum(dim=[1, 2]) * C, min=1.0)

        sample_loss_pos = weighted_pos.sum(dim=[1, 2, 3]) / artifact_weight_sum  # [B]
        sample_loss_neg = weighted_neg.sum(dim=[1, 2, 3]) / artifact_weight_sum  # [B]

        log_ratio = torch.clamp(
            self.beta * (sample_loss_neg - sample_loss_pos), -20.0, 20.0
        )
        npo_loss = -F.logsigmoid(log_ratio)   # [B]

        # Artifact-ratio importance weights
        artifact_ratio = avg_pooled.mean(dim=[1, 2])              # [B]
        sample_weights = 1.0 + artifact_ratio * (self.artifact_weight - 1.0)

        # FIX: apply snr_weights to sample_weights so NPO and recon are
        # weighted consistently across the timestep range.
        # Previously snr_weights was only applied to NPO, meaning the recon
        # term dominated at high-SNR (low-t) timesteps where NPO was
        # downweighted, pulling the loss in an inconsistent direction.
        if snr_weights is not None:
            sample_weights = sample_weights * snr_weights

        weighted_npo_loss = (npo_loss * sample_weights).mean()

        # Clean sample bonus: at artifact_ratio=0 → bonus = 1 + clean_sample_weight
        #                     at artifact_ratio=1 → bonus = 1.0 (no bonus)
        # Upweighting recon for clean samples anchors the model to correct
        # anatomy without requiring a separate regularisation loss.
        clean_ratio  = 1.0 - artifact_ratio
        clean_bonus  = 1.0 + clean_ratio * self.clean_sample_weight   # [B]

        # FIX: apply same snr_weights to recon for timestep consistency.
        recon_per_sample = sample_loss_pos * clean_bonus
        if snr_weights is not None:
            recon_per_sample = recon_per_sample * snr_weights
        recon_loss = recon_per_sample.mean()

        total_loss = weighted_npo_loss + self.recon_weight * recon_loss

        return total_loss, weighted_npo_loss, recon_loss, avg_pooled


# =============================================================================
# LOCAL SYNTHSCARS DATASET
# =============================================================================

IMG_SIZE = 256

preprocess_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

FALLBACK_POS = (
    "Professional photography with anatomically correct features, "
    "proper proportions, natural appearance, high quality, sharp focus, photorealistic"
)
FALLBACK_NEG = (
    "Low quality image with visible generation artifacts, "
    "distorted anatomy, malformed features, unnatural appearance"
)


def polygons_to_mask(refs, img_w, img_h, out_size=IMG_SIZE):
    mask    = np.zeros((out_size, out_size), dtype=np.uint8)
    scale_x = out_size / img_w
    scale_y = out_size / img_h

    for ref in refs:
        for polygon in ref.get("segmentation", []):
            if len(polygon) < 6:
                continue
            coords = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= scale_x
            coords[:, 1] *= scale_y
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

    return mask


class SynthScarsDataset(Dataset):
    def __init__(self, split="train", root="SynthScars/SynthScars", tokenizer=None):
        assert split in ("train", "test")
        self.images_dir  = os.path.join(root, split, "images")
        self.tokenizer   = tokenizer

        annot_path = os.path.join(root, split, "annotations", f"{split}.json")
        with open(annot_path, "r") as f:
            raw = json.load(f)

        self.records = {}
        for entry in raw:
            for _idx, data in entry.items():
                fname = data.get("img_file_name", "")
                if fname:
                    self.records[fname] = data

        valid_exts  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        all_files   = [
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        self.samples = [f for f in all_files if f in self.records]

        missing  = len(all_files) - len(self.samples)
        no_image = [k for k in self.records if k not in set(all_files)]
        if missing:
            print(f"[SynthScarsDataset/{split}] {missing} image(s) skipped — no annotation.")
        if no_image:
            print(f"[SynthScarsDataset/{split}] {len(no_image)} annotation(s) missing image.")
        print(f"[SynthScarsDataset/{split}] {len(self.samples)} usable samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname  = self.samples[idx]
        record = self.records[fname]

        image          = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        orig_w, orig_h = image.size
        pixel_values   = preprocess_img(image)

        refs              = record.get("refs", [])
        segmentation_mask = polygons_to_mask(refs, orig_w, orig_h, IMG_SIZE)

        pos_prompt = record.get("pos_prompt", FALLBACK_POS)
        neg_prompt = record.get("neg_prompt", FALLBACK_NEG)

        if self.tokenizer is not None:
            pos_ids = self._tokenize(pos_prompt)
            neg_ids = self._tokenize(neg_prompt)
        else:
            pos_ids = torch.zeros(77, dtype=torch.long)
            neg_ids = torch.zeros(77, dtype=torch.long)

        return {
            "pixel_values":      pixel_values,
            "segmentation_mask": torch.from_numpy(segmentation_mask).float(),
            "pos_input_ids":     pos_ids,
            "neg_input_ids":     neg_ids,
            "caption":           record.get("caption", ""),
            "img_file_name":     fname,
        }

    def _tokenize(self, text):
        return self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)


# =============================================================================
# MODEL SETUP
# =============================================================================

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

print("Loading tokenizer...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

print("Building datasets...")
train_full = SynthScarsDataset(split="train", root="SynthScars/SynthScars", tokenizer=tokenizer)
test_ds    = SynthScarsDataset(split="test",  root="SynthScars/SynthScars", tokenizer=tokenizer)

val_size   = max(1, int(0.15 * len(train_full)))
train_size = len(train_full) - val_size
train_ds, val_ds = random_split(
    train_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_dataloader = DataLoader(
    train_ds, batch_size=16, shuffle=True,
    num_workers=4, pin_memory=True,
    generator=dataset_generator,
)
val_dataloader = DataLoader(
    val_ds, batch_size=16, shuffle=False,
    num_workers=4, pin_memory=True,
)

print("Loading models...")
vae          = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
unet         = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
scheduler    = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# NOTE: gradient checkpointing is intentionally NOT enabled.
# register_hook on intermediate tensors is incompatible with GC:
# GC recomputes activations during backward, creating new tensor objects
# that do not carry the original hooks, silently disabling AIGR.
# Memory is recovered via increased grad_accum_steps (4 → 6).
# If you hit OOM, reduce batch_size to 8 and set grad_accum_steps=12
# to maintain the same effective batch size of 96.


# =============================================================================
# SNR-WEIGHTED TIMESTEP WEIGHTING
#
# Timestep sampling stays UNIFORM (torch.randint) to keep the reconstruction
# loss stable across the full [0, T] range.
#
# A combined weight w(t) is applied to BOTH NPO and reconstruction terms:
#   w(t) = w_SNR(t) · w_gauss(t)
#
# Min-SNR component (Hang et al. 2023):
#   w_SNR(t) = min(SNR(t), γ) / SNR(t)
#   Downweights high-SNR (low-t) steps where texture dominates.
#
# Gaussian structural-focus component:
#   w_gauss(t) = exp(−½((t − μ) / σ)²)  floored at gaussian_floor
#   Concentrates gradient budget in the structural midband (μ=500)
#   while the floor (0.15) ensures extreme timesteps still receive
#   meaningful gradient signal.
# =============================================================================

SNR_GAMMA   = 5.0
GAUSS_MEAN  = 500.0
GAUSS_STD   = 180.0
GAUSS_FLOOR = 0.15

_alphas_cumprod = scheduler.alphas_cumprod.float()            # [1000], CPU
snr_table       = _alphas_cumprod / (1.0 - _alphas_cumprod)  # [1000], CPU


def get_snr_weights(timesteps, device):
    """
    Combined Min-SNR × Gaussian structural-focus weights.
    Returns shape [B] float tensor on `device`.
    """
    snr_vals = snr_table[timesteps.cpu()].to(device).float()
    snr_w    = torch.minimum(snr_vals, torch.full_like(snr_vals, SNR_GAMMA)) / snr_vals

    gauss_raw = torch.exp(
        -0.5 * ((timesteps.float().to(device) - GAUSS_MEAN) / GAUSS_STD) ** 2
    )
    gauss_w = gauss_raw * (1.0 - GAUSS_FLOOR) + GAUSS_FLOOR

    return snr_w * gauss_w


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

LEARNING_RATE       = 5e-5   # scaled for effective batch 96 vs baseline 8
BETA                = 0.5
ARTIFACT_WEIGHT     = 3.0
BACKGROUND_WEIGHT   = 0.05
RECON_WEIGHT        = 0.3
MASK_ALPHA_MAX      = 3.0    # mask sharpening exponent at t=0; 1.0 = no sharpening
CLEAN_SAMPLE_WEIGHT = 0.5    # recon bonus multiplier for zero-artifact samples
ROUTING_FLOOR       = 0.1    # minimum AIGR routing strength at high t
EPOCHS              = 30

T = scheduler.config.num_train_timesteps   # 1000


# =============================================================================
# AIGR SETUP
# Must happen before accelerator.prepare() — set_attn_processor modifies
# the base UNet in place; DDP wrapping must come after.
# =============================================================================

print("Installing AIGR processors...")
aigr_manager = AIGRManager(unet, routing_floor=ROUTING_FLOOR)


# =============================================================================
# ACCELERATOR + OPTIMIZER + LOSS
# grad_accum_steps=6: effective batch = 16 × 6 = 96
# Increased from 4 to partially offset memory cost of disabling GC.
# =============================================================================

accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=6)
optimizer   = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
loss_fn     = SoftmaskNPOLoss(
    beta=BETA,
    artifact_weight=ARTIFACT_WEIGHT,
    background_weight=BACKGROUND_WEIGHT,
    recon_weight=RECON_WEIGHT,
    mask_alpha_max=MASK_ALPHA_MAX,
    clean_sample_weight=CLEAN_SAMPLE_WEIGHT,
)

unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    unet, optimizer, train_dataloader, val_dataloader
)

device = accelerator.device
vae.to(device)
text_encoder.to(device)


# =============================================================================
# VALIDATION LOOP
#
# unet.eval() deactivates AIGR hooks automatically (hook checks attn.training).
# snr_weights=None and precomputed_avg_pooled=None keep val loss as a clean,
# unweighted scalar for cross-run comparability.
# =============================================================================

def run_validation():
    unet.eval()
    total_loss = total_npo = total_recon = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values  = batch["pixel_values"].to(device)
            seg_masks     = batch["segmentation_mask"].to(device)
            pos_input_ids = batch["pos_input_ids"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)

            latents        = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            pos_embeddings = text_encoder(pos_input_ids)[0]
            neg_embeddings = text_encoder(neg_input_ids)[0]

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, T, (latents.shape[0],), device=device)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred_pos = unet(
                noisy_latents, timesteps, encoder_hidden_states=pos_embeddings
            ).sample
            noise_pred_neg = unet(
                noisy_latents, timesteps, encoder_hidden_states=neg_embeddings
            ).sample

            loss, npo_loss, recon_loss, _ = loss_fn(
                noise_pred_pos, noise_pred_neg, noise, seg_masks,
                timesteps=timesteps, T=T,
                snr_weights=None,
                precomputed_avg_pooled=None,
            )

            total_loss  += accelerator.gather(loss).mean().item()
            total_npo   += accelerator.gather(npo_loss).mean().item()
            total_recon += accelerator.gather(recon_loss).mean().item()
            n_batches   += 1

    n = max(n_batches, 1)
    return {"total": total_loss / n, "npo": total_npo / n, "recon": total_recon / n}


# =============================================================================
# WEIGHTS & BIASES LOGGING
# =============================================================================

def init_wandb():
    wandb.init(
        project="npo-artifact-removal",
        name=(f"aigr_snr_sharpmask_npo_"
              f"lr{LEARNING_RATE}_b{BETA}_amax{MASK_ALPHA_MAX}_rf{ROUTING_FLOOR}"),
        config={
            "epochs":               EPOCHS,
            "learning_rate":        LEARNING_RATE,
            "beta":                 BETA,
            "artifact_weight":      ARTIFACT_WEIGHT,
            "background_weight":    BACKGROUND_WEIGHT,
            "recon_weight":         RECON_WEIGHT,
            "model":                MODEL_NAME,
            "img_size":             IMG_SIZE,
            "batch_size":           16,
            "grad_accum_steps":     6,
            "effective_batch":      96,
            "mixed_precision":      "fp16",
            "seed":                 SEED,
            "snr_gamma":            SNR_GAMMA,
            "gauss_mean":           GAUSS_MEAN,
            "gauss_std":            GAUSS_STD,
            "gauss_floor":          GAUSS_FLOOR,
            "timestep_sampler":     "uniform",
            "mask_alpha_max":       MASK_ALPHA_MAX,
            "mask_sharpening":      "timestep_conditioned_power",
            "aigr":                 True,
            "aigr_routing":         "soft_cosine_scheduled",
            "aigr_routing_floor":   ROUTING_FLOOR,
            "clean_sample_weight":  CLEAN_SAMPLE_WEIGHT,
            "snr_applied_to":       "npo_and_recon",
        },
    )


def log_step(loss, npo_loss, recon_loss, step):
    if accelerator.is_main_process:
        wandb.log({
            "train/loss_step":  loss,
            "train/npo_step":   npo_loss,
            "train/recon_step": recon_loss,
        }, step=step)


def log_epoch(epoch, train_loss, train_npo, train_recon,
              val_loss, val_npo, val_recon, global_step):
    if accelerator.is_main_process:
        wandb.log({
            "epoch":             epoch,
            "train/loss_epoch":  train_loss,
            "train/npo_epoch":   train_npo,
            "train/recon_epoch": train_recon,
            "val/loss":          val_loss,
            "val/npo":           val_npo,
            "val/recon":         val_recon,
        }, step=global_step)

    print(
        f"[Epoch {epoch}] "
        f"Train: {train_loss:.4f} (NPO {train_npo:.4f} / Recon {train_recon:.4f}) | "
        f"Val: {val_loss:.4f} (NPO {val_npo:.4f} / Recon {val_recon:.4f})"
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

print("\n" + "=" * 70)
print("STARTING AIGR + SNR + SHARP-MASK NPO TRAINING")
print("=" * 70)
print(
    f"Epochs: {EPOCHS} | LR: {LEARNING_RATE} | Beta: {BETA} | "
    f"Recon: {RECON_WEIGHT}\n"
    f"Artifact weight: {ARTIFACT_WEIGHT} | Background weight: {BACKGROUND_WEIGHT}\n"
    f"Mask α_max: {MASK_ALPHA_MAX} | SNR γ: {SNR_GAMMA} | "
    f"Gauss μ={GAUSS_MEAN} σ={GAUSS_STD} floor={GAUSS_FLOOR}\n"
    f"AIGR routing floor: {ROUTING_FLOOR} | Clean sample weight: {CLEAN_SAMPLE_WEIGHT}\n"
    f"Effective batch: 96 (16 × 6 accum) | GC: disabled for AIGR hook compatibility"
)
print("=" * 70 + "\n")

init_wandb()
global_step = 0
best_val    = float("inf")

for epoch in range(EPOCHS):
    unet.train()
    running_loss = running_npo = running_recon = 0.0
    n_steps = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

    for batch in pbar:
        with accelerator.accumulate(unet):
            pixel_values  = batch["pixel_values"]
            seg_masks     = batch["segmentation_mask"].to(device)
            pos_input_ids = batch["pos_input_ids"]
            neg_input_ids = batch["neg_input_ids"]

            with torch.no_grad():
                latents        = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                pos_embeddings = text_encoder(pos_input_ids)[0]
                neg_embeddings = text_encoder(neg_input_ids)[0]

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, T, (latents.shape[0],), device=device)
            snr_wts   = get_snr_weights(timesteps, device)

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Compute avg_pooled once at latent resolution.
            # This tensor is shared between AIGR (routing data) and loss_fn
            # (normalisation + sharpening base) — avoids running
            # adaptive_avg_pool2d twice per batch.
            _lh, _lw        = latents.shape[2], latents.shape[3]
            avg_pooled      = F.adaptive_avg_pool2d(
                seg_masks.float().unsqueeze(1), (_lh, _lw)
            ).squeeze(1)                                     # [B, lH, lW]
            t_norm          = (timesteps.float() / float(T)).detach()

            aigr_manager.update(avg_pooled.detach(), t_norm)

            noise_pred_pos = unet(
                noisy_latents, timesteps, encoder_hidden_states=pos_embeddings
            ).sample
            noise_pred_neg = unet(
                noisy_latents, timesteps, encoder_hidden_states=neg_embeddings
            ).sample

            loss, npo_loss, recon_loss, _ = loss_fn(
                noise_pred_pos, noise_pred_neg, noise, seg_masks,
                timesteps=timesteps, T=T,
                snr_weights=snr_wts,
                precomputed_avg_pooled=avg_pooled,   # reuse — no second pool
            )

            accelerator.backward(loss)

            if not torch.isfinite(loss):
                print(
                    f"[WARNING] Non-finite loss ({loss.item():.6f}) "
                    f"at step {global_step} — skipping batch."
                )
                optimizer.zero_grad()
            else:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if not torch.isfinite(loss):
            continue

        # FIX: accumulate metrics and increment counters only on gradient-sync
        # steps (once every grad_accum_steps sub-batches).  Previously,
        # running_loss accumulated every sub-batch while n_steps only counted
        # sync steps, inflating logged train loss by grad_accum_steps (6×).
        if accelerator.sync_gradients:
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

    train_loss  = running_loss  / max(n_steps, 1)
    train_npo   = running_npo   / max(n_steps, 1)
    train_recon = running_recon / max(n_steps, 1)

    print("Running validation...")
    val_losses = run_validation()

    log_epoch(
        epoch + 1,
        train_loss, train_npo, train_recon,
        val_losses["total"], val_losses["npo"], val_losses["recon"],
        global_step,
    )

    if val_losses["total"] < best_val:
        best_val = val_losses["total"]
        accelerator.wait_for_everyone()
        accelerator.unwrap_model(unet).save_pretrained("./aigr_softmask_npo_unet_best")
        print(f"  ↳ New best val loss {best_val:.4f} — checkpoint saved.")

    if (epoch + 1) % 2 == 0:
        accelerator.wait_for_everyone()
        accelerator.unwrap_model(unet).save_pretrained(
            f"./aigr_softmask_npo_checkpoint_epoch_{epoch + 1}"
        )


# =============================================================================
# SAVE FINAL MODEL
# Only UNet is saved — VAE and text encoder are frozen and unchanged.
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE — SAVING FINAL MODEL")
print("=" * 70)

accelerator.wait_for_everyone()
accelerator.unwrap_model(unet).save_pretrained("./aigr_softmask_npo_unet_final")

print("\n✅ Done. Models saved:")
print("  Best checkpoint : ./aigr_softmask_npo_unet_best")
print("  Final UNet      : ./aigr_softmask_npo_unet_final")
print("=" * 70)

if accelerator.is_main_process:
    wandb.finish()
